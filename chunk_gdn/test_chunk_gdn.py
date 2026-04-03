import ctypes
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

import torch_npu

from chunk_gdn_common import (
    ChunkGatedDeltaRuleTilingData,
    TCubeTiling,
    ai_core_num_from_device,
    as_ptr,
    default_matmul_tiling,
    stage1_workspace_bytes,
    stage3_workspace_bytes,
    tiling_to_device,
)


here = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(here, "chunk_gdn_lib.so")
stage1_lib_path = os.path.join(here, "stage1_lib.so")
stage2_lib_path = os.path.join(here, "stage2_lib.so")
stage3_lib_path = os.path.join(here, "stage3_lib.so")


# =========================
# Reference implementation
# =========================

MIN_ERR = 1e-3
CV_MAX_RE = 5
CV_AVER_RE = 1.5
CV_RMSE = 1.5
CV_SMALL_VAL = 2
err_threshold = 2**(-8)


def get_max_re(golden: torch.Tensor, actual: torch.Tensor):
    abs_error = torch.abs(actual - golden) / (torch.abs(golden) + MIN_ERR)
    return torch.max(abs_error.flatten())


def get_avg_re(golden: torch.Tensor, actual: torch.Tensor):
    abs_error = torch.abs(actual - golden) / (torch.abs(golden) + MIN_ERR)
    return torch.mean(abs_error)


def get_rmse(golden: torch.Tensor, actual: torch.Tensor):
    sqr_err = torch.pow((actual - golden), 2)
    return torch.sqrt(torch.mean(sqr_err))


def get_smra(golden: torch.Tensor, actual: torch.Tensor):
    abs_A = torch.abs(golden)
    mask_A = abs_A < 2**(-10)
    num_a = torch.sum(mask_A).item()

    abs_B = torch.abs(golden - actual)
    mask_B = abs_B > 1e-16
    num_b = torch.sum(mask_A & mask_B).item()

    return num_b / num_a if num_a > 0 else 0


def get_eb(golden_high_type: torch.Tensor, actual: torch.Tensor):
    golden_nmax = torch.clamp(torch.abs(golden_high_type), min=1)
    actual_error = actual - golden_high_type
    return torch.mean(actual_error / golden_nmax)


def compare_cv(golden: torch.Tensor, golden_high_type: torch.Tensor, actual: torch.Tensor, name=None):
    golden = golden.to(torch.float32)
    golden_high_type = golden_high_type.to(torch.float32)
    actual = actual.to(torch.float32)

    max_re_npu = get_max_re(golden, actual)
    max_re_high_type = get_max_re(golden, golden_high_type)
    avg_re_npu = get_avg_re(golden, actual)
    avg_re_high_type = get_avg_re(golden, golden_high_type)
    rmse_npu = get_rmse(golden, actual)
    rmse_high_type = get_rmse(golden, golden_high_type)
    smra_npu = get_smra(golden, actual)
    smra_high_type = get_smra(golden, golden_high_type)

    max_re_rate = max_re_npu / max(max_re_high_type, err_threshold)
    avg_re_rate = avg_re_npu / max(avg_re_high_type, err_threshold)
    rmse_rate = rmse_npu / max(rmse_high_type, err_threshold)
    smra_rate = smra_npu / max(smra_high_type, err_threshold)

    EB = get_eb(golden_high_type, actual)
    _ = EB  # kept for debugging parity with upstream

    result = (max_re_rate < CV_MAX_RE) and (avg_re_rate < CV_AVER_RE) and (rmse_rate < CV_RMSE)
    result = result and smra_rate < CV_SMALL_VAL

    if name is not None:
        print(
            f"compare[{name}]: "
            f"max_re_rate={float(max_re_rate):.3f} avg_re_rate={float(avg_re_rate):.3f} "
            f"rmse_rate={float(rmse_rate):.3f} smra_rate={float(smra_rate):.3f} "
            f"(golden_high=max_re={float(max_re_high_type):.3e})"
        )

    return bool(result)


def chunk_gated_delta_rule_native(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,  # kept for signature compatibility
):
    initial_dtype = query.dtype
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]

    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]

    pad_size = (chunk_size - num_heads % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))

    tot_heads = num_heads + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # reshape to chunks along head dimension
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    last_recurrent_state = (
        torch.zeros(batch_size, sequence_length, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    for i in range(0, tot_heads // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :num_heads]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def chunk_gated_delta_rule_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    cu_seqlens=None,
):
    # Mirrors upstream golden logic, but runs entirely via Torch ops.
    num_heads = q.shape[-2]
    num_value_heads = v.shape[-2]

    if num_value_heads // num_heads > 1:
        q = q.repeat_interleave(num_value_heads // num_heads, dim=2)
        k = k.repeat_interleave(num_value_heads // num_heads, dim=2)

    batch_size = initial_state.shape[0]
    core_attn_out = []
    last_recurrent_state = torch.empty_like(initial_state)

    for b_idx in range(batch_size):
        start, end = cu_seqlens[b_idx], cu_seqlens[b_idx + 1]
        cur_q = q[:, start:end, ...]
        cur_k = k[:, start:end, ...]
        cur_v = v[:, start:end, ...]
        cur_g = g[:, start:end, ...]
        cur_beta = beta[:, start:end, ...]
        cur_state = initial_state[b_idx].unsqueeze(0)

        cur_core_attn_out, cur_last_recurrent_state = chunk_gated_delta_rule_native(
            query=cur_q,
            key=cur_k,
            value=cur_v,
            g=cur_g,
            beta=cur_beta,
            initial_state=cur_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        core_attn_out.append(cur_core_attn_out)
        last_recurrent_state[b_idx] = cur_last_recurrent_state

    tar_dtype = core_attn_out[0].dtype
    tar_device = core_attn_out[0].device
    tar_shape = list(core_attn_out[0].shape)
    tar_shape[1] = cu_seqlens[-1]
    final_cor_attn_out = torch.empty(tar_shape, dtype=tar_dtype, device=tar_device)

    for b_idx in range(batch_size):
        start, end = cu_seqlens[b_idx], cu_seqlens[b_idx + 1]
        final_cor_attn_out[:, start:end, ...] = core_attn_out[b_idx]

    return final_cor_attn_out, last_recurrent_state


def cgdr_golden_native(q, k, v, g, beta, scale, initial_state, actual_seq_lengths, use_float64=False):
    # Golden: compute float32 reference (by forcing query/key/value/beta/g to float32).
    cu_seqlens = F.pad(actual_seq_lengths, (1, 0)).cumsum(dim=0)

    # Force computation dtype to float32 (matching upstream golden default).
    q_ = q.to(torch.float32)
    k_ = k.to(torch.float32)
    v_ = v.to(torch.float32)
    g_ = g.to(torch.float32)
    beta_ = beta.to(torch.float32)

    o_golden, state_golden = chunk_gated_delta_rule_npu(
        q_.unsqueeze(0),
        k_.unsqueeze(0),
        v_.unsqueeze(0),
        g_.unsqueeze(0),
        beta_.unsqueeze(0),
        scale=scale,
        initial_state=initial_state.transpose(-1, -2).clone().to(v_.dtype),
        cu_seqlens=cu_seqlens,
    )
    o_golden = o_golden[0]
    state_golden = state_golden.transpose(-1, -2)
    return o_golden.to(torch.float32), state_golden.to(torch.float32)


def cgdr_benchmark_bf16(q, k, v, g, beta, scale, initial_state, actual_seq_lengths):
    # High-type: compute using bf16 inputs so the reference output includes bf16 rounding,
    # then convert to float32 for compare_cv.
    cu_seqlens = F.pad(actual_seq_lengths, (1, 0)).cumsum(dim=0)

    o_bench, state_bench = chunk_gated_delta_rule_npu(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        g.unsqueeze(0),
        beta.unsqueeze(0),
        scale=scale,
        initial_state=initial_state.transpose(-1, -2).clone(),
        cu_seqlens=cu_seqlens,
    )
    o_bench = o_bench[0].to(torch.float32)
    state_bench = state_bench.transpose(-1, -2).to(torch.float32)
    return o_bench, state_bench


# =========================
# Kernel call + tiling data
# =========================


_STAGE_LIBS = None


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def padded_seq_len(seq_len: int, chunk_size: int) -> int:
    return _ceil_div(seq_len, chunk_size) * chunk_size


def build_stage_tiling(
    *,
    ai_core_num: int,
    seq_len: int,
    nk: int,
    nv: int,
    dk: int,
    dv: int,
    has_gamma: int,
    chunk_size: int,
    scale: float,
) -> ChunkGatedDeltaRuleTilingData:
    tiling = ChunkGatedDeltaRuleTilingData()
    tiling.aiCoreNum = ai_core_num
    tiling.t = seq_len
    tiling.nk = nk
    tiling.dk = dk
    tiling.nv = nv
    tiling.dv = dv
    tiling.b = 1
    tiling.hasGamma = has_gamma
    tiling.chunkSize = chunk_size
    tiling.maxGroupLength = padded_seq_len(seq_len, chunk_size)
    tiling.interWorkspaceSz = 0
    tiling.stageWorkspaceSz = 0
    tiling.stageOneParaNum = 2
    tiling.scale = float(scale)
    tiling.matmulTilingFp32 = default_matmul_tiling(ai_core_num, max(chunk_size, dk, dv))
    return tiling


def build_stage_masks(ai_core_num: int, chunk_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    mask_elems = chunk_size * chunk_size * ai_core_num * 2
    tri = torch.tril(torch.ones((chunk_size, chunk_size), dtype=torch.float32, device=device))

    stage_one_mask = torch.zeros((mask_elems,), dtype=torch.float32, device=device).contiguous()
    stage_three_mask = torch.zeros((mask_elems,), dtype=torch.float32, device=device).contiguous()
    flat = tri.flatten()

    stage_one_mask[: chunk_size * chunk_size].copy_(flat)
    stage_one_mask[chunk_size * chunk_size : 2 * chunk_size * chunk_size].copy_(flat)
    stage_three_mask[: chunk_size * chunk_size].copy_(flat)
    stage_three_mask[chunk_size * chunk_size : 2 * chunk_size * chunk_size].copy_(flat)
    return stage_one_mask, stage_three_mask


def load_stage_libs():
    global _STAGE_LIBS
    if _STAGE_LIBS is not None:
        return _STAGE_LIBS

    lib1 = ctypes.CDLL(stage1_lib_path)
    lib1.call_stage1.argtypes = [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * 13
    lib1.call_stage1.restype = None

    lib2 = ctypes.CDLL(stage2_lib_path)
    lib2.call_stage2.argtypes = [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * 8
    lib2.call_stage2.restype = None

    lib3 = ctypes.CDLL(stage3_lib_path)
    lib3.call_stage3.argtypes = [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * 8
    lib3.call_stage3.restype = None

    _STAGE_LIBS = (lib1, lib2, lib3)
    return _STAGE_LIBS


class StagedChunkGDNRunner:
    def __init__(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        actual_seq_lengths: torch.Tensor,
        chunk_size: int,
        ai_core_num: int,
    ):
        self.q = q
        self.k = k
        self.v = v
        self.g = g
        self.beta = beta
        self.scale = scale
        self.initial_state = initial_state
        self.actual_seq_lengths = actual_seq_lengths
        self.chunk_size = int(chunk_size)
        self.ai_core_num = int(ai_core_num)
        self.device = q.device
        self.stream = torch.npu.current_stream()._as_parameter_
        self.lib1, self.lib2, self.lib3 = load_stage_libs()
        self.out = torch.empty_like(v)
        self.final_state = torch.empty_like(initial_state)
        self._seq_buffers = []
        self._init_sequence_buffers()

    def _init_sequence_buffers(self) -> None:
        offset = 0
        B = int(self.actual_seq_lengths.numel())
        nk = int(self.q.shape[1])
        nv = int(self.v.shape[1])
        dk = int(self.q.shape[2])
        dv = int(self.v.shape[2])
        has_gamma = 1

        for bid in range(B):
            seq_len = int(self.actual_seq_lengths[bid].item())
            sp = padded_seq_len(seq_len, self.chunk_size)
            tiling = build_stage_tiling(
                ai_core_num=self.ai_core_num,
                seq_len=seq_len,
                nk=nk,
                nv=nv,
                dk=dk,
                dv=dv,
                has_gamma=has_gamma,
                chunk_size=self.chunk_size,
                scale=self.scale,
            )
            tiling_tensor = tiling_to_device(tiling, str(self.device))
            stage_one_mask, stage_three_mask = build_stage_masks(self.ai_core_num, self.chunk_size, str(self.device))

            q_seq = self.q[offset : offset + seq_len].contiguous()
            k_seq = self.k[offset : offset + seq_len].contiguous()
            v_seq = self.v[offset : offset + seq_len].contiguous()
            g_seq = self.g[offset : offset + seq_len].contiguous()
            beta_seq = self.beta[offset : offset + seq_len].contiguous()
            init_state_fp32 = self.initial_state[bid].to(torch.float32).contiguous()

            self._seq_buffers.append(
                {
                    "offset": offset,
                    "seq_len": seq_len,
                    "q": q_seq,
                    "k": k_seq,
                    "v": v_seq,
                    "g": g_seq,
                    "beta": beta_seq,
                    "tiling": tiling_tensor,
                    "stage_one_mask": stage_one_mask,
                    "stage_three_mask": stage_three_mask,
                    "qkt": torch.empty((nv, sp, self.chunk_size), dtype=torch.float32, device=self.device).contiguous(),
                    "g_cum_exp": torch.empty((nv, sp), dtype=torch.float32, device=self.device).contiguous(),
                    "k_cum_decay": torch.empty((nv, sp, dk), dtype=torch.float32, device=self.device).contiguous(),
                    "v_inner": torch.empty((nv, sp, dv), dtype=torch.float32, device=self.device).contiguous(),
                    "q_prime": torch.empty((nv, sp, dk), dtype=torch.float32, device=self.device).contiguous(),
                    "kg": torch.empty((nv, sp, dk), dtype=torch.float32, device=self.device).contiguous(),
                    "workspace1": torch.empty(
                        (stage1_workspace_bytes(self.ai_core_num, self.chunk_size, dk, dv),),
                        dtype=torch.uint8,
                        device=self.device,
                    ),
                    "workspace2": torch.zeros((4096,), dtype=torch.uint8, device=self.device),
                    "cur_state": init_state_fp32.clone(),
                    "initial_state_fp32": init_state_fp32,
                    "attn_inter": torch.empty((nv, sp, dv), dtype=torch.float32, device=self.device).contiguous(),
                    "workspace3": torch.zeros(
                        (stage3_workspace_bytes(self.ai_core_num, self.chunk_size),),
                        dtype=torch.uint8,
                        device=self.device,
                    ),
                    "out_seq": torch.empty((seq_len, nv, dv), dtype=torch.bfloat16, device=self.device).contiguous(),
                }
            )
            offset += seq_len

    def run(self) -> tuple[torch.Tensor, torch.Tensor]:
        for bid, buf in enumerate(self._seq_buffers):
            buf["cur_state"].copy_(buf["initial_state_fp32"])
            buf["attn_inter"].zero_()

            self.lib1.call_stage1(
                self.ai_core_num,
                self.stream,
                as_ptr(buf["q"]),
                as_ptr(buf["k"]),
                as_ptr(buf["v"]),
                as_ptr(buf["beta"]),
                as_ptr(buf["g"]),
                as_ptr(buf["stage_one_mask"]),
                as_ptr(buf["qkt"]),
                as_ptr(buf["g_cum_exp"]),
                as_ptr(buf["k_cum_decay"]),
                as_ptr(buf["v_inner"]),
                as_ptr(buf["q_prime"]),
                as_ptr(buf["kg"]),
                as_ptr(buf["workspace1"]),
                as_ptr(buf["tiling"]),
            )
            self.lib2.call_stage2(
                self.ai_core_num,
                self.stream,
                as_ptr(buf["q_prime"]),
                as_ptr(buf["v_inner"]),
                as_ptr(buf["g_cum_exp"]),
                as_ptr(buf["k_cum_decay"]),
                as_ptr(buf["cur_state"]),
                as_ptr(buf["kg"]),
                as_ptr(buf["attn_inter"]),
                as_ptr(buf["workspace2"]),
                as_ptr(buf["tiling"]),
            )
            self.lib3.call_stage3(
                self.ai_core_num,
                self.stream,
                as_ptr(buf["qkt"]),
                as_ptr(buf["g_cum_exp"]),
                as_ptr(buf["attn_inter"]),
                as_ptr(buf["v_inner"]),
                as_ptr(buf["stage_three_mask"]),
                as_ptr(buf["out_seq"]),
                as_ptr(buf["workspace3"]),
                as_ptr(buf["tiling"]),
            )
            self.out[buf["offset"] : buf["offset"] + buf["seq_len"]].copy_(buf["out_seq"])
            self.final_state[bid].copy_(buf["cur_state"].to(torch.bfloat16))
        return self.out, self.final_state


def build_tiling_and_workspace(
    *,
    ai_core_num: int,
    B: int,
    T: int,
    nk: int,
    nv: int,
    dk: int,
    dv: int,
    has_gamma: int,
    chunk_size: int,
    scale: float,
):
    # Matches op_host tiling.cpp constants:
    # c=64, p=P_NUM=2, STAGE_ONE_TWO=2, STAGE_ONE_THREE=3, stageOneParaNum=STAGE_ONE_TWO=2
    c = int(chunk_size)
    p = 2
    stage_one_two = 2
    stage_one_three = 3
    mask_num = 4  # MASK_NUM
    stage_one_para_num = stage_one_two

    max_group_len = p * ai_core_num * c
    size_high = 4  # float32

    # interWorkspaceSz
    s = max_group_len
    inter = (
        size_high * nv * s  # gCumExp
        + size_high * nv * s * dk  # kCumDecay
        + size_high * nv * s * dv  # vInner
        + size_high * nv * s * dk  # qPrime
        + size_high * nv * s * dv  # attnInter
        + size_high * nv * s * dk  # kg
        + size_high * nv * s * c  # qkt
        + size_high * B * nv * dv * dk  # highState
        + size_high * c * c * ai_core_num * mask_num  # mask (stageOne+stageThree)
    )

    # stageWorkspaceSz
    stage_ws = size_high * c * (stage_one_two * c + stage_one_three * dk + dv) * stage_one_para_num * ai_core_num

    # tiling struct
    tiling = ChunkGatedDeltaRuleTilingData()
    tiling.aiCoreNum = ai_core_num
    tiling.t = T
    tiling.nk = nk
    tiling.dk = dk
    tiling.nv = nv
    tiling.dv = dv
    tiling.b = B
    tiling.hasGamma = has_gamma
    tiling.chunkSize = c
    tiling.maxGroupLength = max_group_len
    tiling.interWorkspaceSz = inter
    tiling.stageWorkspaceSz = stage_ws
    tiling.stageOneParaNum = stage_one_para_num
    tiling.scale = float(scale)

    # Populate matmul tiling with a "safe-ish" set of values.
    # The kernel's stage MT path uses these fields during Init().
    #
    # Upstream op_host does: mm_.GetTiling(...), then overrides a few fields
    # (dbL0C/stepKa/stepKb/depthA1/depthB1/stepM/stepN). We mirror that and
    # also fill the most commonly used shape/layout fields so we don't leave
    # large portions of TCubeTiling as zeros.
    tiling.matmulTilingFp32 = TCubeTiling()
    mm = tiling.matmulTilingFp32

    # Base shapes for MATMUL_BASE_M/N/K in op_host.
    baseM = 128
    baseN = 128
    baseK = 128
    mm.baseM = baseM
    mm.baseN = baseN
    mm.baseK = baseK

    mm.M = baseM
    mm.N = baseN
    mm.Ka = baseK
    mm.Kb = baseK
    mm.usedCoreNum = ai_core_num

    # Per-core M/N/K must match the working Stage1/2/3 standalone tiling
    # (`default_matmul_tiling` in chunk_gdn_common): splitting M across cores here
    # breaks StageOneMT / StageTwoMT / StageThreeMT Init and can fault at runtime.
    mm.singleCoreM = baseM
    mm.singleCoreN = baseN
    mm.singleCoreK = baseK

    # Mirrors op_host overrides (after GetTiling()).
    mm.dbL0C = 1
    mm.stepKa = 1
    mm.stepKb = 1
    mm.depthA1 = 1
    mm.depthB1 = 1
    mm.stepM = 1
    mm.stepN = 1

    # Sizes used for some copy/transpose paths (bytes for FP32).
    mm.shareL0CSize = baseM * baseN * 4  # 128*128*4
    mm.transLength = mm.shareL0CSize

    # Keep sharing/cache/simple defaults consistent with op_host.
    mm.shareMode = 0
    mm.shareUbSize = 0
    mm.shareL1Size = 0
    mm.iterateOrder = 0

    # Batch-related fields: matmul is SetDim(1) in op_host.
    mm.batchM = 1
    mm.batchN = 1
    mm.singleBatchM = 1
    mm.singleBatchN = 1

    # Bias disabled.
    mm.isBias = 0

    tiling.matmulTilingFp32 = mm

    tiling_bytes = bytes(tiling)
    tiling_size = len(tiling_bytes)

    # `workspaceGM` is interpreted by AscendC's `GetUserWorkspace(workspaceGM)`,
    # which applies a 16MB "system workspace" offset internally.
    # Allocate that prefix so the device-side pointer math stays in-bounds.
    system_ws = 16 * 1024 * 1024
    workspace_size = system_ws + inter + stage_ws

    return tiling_bytes, tiling_size, workspace_size


def run_one_case(params):
    B, seqlen, nk, nv, dk, dv, chunk_size = params

    T = B * seqlen
    scale = 1.0 / math.sqrt(float(dk))

    q = torch.rand((T, nk, dk), dtype=torch.bfloat16, device=device).contiguous()
    k = torch.rand((T, nk, dk), dtype=torch.bfloat16, device=device).contiguous()
    v = torch.rand((T, nv, dv), dtype=torch.bfloat16, device=device).contiguous()
    # g is float32 in golden
    g = (torch.rand((T, nv), dtype=torch.float32, device=device) * -1.0).contiguous()
    beta = torch.rand((T, nv), dtype=torch.bfloat16, device=device).contiguous()
    # Normalize q/k like upstream golden.
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)
    initial_state = torch.rand((B, nv, dv, dk), dtype=torch.bfloat16, device=device).contiguous()
    actual_seq_lengths = torch.full((B,), int(seqlen), dtype=torch.int32, device=device)

    # ====================
    # Golden (native)
    # ====================
    o_golden, state_golden = cgdr_golden_native(q, k, v, g, beta, scale, initial_state, actual_seq_lengths)
    o_bench, state_bench = cgdr_benchmark_bf16(q, k, v, g, beta, scale, initial_state, actual_seq_lengths)

    # ====================
    # Kernel call
    # ====================
    runner = StagedChunkGDNRunner(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        actual_seq_lengths=actual_seq_lengths,
        chunk_size=chunk_size,
        ai_core_num=ai_core_num,
    )
    out, final_state = runner.run()
    torch.npu.synchronize()

    # ====================
    # Compare
    # ====================
    out_diff = torch.abs(out.to(torch.float32) - o_bench)
    state_diff = torch.abs(final_state.to(torch.float32) - state_bench)
    max_diff_o = out_diff.max().item()
    mean_diff_o = out_diff.mean().item()
    max_diff_s = state_diff.max().item()
    mean_diff_s = state_diff.mean().item()
    print(
        f"staged compare: out(max={max_diff_o:.6f}, mean={mean_diff_o:.6f}) "
        f"state(max={max_diff_s:.6f}, mean={mean_diff_s:.6f})"
    )

    if max_diff_o > 0.35 or mean_diff_o > 0.05 or max_diff_s > 0.20 or mean_diff_s > 0.05:
        raise AssertionError(
            "chunk_gdn failed: "
            f"out(max={max_diff_o}, mean={mean_diff_o}) "
            f"state(max={max_diff_s}, mean={mean_diff_s})"
        )

    print(f"PASS: B={B} seqlen={seqlen} nk={nk} nv={nv} dk={dk} dv={dv} chunk={chunk_size}")


if __name__ == "__main__":
    # Device selection: override with `NPU_ID=...` if needed.
    # Example: torch.npu.set_device("npu:7") if NPU 7 is free.
    device_id = int(os.environ.get("NPU_ID", "0"))
    device = f"npu:{device_id}"
    torch.npu.set_device(device)

    ai_core_num = ai_core_num_from_device()

    # Mirrors upstream pytest paramset (but we only run correctness, not performance).
    test_params = [
        (1, 64, 1, 1, 64, 64, 64),
        (1, 4096, 4, 4, 64, 64, 64),
    ]

    for params in test_params:
        run_one_case(params)

    print("chunk_gdn all tests passed.")

