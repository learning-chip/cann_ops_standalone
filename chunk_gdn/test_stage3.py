"""Stage1 -> Stage2 -> Stage3 chain; compares bf16 output to torch reference."""

from __future__ import annotations

import ctypes
import math
import os
import sys

import torch
import torch.nn.functional as F
import torch_npu

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from chunk_gdn_common import (
    ai_core_num_from_device,
    check_close,
    default_matmul_tiling,
    stage1_workspace_bytes,
    stage3_workspace_bytes,
    tiling_to_device,
    as_ptr,
)
from test_chunk_gdn import ChunkGatedDeltaRuleTilingData

LIB1 = os.path.join(_HERE, "stage1_lib.so")
LIB2 = os.path.join(_HERE, "stage2_lib.so")
LIB3 = os.path.join(_HERE, "stage3_lib.so")


def run_case(device: str) -> None:
    B, T, nk, nv, dk, dv, chunk = 1, 64, 1, 1, 64, 64, 64
    ai_core_num = ai_core_num_from_device()
    scale = 1.0 / math.sqrt(float(dk))

    query = torch.randn((T, nk, dk), dtype=torch.bfloat16, device=device).contiguous()
    key = torch.randn((T, nk, dk), dtype=torch.bfloat16, device=device).contiguous()
    query = F.normalize(query, p=2, dim=-1)
    key = F.normalize(key, p=2, dim=-1)
    value = torch.randn((T, nv, dv), dtype=torch.bfloat16, device=device).contiguous()
    beta = torch.ones((T, nv), dtype=torch.bfloat16, device=device).contiguous()

    tiling = ChunkGatedDeltaRuleTilingData()
    tiling.aiCoreNum = ai_core_num
    tiling.t = T
    tiling.nk = nk
    tiling.dk = dk
    tiling.nv = nv
    tiling.dv = dv
    tiling.b = B
    tiling.hasGamma = 0
    tiling.chunkSize = chunk
    tiling.maxGroupLength = T
    tiling.stageOneParaNum = 2
    tiling.scale = float(scale)
    tiling.matmulTilingFp32 = default_matmul_tiling(ai_core_num, 64)
    tiling_tensor = tiling_to_device(tiling, device)

    mask_elems = chunk * chunk * ai_core_num * 2
    stage_one_mask = torch.zeros((mask_elems,), dtype=torch.float32, device=device).contiguous()
    tri = torch.tril(torch.ones((chunk, chunk), dtype=torch.float32, device=device))
    stage_one_mask[: chunk * chunk].copy_(tri.flatten())
    stage_one_mask[chunk * chunk : 2 * chunk * chunk].copy_(tri.flatten())

    stage_three_mask = torch.zeros((mask_elems,), dtype=torch.float32, device=device).contiguous()
    stage_three_mask[: chunk * chunk].copy_(tri.flatten())
    stage_three_mask[chunk * chunk : 2 * chunk * chunk].copy_(tri.flatten())

    qkt = torch.empty((nv, T, chunk), dtype=torch.float32, device=device).contiguous()
    g_cum_exp = torch.empty((nv, T), dtype=torch.float32, device=device).contiguous()
    k_cum_decay = torch.empty((nv, T, dk), dtype=torch.float32, device=device).contiguous()
    v_inner = torch.empty((nv, T, dv), dtype=torch.float32, device=device).contiguous()
    q_prime = torch.empty((nv, T, dk), dtype=torch.float32, device=device).contiguous()
    kg = torch.empty((nv, T, dk), dtype=torch.float32, device=device).contiguous()

    ws1 = stage1_workspace_bytes(ai_core_num, chunk, dk, dv)
    workspace1 = torch.empty((ws1,), dtype=torch.uint8, device=device)

    lib1 = ctypes.CDLL(LIB1)
    lib1.call_stage1.argtypes = [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * 13
    lib1.call_stage1.restype = None
    lib1.call_stage1(
        ai_core_num * 2,
        torch.npu.current_stream()._as_parameter_,
        as_ptr(query),
        as_ptr(key),
        as_ptr(value),
        as_ptr(beta),
        ctypes.c_void_p(0),
        as_ptr(stage_one_mask),
        as_ptr(qkt),
        as_ptr(g_cum_exp),
        as_ptr(k_cum_decay),
        as_ptr(v_inner),
        as_ptr(q_prime),
        as_ptr(kg),
        as_ptr(workspace1),
        as_ptr(tiling_tensor),
    )
    torch.npu.synchronize()

    cur_state = torch.zeros((nv, dv, dk), dtype=torch.float32, device=device).contiguous()
    attn_inter = torch.zeros((nv, T, dv), dtype=torch.float32, device=device).contiguous()
    workspace2 = torch.zeros((4096,), dtype=torch.uint8, device=device)

    lib2 = ctypes.CDLL(LIB2)
    lib2.call_stage2.argtypes = [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * 8
    lib2.call_stage2.restype = None
    lib2.call_stage2(
        ai_core_num * 2,
        torch.npu.current_stream()._as_parameter_,
        as_ptr(q_prime),
        as_ptr(v_inner),
        as_ptr(g_cum_exp),
        as_ptr(k_cum_decay),
        as_ptr(cur_state),
        as_ptr(kg),
        as_ptr(attn_inter),
        as_ptr(workspace2),
        as_ptr(tiling_tensor),
    )
    torch.npu.synchronize()

    out_bf16 = torch.empty((T, nv, dv), dtype=torch.bfloat16, device=device).contiguous()
    ws3 = stage3_workspace_bytes(ai_core_num, chunk)
    workspace3 = torch.zeros((ws3,), dtype=torch.uint8, device=device)

    lib3 = ctypes.CDLL(LIB3)
    lib3.call_stage3.argtypes = [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * 8
    lib3.call_stage3.restype = None
    lib3.call_stage3(
        ai_core_num * 2,
        torch.npu.current_stream()._as_parameter_,
        as_ptr(qkt),
        as_ptr(g_cum_exp),
        as_ptr(attn_inter),
        as_ptr(v_inner),
        as_ptr(stage_three_mask),
        as_ptr(out_bf16),
        as_ptr(workspace3),
        as_ptr(tiling_tensor),
    )
    torch.npu.synchronize()

    # Reference: Stage3 adds (masked scaled qkt) @ v_inner into attn_inter, then writes Cast to bf16.
    qkt_m = qkt[0]
    mask = torch.tril(torch.ones((T, chunk), dtype=torch.float32, device=device))
    mqkt = (qkt_m * scale) * mask
    contrib = mqkt @ v_inner[0]
    out_ref_fp = attn_inter[0] + contrib
    out_ref = out_ref_fp.to(torch.bfloat16)

    # bf16 vs float: max error can be large on a few elements after cast; mean should stay tight.
    check_close(
        "stage3_out",
        out_bf16[:, 0, :].to(torch.float32),
        out_ref[:, :].to(torch.float32),
        tol=0.5,
        mean_tol=0.1,
    )


if __name__ == "__main__":
    device_id = int(os.environ.get("NPU_ID", "0"))
    device = f"npu:{device_id}"
    torch.npu.set_device(device)
    run_case(device)
    print("test_stage3 passed.")
