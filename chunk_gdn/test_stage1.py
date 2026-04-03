"""On-device test for standalone Stage1 (`stage1_lib.so`)."""

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
    ChunkGatedDeltaRuleTilingData,
    default_matmul_tiling,
    stage1_workspace_bytes,
    tiling_to_device,
    as_ptr,
)

LIB_PATH = os.path.join(_HERE, "stage1_lib.so")


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

    qkt = torch.empty((nv, T, chunk), dtype=torch.float32, device=device).contiguous()
    g_cum_exp = torch.empty((nv, T), dtype=torch.float32, device=device).contiguous()
    k_cum_decay = torch.empty((nv, T, dk), dtype=torch.float32, device=device).contiguous()
    v_inner = torch.empty((nv, T, dv), dtype=torch.float32, device=device).contiguous()
    q_prime = torch.empty((nv, T, dk), dtype=torch.float32, device=device).contiguous()
    kg = torch.empty((nv, T, dk), dtype=torch.float32, device=device).contiguous()

    workspace = torch.empty((stage1_workspace_bytes(ai_core_num, chunk, dk, dv),), dtype=torch.uint8, device=device)

    lib = ctypes.CDLL(LIB_PATH)
    lib.call_stage1.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.call_stage1.restype = None

    lib.call_stage1(
        ai_core_num,
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
        as_ptr(workspace),
        as_ptr(tiling_tensor),
    )
    torch.npu.synchronize()

    q_ref = query[:, 0, :].to(torch.float32)
    k_ref = key[:, 0, :].to(torch.float32)
    qk_ref = q_ref @ k_ref.transpose(0, 1)

    float_size = 4
    off_gbk = ai_core_num * chunk * dk * float_size
    off_kk = ai_core_num * chunk * chunk * float_size
    off_vbeta = ai_core_num * chunk * dv * float_size
    off_attn = ai_core_num * chunk * chunk * float_size
    off_qcont = off_gbk + off_kk + off_vbeta + off_attn
    off_kcont = off_qcont + ai_core_num * chunk * dk * float_size
    per_core = chunk * dk * float_size
    q_cont = workspace[off_qcont : off_qcont + per_core].view(torch.float32).view(chunk, dk)
    k_cont = workspace[off_kcont : off_kcont + per_core].view(torch.float32).view(chunk, dk)

    check_close("q_cont(ws)", q_cont, q_ref, tol=3e-3)
    check_close("k_cont(ws)", k_cont, k_ref, tol=3e-3)
    check_close("qkt", qkt[0], qk_ref, tol=3e-3)
    check_close("q_prime", q_prime[0], q_ref * scale, tol=3e-3)
    check_close("kg(no_gamma)", kg[0], k_ref, tol=3e-3)
    check_close("g_cum_exp(no_gamma)", g_cum_exp[0], torch.zeros_like(g_cum_exp[0]), tol=3e-3)
    assert torch.isfinite(k_cum_decay).all() and torch.isfinite(v_inner).all()


if __name__ == "__main__":
    device_id = int(os.environ.get("NPU_ID", "0"))
    device = f"npu:{device_id}"
    torch.npu.set_device(device)
    run_case(device)
    print("test_stage1 passed.")
