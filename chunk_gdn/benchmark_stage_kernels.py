"""
Benchmark standalone Stage1 / Stage2 / Stage3 shared libraries across a broader shape matrix.

For each benchmark case:
- first validate the end-to-end staged pipeline against the torch reference
- then time Stage1 / Stage2 / Stage3 in isolation

For `B > 1`, the stage benchmark replays the same per-sequence launch `B` times per timed
iteration. This preserves batch semantics for throughput accounting while using the same
correctly-working per-stage kernels.
"""

from __future__ import annotations

import csv
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
    ChunkGatedDeltaRuleTilingData,
    default_matmul_tiling,
    stage1_workspace_bytes,
    stage3_workspace_bytes,
    tiling_to_device,
    as_ptr,
)
from test_chunk_gdn import StagedChunkGDNRunner, cgdr_benchmark_bf16

LIB1 = os.path.join(_HERE, "stage1_lib.so")
LIB2 = os.path.join(_HERE, "stage2_lib.so")
LIB3 = os.path.join(_HERE, "stage3_lib.so")

BENCHMARK_CASES = [
    {"name": "b1_s4096_h4_d64_c64", "B": 1, "seqlen": 4096, "nk": 4, "nv": 4, "dk": 64, "dv": 64, "chunk": 64},
    {"name": "b1_s8192_h16_d64_c64", "B": 1, "seqlen": 8192, "nk": 16, "nv": 16, "dk": 64, "dv": 64, "chunk": 64},
    {"name": "b1_s4096_h32_d64_c64", "B": 1, "seqlen": 4096, "nk": 32, "nv": 32, "dk": 64, "dv": 64, "chunk": 64},
    {"name": "b1_s2048_h64_d64_c64", "B": 1, "seqlen": 2048, "nk": 64, "nv": 64, "dk": 64, "dv": 64, "chunk": 64},
    {"name": "b8_s2048_h16_d64_c64", "B": 8, "seqlen": 2048, "nk": 16, "nv": 16, "dk": 64, "dv": 64, "chunk": 64},
    {"name": "b32_s1024_h16_d64_c64", "B": 32, "seqlen": 1024, "nk": 16, "nv": 16, "dk": 64, "dv": 64, "chunk": 64},
    {"name": "b128_s256_h16_d64_c64", "B": 128, "seqlen": 256, "nk": 16, "nv": 16, "dk": 64, "dv": 64, "chunk": 64},
    {"name": "b8_s1024_h32_d64_c64", "B": 8, "seqlen": 1024, "nk": 32, "nv": 32, "dk": 64, "dv": 64, "chunk": 64},
]


def benchmark_with_events(fn, warmup_iters: int = 5, benchmark_iters: int = 20) -> float:
    start_events = [torch.npu.Event(enable_timing=True) for _ in range(benchmark_iters)]
    end_events = [torch.npu.Event(enable_timing=True) for _ in range(benchmark_iters)]
    for _ in range(warmup_iters):
        fn()
    torch.npu.synchronize()
    for i in range(benchmark_iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.npu.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return float(sum(times_ms) / len(times_ms))


def ms_to_tflops_per_s(flops: float, ms: float) -> float:
    if ms <= 0:
        return float("nan")
    return flops / (ms * 1e-3) / 1e12


def ms_to_operand_gibs(operand_bytes: int, ms: float) -> float:
    if ms <= 0:
        return float("nan")
    return operand_bytes / (ms * 1e-3) / (1024.0**3)


def estimate_stage1_flops(T: int, nv: int, dk: int, dv: int, chunk: int) -> float:
    """Heuristic MACs×2 for Stage1 (chunk QK, decay, inner products)."""
    nc = (T + chunk - 1) // chunk
    return 2.0 * nv * nc * (chunk * chunk * dk + chunk * dk * dv + chunk * dv * dk)


def estimate_stage2_flops(T: int, nv: int, dk: int, dv: int) -> float:
    """Heuristic for recurrent state + attn_inter matmul chains along T."""
    return 2.0 * nv * T * (dk * dv * 6 + dk * dk * 2)


def estimate_stage3_flops(T: int, nv: int, dv: int, chunk: int) -> float:
    """Masked qkt @ v style contraction (length T, chunk tile)."""
    return 2.0 * nv * T * chunk * dv * 4


def nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def make_tiling(
    *,
    ai_core_num: int,
    B: int,
    T: int,
    nk: int,
    nv: int,
    dk: int,
    dv: int,
    chunk: int,
) -> ChunkGatedDeltaRuleTilingData:
    scale = 1.0 / math.sqrt(float(dk))
    tiling = ChunkGatedDeltaRuleTilingData()
    tiling.aiCoreNum = ai_core_num
    tiling.t = T
    tiling.nk = nk
    tiling.dk = dk
    tiling.nv = nv
    tiling.dv = dv
    tiling.b = B
    tiling.hasGamma = 1
    tiling.chunkSize = chunk
    tiling.maxGroupLength = T
    tiling.stageOneParaNum = 2
    tiling.scale = float(scale)
    tiling.matmulTilingFp32 = default_matmul_tiling(ai_core_num, max(chunk, dk, dv))
    return tiling


def make_masks(chunk: int, ai_core_num: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    mask_elems = chunk * chunk * ai_core_num * 2
    tri = torch.tril(torch.ones((chunk, chunk), dtype=torch.float32, device=device)).flatten()
    stage_one_mask = torch.zeros((mask_elems,), dtype=torch.float32, device=device).contiguous()
    stage_three_mask = torch.zeros((mask_elems,), dtype=torch.float32, device=device).contiguous()
    stage_one_mask[: chunk * chunk].copy_(tri)
    stage_one_mask[chunk * chunk : 2 * chunk * chunk].copy_(tri)
    stage_three_mask[: chunk * chunk].copy_(tri)
    stage_three_mask[chunk * chunk : 2 * chunk * chunk].copy_(tri)
    return stage_one_mask, stage_three_mask


def validate_case(case: dict, device: str, ai_core_num: int) -> dict:
    B, seqlen = case["B"], case["seqlen"]
    nk, nv, dk, dv, chunk = case["nk"], case["nv"], case["dk"], case["dv"], case["chunk"]
    T = B * seqlen
    scale = 1.0 / math.sqrt(float(dk))

    q = torch.rand((T, nk, dk), dtype=torch.bfloat16, device=device).contiguous()
    k = torch.rand((T, nk, dk), dtype=torch.bfloat16, device=device).contiguous()
    v = torch.rand((T, nv, dv), dtype=torch.bfloat16, device=device).contiguous()
    g = (torch.rand((T, nv), dtype=torch.float32, device=device) * -1.0).contiguous()
    beta = torch.rand((T, nv), dtype=torch.bfloat16, device=device).contiguous()
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)
    initial_state = torch.rand((B, nv, dv, dk), dtype=torch.bfloat16, device=device).contiguous()
    actual_seq_lengths = torch.full((B,), int(seqlen), dtype=torch.int32, device=device)

    runner = StagedChunkGDNRunner(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        actual_seq_lengths=actual_seq_lengths,
        chunk_size=chunk,
        ai_core_num=ai_core_num,
    )
    out, final_state = runner.run()
    torch.npu.synchronize()

    ref_out, ref_state = cgdr_benchmark_bf16(q, k, v, g, beta, scale, initial_state, actual_seq_lengths)
    out_diff = (out.to(torch.float32) - ref_out).abs()
    state_diff = (final_state.to(torch.float32) - ref_state).abs()

    return {
        "out_max_abs": out_diff.max().item(),
        "out_mean_abs": out_diff.mean().item(),
        "state_max_abs": state_diff.max().item(),
        "state_mean_abs": state_diff.mean().item(),
    }


def prepare_representative_sequence(case: dict, device: str, ai_core_num: int) -> dict:
    seqlen = case["seqlen"]
    nk, nv, dk, dv, chunk = case["nk"], case["nv"], case["dk"], case["dv"], case["chunk"]
    scale = 1.0 / math.sqrt(float(dk))

    tiling = make_tiling(
        ai_core_num=ai_core_num,
        B=1,
        T=seqlen,
        nk=nk,
        nv=nv,
        dk=dk,
        dv=dv,
        chunk=chunk,
    )
    tiling_tensor = tiling_to_device(tiling, device)
    tiling_nbytes = tiling_tensor.numel() * tiling_tensor.element_size()
    stage_one_mask, stage_three_mask = make_masks(chunk, ai_core_num, device)

    query = torch.randn((seqlen, nk, dk), dtype=torch.bfloat16, device=device).contiguous()
    key = torch.randn((seqlen, nk, dk), dtype=torch.bfloat16, device=device).contiguous()
    query = F.normalize(query, p=2, dim=-1)
    key = F.normalize(key, p=2, dim=-1)
    value = torch.randn((seqlen, nv, dv), dtype=torch.bfloat16, device=device).contiguous()
    beta = torch.rand((seqlen, nv), dtype=torch.bfloat16, device=device).contiguous()
    gamma = (torch.rand((seqlen, nv), dtype=torch.float32, device=device) * -1.0).contiguous()

    qkt = torch.empty((nv, seqlen, chunk), dtype=torch.float32, device=device).contiguous()
    g_cum_exp = torch.empty((nv, seqlen), dtype=torch.float32, device=device).contiguous()
    k_cum_decay = torch.empty((nv, seqlen, dk), dtype=torch.float32, device=device).contiguous()
    v_inner = torch.empty((nv, seqlen, dv), dtype=torch.float32, device=device).contiguous()
    q_prime = torch.empty((nv, seqlen, dk), dtype=torch.float32, device=device).contiguous()
    kg = torch.empty((nv, seqlen, dk), dtype=torch.float32, device=device).contiguous()

    workspace1 = torch.empty((stage1_workspace_bytes(ai_core_num, chunk, dk, dv),), dtype=torch.uint8, device=device)
    workspace2 = torch.zeros((4096,), dtype=torch.uint8, device=device)
    workspace3 = torch.zeros((stage3_workspace_bytes(ai_core_num, chunk),), dtype=torch.uint8, device=device)
    cur_state = torch.zeros((nv, dv, dk), dtype=torch.float32, device=device).contiguous()
    attn_inter = torch.zeros((nv, seqlen, dv), dtype=torch.float32, device=device).contiguous()
    out_bf16 = torch.empty((seqlen, nv, dv), dtype=torch.bfloat16, device=device).contiguous()

    return {
        "scale": scale,
        "tiling_tensor": tiling_tensor,
        "tiling_nbytes": tiling_nbytes,
        "query": query,
        "key": key,
        "value": value,
        "beta": beta,
        "gamma": gamma,
        "stage_one_mask": stage_one_mask,
        "stage_three_mask": stage_three_mask,
        "qkt": qkt,
        "g_cum_exp": g_cum_exp,
        "k_cum_decay": k_cum_decay,
        "v_inner": v_inner,
        "q_prime": q_prime,
        "kg": kg,
        "workspace1": workspace1,
        "workspace2": workspace2,
        "workspace3": workspace3,
        "cur_state": cur_state,
        "attn_inter": attn_inter,
        "out_bf16": out_bf16,
    }


def run_benchmarks() -> None:
    device_id = int(os.environ.get("NPU_ID", "0"))
    device = f"npu:{device_id}"
    torch.npu.set_device(device)

    ai_core_num = ai_core_num_from_device()

    lib1 = ctypes.CDLL(LIB1)
    lib1.call_stage1.argtypes = [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * 13
    lib1.call_stage1.restype = None

    lib2 = ctypes.CDLL(LIB2)
    lib2.call_stage2.argtypes = [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * 8
    lib2.call_stage2.restype = None

    lib3 = ctypes.CDLL(LIB3)
    lib3.call_stage3.argtypes = [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * 8
    lib3.call_stage3.restype = None

    stream = torch.npu.current_stream()._as_parameter_

    rows: list[dict] = []

    for case in BENCHMARK_CASES:
        B = case["B"]
        seqlen = case["seqlen"]
        nk, nv, dk, dv, chunk = case["nk"], case["nv"], case["dk"], case["dv"], case["chunk"]
        name = case["name"]

        try:
            correctness = validate_case(case, device, ai_core_num)
        except RuntimeError as e:
            print(f"SKIP[{name}] correctness validation crashed: {e}")
            continue
        print(
            f"[{name}] correctness "
            f"out(max={correctness['out_max_abs']:.6f}, mean={correctness['out_mean_abs']:.6f}) "
            f"state(max={correctness['state_max_abs']:.6f}, mean={correctness['state_mean_abs']:.6f})"
        )

        rep = prepare_representative_sequence(case, device, ai_core_num)
        tiling_tensor = rep["tiling_tensor"]
        tiling_nbytes = rep["tiling_nbytes"]

        op1_bytes_single = (
            nbytes(rep["query"])
            + nbytes(rep["key"])
            + nbytes(rep["value"])
            + nbytes(rep["beta"])
            + nbytes(rep["gamma"])
            + nbytes(rep["stage_one_mask"])
            + nbytes(rep["qkt"])
            + nbytes(rep["g_cum_exp"])
            + nbytes(rep["k_cum_decay"])
            + nbytes(rep["v_inner"])
            + nbytes(rep["q_prime"])
            + nbytes(rep["kg"])
            + tiling_nbytes
        )
        f1 = B * estimate_stage1_flops(seqlen, nv, dk, dv, chunk)

        def run_s1_once() -> None:
            lib1.call_stage1(
                ai_core_num,
                stream,
                as_ptr(rep["query"]),
                as_ptr(rep["key"]),
                as_ptr(rep["value"]),
                as_ptr(rep["beta"]),
                as_ptr(rep["gamma"]),
                as_ptr(rep["stage_one_mask"]),
                as_ptr(rep["qkt"]),
                as_ptr(rep["g_cum_exp"]),
                as_ptr(rep["k_cum_decay"]),
                as_ptr(rep["v_inner"]),
                as_ptr(rep["q_prime"]),
                as_ptr(rep["kg"]),
                as_ptr(rep["workspace1"]),
                as_ptr(tiling_tensor),
            )

        def run_s1() -> None:
            for _ in range(B):
                run_s1_once()

        s1_ms = benchmark_with_events(run_s1)
        rows.append(
            {
                "case": name,
                "stage": "1",
                "B": B,
                "seqlen": seqlen,
                "T_total": B * seqlen,
                "nk": nk,
                "nv": nv,
                "dk": dk,
                "dv": dv,
                "chunk": chunk,
                "ms": s1_ms,
                "flops_est": f1,
                "tflops_est": ms_to_tflops_per_s(f1, s1_ms),
                "operand_bytes": B * op1_bytes_single,
                "operand_gibs": ms_to_operand_gibs(B * op1_bytes_single, s1_ms),
                "ai_core_num": ai_core_num,
                **correctness,
            }
        )

        # --- Stage2: snapshot after Stage1, restore each timed iteration ---
        torch.npu.synchronize()
        run_s1_once()
        torch.npu.synchronize()

        snap_qp = rep["q_prime"].clone()
        snap_vi = rep["v_inner"].clone()
        snap_g = rep["g_cum_exp"].clone()
        snap_kcd = rep["k_cum_decay"].clone()
        snap_kg = rep["kg"].clone()

        f2 = B * estimate_stage2_flops(seqlen, nv, dk, dv)
        op2_bytes_single = (
            nbytes(rep["q_prime"])
            + nbytes(rep["v_inner"])
            + nbytes(rep["g_cum_exp"])
            + nbytes(rep["k_cum_decay"])
            + nbytes(rep["cur_state"])
            + nbytes(rep["kg"])
            + nbytes(rep["attn_inter"])
            + tiling_nbytes
        )

        def run_s2_once() -> None:
            rep["q_prime"].copy_(snap_qp)
            rep["v_inner"].copy_(snap_vi)
            rep["g_cum_exp"].copy_(snap_g)
            rep["k_cum_decay"].copy_(snap_kcd)
            rep["kg"].copy_(snap_kg)
            rep["cur_state"].zero_()
            rep["attn_inter"].zero_()
            lib2.call_stage2(
                ai_core_num,
                stream,
                as_ptr(rep["q_prime"]),
                as_ptr(rep["v_inner"]),
                as_ptr(rep["g_cum_exp"]),
                as_ptr(rep["k_cum_decay"]),
                as_ptr(rep["cur_state"]),
                as_ptr(rep["kg"]),
                as_ptr(rep["attn_inter"]),
                as_ptr(rep["workspace2"]),
                as_ptr(tiling_tensor),
            )

        def run_s2() -> None:
            for _ in range(B):
                run_s2_once()

        s2_ms = benchmark_with_events(run_s2)
        rows.append(
            {
                "case": name,
                "stage": "2",
                "B": B,
                "seqlen": seqlen,
                "T_total": B * seqlen,
                "nk": nk,
                "nv": nv,
                "dk": dk,
                "dv": dv,
                "chunk": chunk,
                "ms": s2_ms,
                "flops_est": f2,
                "tflops_est": ms_to_tflops_per_s(f2, s2_ms),
                "operand_bytes": B * op2_bytes_single,
                "operand_gibs": ms_to_operand_gibs(B * op2_bytes_single, s2_ms),
                "ai_core_num": ai_core_num,
                **correctness,
            }
        )

        # --- Stage3: run S1+S2 once, snapshot, restore each timed iteration ---
        run_s2_once()
        torch.npu.synchronize()

        snap_qkt = rep["qkt"].clone()
        snap_ge = rep["g_cum_exp"].clone()
        snap_ai = rep["attn_inter"].clone()
        snap_vi2 = rep["v_inner"].clone()

        f3 = B * estimate_stage3_flops(seqlen, nv, dv, chunk)
        op3_bytes_single = (
            nbytes(rep["qkt"])
            + nbytes(rep["g_cum_exp"])
            + nbytes(rep["attn_inter"])
            + nbytes(rep["v_inner"])
            + nbytes(rep["stage_three_mask"])
            + nbytes(rep["out_bf16"])
            + tiling_nbytes
        )

        def run_s3_once() -> None:
            rep["qkt"].copy_(snap_qkt)
            rep["g_cum_exp"].copy_(snap_ge)
            rep["attn_inter"].copy_(snap_ai)
            rep["v_inner"].copy_(snap_vi2)
            rep["out_bf16"].zero_()
            lib3.call_stage3(
                ai_core_num,
                stream,
                as_ptr(rep["qkt"]),
                as_ptr(rep["g_cum_exp"]),
                as_ptr(rep["attn_inter"]),
                as_ptr(rep["v_inner"]),
                as_ptr(rep["stage_three_mask"]),
                as_ptr(rep["out_bf16"]),
                as_ptr(rep["workspace3"]),
                as_ptr(tiling_tensor),
            )

        def run_s3() -> None:
            for _ in range(B):
                run_s3_once()

        s3_ms = benchmark_with_events(run_s3)
        rows.append(
            {
                "case": name,
                "stage": "3",
                "B": B,
                "seqlen": seqlen,
                "T_total": B * seqlen,
                "nk": nk,
                "nv": nv,
                "dk": dk,
                "dv": dv,
                "chunk": chunk,
                "ms": s3_ms,
                "flops_est": f3,
                "tflops_est": ms_to_tflops_per_s(f3, s3_ms),
                "operand_bytes": B * op3_bytes_single,
                "operand_gibs": ms_to_operand_gibs(B * op3_bytes_single, s3_ms),
                "ai_core_num": ai_core_num,
                **correctness,
            }
        )

        print(
            f"[{name}] B={B} S={seqlen} nk/nv={nk}/{nv} dk/dv={dk}/{dv} chunk={chunk}  "
            f"stage1={rows[-3]['ms']:.3f} ms ({rows[-3]['tflops_est']:.4f} TFLOP/s est, "
            f"{rows[-3]['operand_gibs']:.4f} GiB/s op), "
            f"stage2={rows[-2]['ms']:.3f} ms ({rows[-2]['tflops_est']:.4f} TFLOP/s est, "
            f"{rows[-2]['operand_gibs']:.4f} GiB/s op), "
            f"stage3={rows[-1]['ms']:.3f} ms ({rows[-1]['tflops_est']:.4f} TFLOP/s est, "
            f"{rows[-1]['operand_gibs']:.4f} GiB/s op)"
        )

    csv_path = os.path.join(_HERE, "benchmark_stage_kernels.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    run_benchmarks()
