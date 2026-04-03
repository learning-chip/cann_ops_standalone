"""
Benchmark the end-to-end staged custom kernel pipeline vs the torch reference.

The custom path reuses the verified `stage1_lib.so` / `stage2_lib.so` / `stage3_lib.so`
wrappers in a host-orchestrated Stage1 -> Stage2 -> Stage3 sequence. Timing uses
`torch.npu.Event` pairs. CSV output: `benchmark_chunk_gdn.csv`.
"""

from __future__ import annotations

import csv
import math
import os
import sys

import torch
import torch.nn.functional as F

import torch_npu

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from test_chunk_gdn import (
    StagedChunkGDNRunner,
    build_stage_tiling,
    cgdr_benchmark_bf16,
    cgdr_golden_native,
    padded_seq_len,
)


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


def estimate_chunk_gdn_flops(T: int, nk: int, nv: int, dk: int, dv: int, chunk: int) -> float:
    """Heuristic end-to-end FLOP estimate from the staged kernel matmul paths."""
    del nk
    c = chunk
    nc = (T + c - 1) // c
    stage1 = nv * nc * (6.0 * c * c * dk + 2.0 * c * c * dv + 2.0 * c * c * c)
    stage2 = nv * nc * (6.0 * c * dk * dv)
    stage3 = nv * nc * (2.0 * c * c * dv)
    return stage1 + stage2 + stage3


def estimate_effective_io_bytes(
    seq_lens: list[int],
    nk: int,
    nv: int,
    dk: int,
    dv: int,
    chunk: int,
    ai_core_num: int,
    scale: float,
) -> int:
    """Stage-operand footprint excluding scratch workspaces."""
    b2 = 2  # bf16
    f4 = 4  # float32
    stage_mask_bytes = chunk * chunk * ai_core_num * 2 * f4
    state_bytes = nv * dv * dk * f4
    total = 0

    for seq_len in seq_lens:
        sp = padded_seq_len(seq_len, chunk)
        tiling_nbytes = len(
            bytes(
                build_stage_tiling(
                    ai_core_num=ai_core_num,
                    seq_len=seq_len,
                    nk=nk,
                    nv=nv,
                    dk=dk,
                    dv=dv,
                    has_gamma=1,
                    chunk_size=chunk,
                    scale=scale,
                )
            )
        )

        total += (
            seq_len * nk * dk * b2
            + seq_len * nk * dk * b2
            + seq_len * nv * dv * b2
            + seq_len * nv * b2
            + seq_len * nv * f4
            + stage_mask_bytes
            + nv * sp * chunk * f4
            + nv * sp * f4
            + nv * sp * dk * f4
            + nv * sp * dv * f4
            + nv * sp * dk * f4
            + nv * sp * dk * f4
            + tiling_nbytes
        )
        total += (
            nv * sp * dk * f4
            + nv * sp * dv * f4
            + nv * sp * f4
            + nv * sp * dk * f4
            + state_bytes
            + nv * sp * dk * f4
            + nv * sp * dv * f4
            + tiling_nbytes
        )
        total += (
            nv * sp * chunk * f4
            + nv * sp * f4
            + nv * sp * dv * f4
            + nv * sp * dv * f4
            + stage_mask_bytes
            + seq_len * nv * dv * b2
            + tiling_nbytes
        )

    return total


def ms_to_tflops_per_s(flops: float, ms: float) -> float:
    if ms <= 0:
        return float("nan")
    return flops / (ms * 1e-3) / 1e12


def ms_to_effective_gibs(effective_bytes: int, ms: float) -> float:
    """GiB/s from effective (user-visible) byte count and kernel time."""
    if ms <= 0:
        return float("nan")
    return effective_bytes / (ms * 1e-3) / (1024.0**3)


def run_benchmarks(*, run_custom_kernel: bool = True) -> None:
    device_id = int(os.environ.get("NPU_ID", "0"))
    device = f"npu:{device_id}"
    torch.npu.set_device(device)

    try:
        from chunk_gdn_common import ai_core_num_from_device

        ai_core_num = ai_core_num_from_device()
    except Exception:
        ai_core_num = 8

    cases = [
        {"name": "gdn_b1_s4096_h4", "B": 1, "seqlen": 4096, "nk": 4, "nv": 4, "dk": 64, "dv": 64, "chunk": 64},
        {"name": "gdn_b1_s16384_h4", "B": 1, "seqlen": 16384, "nk": 4, "nv": 4, "dk": 64, "dv": 64, "chunk": 64},
        {"name": "gdn_b1_s65536_h4", "B": 1, "seqlen": 65536, "nk": 4, "nv": 4, "dk": 64, "dv": 64, "chunk": 64},
    ]

    error_warn_threshold = {"out_max": 0.35, "out_mean": 0.05, "state_max": 0.30, "state_mean": 0.05}
    results = []
    skipped_cases: list[str] = []

    for i, case in enumerate(cases):
        B = case["B"]
        seqlen = case["seqlen"]
        nk, nv, dk, dv, chunk_size = case["nk"], case["nv"], case["dk"], case["dv"], case["chunk"]
        T = B * seqlen
        scale = 1.0 / math.sqrt(float(dk))

        # Match `test_chunk_gdn.run_one_case`: allocate on-device (same RNG + layout as unit test).
        q = torch.rand((T, nk, dk), dtype=torch.bfloat16, device=device).contiguous()
        k = torch.rand((T, nk, dk), dtype=torch.bfloat16, device=device).contiguous()
        v = torch.rand((T, nv, dv), dtype=torch.bfloat16, device=device).contiguous()
        g = (torch.rand((T, nv), dtype=torch.float32, device=device) * -1.0).contiguous()
        beta = torch.rand((T, nv), dtype=torch.bfloat16, device=device).contiguous()
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        initial_state = torch.rand((B, nv, dv, dk), dtype=torch.bfloat16, device=device).contiguous()
        actual_seq_lengths = torch.full((B,), int(seqlen), dtype=torch.int32, device=device)

        flops_est = estimate_chunk_gdn_flops(T, nk, nv, dk, dv, chunk_size)
        seq_lens = [int(x) for x in actual_seq_lengths.cpu().tolist()]
        effective_io_bytes = estimate_effective_io_bytes(
            seq_lens, nk, nv, dk, dv, chunk_size, ai_core_num, scale
        )

        runner = None
        if run_custom_kernel:
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

        def run_custom() -> None:
            assert runner is not None
            runner.run()

        def run_ref() -> None:
            cgdr_benchmark_bf16(q, k, v, g, beta, scale, initial_state, actual_seq_lengths)

        mean_abs_err = float("nan")
        max_abs_err = float("nan")
        max_abs_err_state = float("nan")
        custom_ms = float("nan")
        custom_tflops = float("nan")
        custom_eff_gibs = float("nan")

        # Time torch reference before any custom-kernel launch so we still get baseline ms if the .so faults.
        try:
            ref_ms = benchmark_with_events(run_ref)
        except RuntimeError as e:
            print(f"WARNING[{case['name']}]: skipped (timing ref): {e}")
            skipped_cases.append(case["name"])
            continue

        ref_tflops = ms_to_tflops_per_s(flops_est, ref_ms)
        ref_eff_gibs = ms_to_effective_gibs(effective_io_bytes, ref_ms)

        if run_custom_kernel:
            try:
                run_custom()
                torch.npu.synchronize()
            except RuntimeError as e:
                print(
                    f"ERROR[{case['name']}]: custom kernel failed on smoke run: {e}\n"
                    "Stopping further cases (NPU may be unusable; use a fresh process or "
                    "`python benchmark_chunk_gdn.py --torch-ref-only` for reference-only numbers)."
                )
                skipped_cases.append(case["name"])
                skipped_cases.extend(c["name"] for c in cases[i + 1 :])
                results.append(
                    {
                        "case": case["name"],
                        "B": B,
                        "seqlen": seqlen,
                        "T": T,
                        "nk": nk,
                        "nv": nv,
                        "dk": dk,
                        "dv": dv,
                        "chunk": chunk_size,
                        "block_dim": ai_core_num,
                        "ai_core_num": ai_core_num,
                        "flops_estimate": flops_est,
                        "effective_io_bytes": effective_io_bytes,
                        "workspace_bytes_excluded_from_bw": "",
                        "custom_ms": float("nan"),
                        "torch_ref_ms": ref_ms,
                        "custom_tflops_est": float("nan"),
                        "torch_ref_tflops_est": ref_tflops,
                        "custom_effective_gibs": float("nan"),
                        "torch_ref_effective_gibs": ref_eff_gibs,
                        "mean_abs_err": float("nan"),
                        "max_abs_err_out": float("nan"),
                        "max_abs_err_state": float("nan"),
                    }
                )
                print(
                    f"[{case['name']}] torch_ref={ref_ms:.3f} ms ({ref_tflops:.4f} TFLOP/s est, {ref_eff_gibs:.4f} GiB/s eff); "
                    "custom kernel not measured (smoke failed)"
                )
                break

            o_golden, state_golden = cgdr_golden_native(
                q, k, v, g, beta, scale, initial_state, actual_seq_lengths
            )
            o_bench, state_bench = cgdr_benchmark_bf16(
                q, k, v, g, beta, scale, initial_state, actual_seq_lengths
            )
            out, final_state = runner.out, runner.final_state

            mean_abs_err = torch.mean(torch.abs(out.to(torch.float32) - o_bench)).item()
            max_abs_err = torch.max(torch.abs(out.to(torch.float32) - o_bench)).item()
            mean_abs_err_state = torch.mean(torch.abs(final_state.to(torch.float32) - state_bench)).item()
            max_abs_err_state = torch.max(torch.abs(final_state.to(torch.float32) - state_bench)).item()

            if (
                max_abs_err > error_warn_threshold["out_max"]
                or mean_abs_err > error_warn_threshold["out_mean"]
                or max_abs_err_state > error_warn_threshold["state_max"]
                or mean_abs_err_state > error_warn_threshold["state_mean"]
            ):
                print(
                    f"WARNING[{case['name']}]: skipped (golden mismatch): "
                    f"out max_abs={max_abs_err:.6f}, out mean_abs={mean_abs_err:.6f}, "
                    f"state max_abs={max_abs_err_state:.6f}, state mean_abs={mean_abs_err_state:.6f}, "
                    f"threshold={error_warn_threshold}"
                )
                skipped_cases.append(case["name"])
                continue

            try:
                custom_ms = benchmark_with_events(run_custom)
            except RuntimeError as e:
                print(
                    f"ERROR[{case['name']}]: timing custom kernel failed: {e}\n"
                    "Stopping further cases."
                )
                skipped_cases.append(case["name"])
                skipped_cases.extend(c["name"] for c in cases[i + 1 :])
                results.append(
                    {
                        "case": case["name"],
                        "B": B,
                        "seqlen": seqlen,
                        "T": T,
                        "nk": nk,
                        "nv": nv,
                        "dk": dk,
                        "dv": dv,
                        "chunk": chunk_size,
                        "block_dim": ai_core_num,
                        "ai_core_num": ai_core_num,
                        "flops_estimate": flops_est,
                        "effective_io_bytes": effective_io_bytes,
                        "workspace_bytes_excluded_from_bw": "",
                        "custom_ms": float("nan"),
                        "torch_ref_ms": ref_ms,
                        "custom_tflops_est": float("nan"),
                        "torch_ref_tflops_est": ref_tflops,
                        "custom_effective_gibs": float("nan"),
                        "torch_ref_effective_gibs": ref_eff_gibs,
                        "mean_abs_err": mean_abs_err,
                        "max_abs_err_out": max_abs_err,
                        "max_abs_err_state": max_abs_err_state,
                    }
                )
                print(
                    f"[{case['name']}] torch_ref={ref_ms:.3f} ms ({ref_tflops:.4f} TFLOP/s est, {ref_eff_gibs:.4f} GiB/s eff); "
                    "custom kernel not measured (timing failed)"
                )
                break

            custom_tflops = ms_to_tflops_per_s(flops_est, custom_ms)
            custom_eff_gibs = ms_to_effective_gibs(effective_io_bytes, custom_ms)

        if run_custom_kernel:
            print(
                f"[{case['name']}] kernel={custom_ms:.3f} ms ({custom_tflops:.4f} TFLOP/s est, {custom_eff_gibs:.4f} GiB/s eff), "
                f"torch_ref={ref_ms:.3f} ms ({ref_tflops:.4f} TFLOP/s est, {ref_eff_gibs:.4f} GiB/s eff), "
                f"mean_abs_err={mean_abs_err:.6f}, max_abs_err={max_abs_err:.6f}"
            )
        else:
            print(
                f"[{case['name']}] torch_ref={ref_ms:.3f} ms ({ref_tflops:.4f} TFLOP/s est, {ref_eff_gibs:.4f} GiB/s eff) "
                f"(custom kernel not timed)"
            )

        results.append(
            {
                "case": case["name"],
                "B": B,
                "seqlen": seqlen,
                "T": T,
                "nk": nk,
                "nv": nv,
                "dk": dk,
                "dv": dv,
                "chunk": chunk_size,
                "block_dim": ai_core_num if run_custom_kernel else "",
                "ai_core_num": ai_core_num,
                "flops_estimate": flops_est,
                "effective_io_bytes": effective_io_bytes,
                "workspace_bytes_excluded_from_bw": "",
                "custom_ms": custom_ms,
                "torch_ref_ms": ref_ms,
                "custom_tflops_est": custom_tflops,
                "torch_ref_tflops_est": ref_tflops,
                "custom_effective_gibs": custom_eff_gibs,
                "torch_ref_effective_gibs": ref_eff_gibs,
                "speedup_vs_torch": (ref_ms / custom_ms) if run_custom_kernel and custom_ms > 0 else float("nan"),
                "mean_abs_err": mean_abs_err,
                "max_abs_err_out": max_abs_err,
                "max_abs_err_state": max_abs_err_state,
            }
        )

    def _csv_val(x):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return ""
        return x

    if results:
        csv_path = os.path.join(_HERE, "benchmark_chunk_gdn.csv")
        rows = [{k: _csv_val(v) for k, v in row.items()} for row in results]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote benchmark csv: {csv_path}")
    else:
        print("WARNING: no successful benchmark cases; csv not written.")
    if skipped_cases:
        print(f"NOTE: skipped cases: {skipped_cases}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Benchmark chunk_gdn kernel vs torch reference.")
    ap.add_argument(
        "--torch-ref-only",
        action="store_true",
        help="Time only the torch reference (no custom kernel .so). Useful if the device is unstable after kernel runs.",
    )
    args = ap.parse_args()
    run_benchmarks(run_custom_kernel=not args.torch_ref_only)
