#!/usr/bin/env python3
"""
Benchmark paged-KV decode attention for Qwen3-style GQA shapes using:
  - torch_npu._npu_paged_attention (ATB v1 layout: ND cache)
  - torch_npu.atb._npu_paged_attention_v2 (mask_type=0, no mask)

KV is packed like `bench_ifa_gpa_paged.py`: logical dense [B, kv_seq, Hkv*D] with
full blocks, mapped through `block_table` [B, nb].

Throughput metrics match `bench_ifa_gpa.py` / `bench_ifa_gpa_paged.py`:
  - FLOPs: gqa_decode_matmul_flops (QK + PV, Hq counts for matmuls)
  - Bytes: gqa_tensor_bytes_bsh (logical Q + K + V + O; GQA uses Hkv for K/V width)

Timer: same event-based mean as `benchmark_with_events`.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None

_ATB_PA = os.path.dirname(os.path.abspath(__file__))
_IFA = os.path.join(_ATB_PA, "..", "ifa")
for _p in (_IFA,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bench_ifa_gpa import (  # noqa: E402
    benchmark_with_events,
    default_cases,
    gqa_decode_matmul_flops,
    gqa_tensor_bytes_bsh,
)
from bench_ifa_gpa_paged import (  # noqa: E402
    PagedCase,
    default_paged_cases,
    pack_dense_kv_bsh_to_paged,
    run_incre_flash_paged,
)


def ifa_readme_paged_cases(block_size: int = 128) -> list[PagedCase]:
    """Shapes from `bench_ifa_gpa.default_cases()` (IFA README table); paged KV with fixed block_size."""
    out: list[PagedCase] = []
    for c in default_cases():
        if c.kv_seq % block_size != 0:
            continue
        out.append(
            PagedCase(
                c.name,
                c.batch,
                c.num_heads,
                c.num_kv_heads,
                c.head_dim,
                c.kv_seq,
                block_size,
            )
        )
    return out


def resolve_npu_device_id(cli_device: int | None) -> int:
    """Prefer explicit --device, then ASCEND_DEVICE_ID, else 0."""
    if cli_device is not None:
        return cli_device
    env = os.environ.get("ASCEND_DEVICE_ID", "").strip()
    if env.isdigit():
        return int(env)
    return 0


def pack_kv_bsh_to_atb_nhd_paged(
    k_dense: torch.Tensor,
    v_dense: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense BSH [B, L, Hkv*D] -> ATB paged [num_blocks, block_size, Hkv, D]."""
    if k_dense.dim() != 3 or v_dense.shape != k_dense.shape:
        raise ValueError("k/v must be [B, L, hkv*D] and match")
    bsz, L, hdim = k_dense.shape
    if hdim != num_kv_heads * head_dim:
        raise ValueError(f"last dim {hdim} != {num_kv_heads}*{head_dim}")
    if L % block_size != 0:
        raise ValueError(f"L={L} must be divisible by block_size={block_size}")
    nb = L // block_size
    k4 = k_dense.view(bsz, L, num_kv_heads, head_dim)
    v4 = v_dense.view(bsz, L, num_kv_heads, head_dim)
    k_page = (
        k4.view(bsz, nb, block_size, num_kv_heads, head_dim)
        .reshape(bsz * nb, block_size, num_kv_heads, head_dim)
        .contiguous()
    )
    v_page = (
        v4.view(bsz, nb, block_size, num_kv_heads, head_dim)
        .reshape(bsz * nb, block_size, num_kv_heads, head_dim)
        .contiguous()
    )
    device = k_dense.device
    block_table = torch.arange(nb, dtype=torch.int32, device=device).unsqueeze(0).expand(bsz, -1).clone()
    block_table = block_table + (torch.arange(bsz, dtype=torch.int32, device=device) * nb).unsqueeze(1)
    return k_page, v_page, block_table


def q_bsh_to_atb(q_bsh: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """[B, 1, Hq*D] -> [B, Hq, D]."""
    b, q_seq, hdim = q_bsh.shape
    if q_seq != 1 or hdim != num_heads * head_dim:
        raise ValueError(f"expected [B,1,Hq*D], got {tuple(q_bsh.shape)}")
    return q_bsh.view(b, num_heads, head_dim).contiguous()


def metrics_line(
    label: str,
    case_name: str,
    ms: float,
    flops: float,
    nbytes: float,
) -> str:
    t_s = ms * 1e-3
    tflops = flops / t_s / 1e12
    gibs = (nbytes / t_s) / (1024**3)
    ai = flops / nbytes if nbytes else 0.0
    return (
        f"{case_name} | {label:22s} | {ms:7.4f} ms | {tflops:7.4f} TFLOP/s | "
        f"{gibs:7.4f} GiB/s | AI={ai:.4f} F/B"
    )


def bench_all_apis_for_case(
    case: PagedCase,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    run_ifa: bool,
) -> None:
    b = case.batch
    nq = case.num_heads
    nkv = case.num_kv_heads
    d = case.head_dim
    s_kv = case.kv_seq
    bs = case.block_size
    q_seq = 1

    scale = 1.0 / math.sqrt(float(d))
    elem_size = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    flops = gqa_decode_matmul_flops(b, nq, q_seq, s_kv, d)
    nbytes = gqa_tensor_bytes_bsh(b, q_seq, s_kv, nq, nkv, d, elem_size)

    q_bsh = torch.randn(b, q_seq, nq * d, dtype=dtype, device="npu")
    k_dense = torch.randn(b, s_kv, nkv * d, dtype=dtype, device="npu")
    v_dense = torch.randn(b, s_kv, nkv * d, dtype=dtype, device="npu")

    q_atb = q_bsh_to_atb(q_bsh, nq, d)
    k_page, v_page, block_table = pack_kv_bsh_to_atb_nhd_paged(k_dense, v_dense, nkv, d, bs)
    context_lens = torch.tensor([s_kv] * b, dtype=torch.int32, device="cpu")
    out_v1 = torch.empty(b, nq, d, dtype=dtype, device="npu")
    out_v2 = torch.empty_like(out_v1)

    if run_ifa:
        k_ifa, v_ifa, bt_ifa, _ = pack_dense_kv_bsh_to_paged(k_dense, v_dense, bs)

        def forward_ifa():
            return run_incre_flash_paged(
                q_bsh,
                k_ifa,
                v_ifa,
                nq,
                nkv,
                scale,
                bt_ifa,
                [s_kv] * b,
                bs,
            )

        ms_ifa = benchmark_with_events(forward_ifa, warmup_iters=warmup, benchmark_iters=iters)
        print(metrics_line("npu_incre_flash (paged)", case.name, ms_ifa, flops, nbytes))

    def forward_v1():
        torch_npu._npu_paged_attention(
            q_atb,
            k_page,
            v_page,
            nkv,
            nq,
            scale,
            block_table,
            context_lens,
            out_v1,
        )
        return out_v1

    def forward_v2():
        torch_npu.atb._npu_paged_attention_v2(
            q_atb,
            k_page,
            block_table,
            context_lens,
            value_cache=v_page,
            mask=None,
            num_kv_heads=nkv,
            num_heads=nq,
            scale_value=scale,
            mask_type=0,
            out=out_v2,
        )
        return out_v2

    ms_v1 = benchmark_with_events(forward_v1, warmup_iters=warmup, benchmark_iters=iters)
    print(metrics_line("_npu_paged_attention", case.name, ms_v1, flops, nbytes))

    ms_v2 = benchmark_with_events(forward_v2, warmup_iters=warmup, benchmark_iters=iters)
    print(metrics_line("_npu_paged_attention_v2", case.name, ms_v2, flops, nbytes))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ATB paged attention vs IFA paged GQA benchmark (Qwen3 shapes, block_table)."
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--no-ifa",
        action="store_true",
        help="Skip npu_incre_flash_attention (paged) baseline.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        metavar="N",
        help="NPU device id (default: ASCEND_DEVICE_ID env or 0). Pick a free id via npu-smi info.",
    )
    parser.add_argument(
        "--suite",
        choices=("paged", "ifa-gpa"),
        default="paged",
        help="paged: bench_ifa_gpa_paged default cases; ifa-gpa: same shapes as bench_ifa_gpa / IFA README.",
    )
    args = parser.parse_args()

    if torch_npu is None or not torch.npu.is_available():
        print("NPU / torch_npu required.", file=sys.stderr)
        sys.exit(1)

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    npu_id = resolve_npu_device_id(args.device)
    torch.npu.set_device(npu_id)

    print(
        "Paged GQA decode — same FLOP/byte model as bench_ifa_gpa_paged.py "
        "(logical Q+K+V+O); timer: benchmark_with_events"
    )
    print(f"npu:{npu_id} dtype={dtype} warmup={args.warmup} benchmark_iters={args.iters}")
    print(
        "case | API | ms | TFLOP/s | GiB/s | AI\n"
        "(v2 uses mask_type=0, mask=None; v1/v2 cache layout "
        "[num_blocks, block_size, kv_heads, head_dim])"
    )
    print("---")

    cases: list[PagedCase] = (
        default_paged_cases() if args.suite == "paged" else ifa_readme_paged_cases()
    )
    for c in cases:
        try:
            bench_all_apis_for_case(c, dtype, args.warmup, args.iters, run_ifa=not args.no_ifa)
        except RuntimeError as e:
            print(f"{c.name}: FAILED — {e}")


if __name__ == "__main__":
    main()
