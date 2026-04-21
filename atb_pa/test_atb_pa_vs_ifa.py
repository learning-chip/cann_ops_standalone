#!/usr/bin/env python3
"""
Numerical agreement: paged `npu_incre_flash_attention` (reference) vs
`torch_npu._npu_paged_attention` and `torch_npu.atb._npu_paged_attention_v2`.

Uses the same dense Q/K/V and paged `block_table` layout as `bench_atb_pa_gqa_paged.py`.
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
for _p in (_ATB_PA, _IFA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bench_ifa_gpa_paged import (  # noqa: E402
    PagedCase,
    default_paged_cases,
    pack_dense_kv_bsh_to_paged,
    run_incre_flash_paged,
)

from bench_atb_pa_gqa_paged import (  # noqa: E402
    ifa_readme_paged_cases,
    pack_kv_bsh_to_atb_nhd_paged,
    q_bsh_to_atb,
    resolve_npu_device_id,
)


def quick_cases() -> list[PagedCase]:
    """Small shapes for fast smoke tests."""
    return [
        PagedCase("quick_b1_h32_kv8_kv512_bs128", 1, 32, 8, 128, 512, 128),
        PagedCase("quick_b2_h16_kv8_kv256_bs128", 2, 16, 8, 128, 256, 128),
        PagedCase("quick_b4_h32_kv8_kv1024_bs128", 4, 32, 8, 128, 1024, 128),
    ]


def run_reference_and_atb(
    case: PagedCase,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    b = case.batch
    nq = case.num_heads
    nkv = case.num_kv_heads
    d = case.head_dim
    s_kv = case.kv_seq
    bs = case.block_size
    q_seq = 1
    scale = 1.0 / math.sqrt(float(d))

    q_bsh = torch.randn(b, q_seq, nq * d, dtype=dtype, device="npu")
    k_dense = torch.randn(b, s_kv, nkv * d, dtype=dtype, device="npu")
    v_dense = torch.randn(b, s_kv, nkv * d, dtype=dtype, device="npu")

    k_ifa, v_ifa, bt_ifa, _ = pack_dense_kv_bsh_to_paged(k_dense, v_dense, bs)

    o_ifa = run_incre_flash_paged(
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
    ref = o_ifa.view(b, nq, d).contiguous()

    q_atb = q_bsh_to_atb(q_bsh, nq, d)
    k_page, v_page, block_table = pack_kv_bsh_to_atb_nhd_paged(k_dense, v_dense, nkv, d, bs)
    context_lens = torch.tensor([s_kv] * b, dtype=torch.int32, device="cpu")

    out_v1 = torch.empty(b, nq, d, dtype=dtype, device="npu")
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

    out_v2 = torch.empty(b, nq, d, dtype=dtype, device="npu")
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

    torch.npu.synchronize()
    return ref, out_v1, out_v2


def verify_case(
    case: PagedCase,
    dtype: torch.dtype,
    rtol: float,
    atol: float,
    seed: int,
) -> None:
    ref, v1, v2 = run_reference_and_atb(case, dtype, seed)
    torch.testing.assert_close(v1, ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(v2, ref, rtol=rtol, atol=atol)


def main() -> None:
    parser = argparse.ArgumentParser(description="ATB paged attention vs IFA paged (numerical check).")
    parser.add_argument("--device", type=int, default=None, help="NPU id (default: ASCEND_DEVICE_ID or 0).")
    parser.add_argument(
        "--suite",
        choices=("quick", "ifa-gpa", "paged"),
        default="quick",
        help="quick: small shapes; ifa-gpa: bench_ifa_gpa README shapes; paged: bench_ifa_gpa_paged cases.",
    )
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if torch_npu is None or not torch.npu.is_available():
        print("NPU / torch_npu required.", file=sys.stderr)
        sys.exit(1)

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    npu_id = resolve_npu_device_id(args.device)
    torch.npu.set_device(npu_id)

    if args.suite == "quick":
        cases = quick_cases()
    elif args.suite == "ifa-gpa":
        cases = ifa_readme_paged_cases()
    else:
        cases = default_paged_cases()

    print(
        f"npu:{npu_id} suite={args.suite} dtype={dtype} cases={len(cases)} "
        f"rtol={args.rtol} atol={args.atol} seed={args.seed}"
    )

    failed = 0
    for i, c in enumerate(cases):
        try:
            verify_case(c, dtype, args.rtol, args.atol, args.seed + i)
            print(f"  OK  {c.name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {c.name}\n{e}", file=sys.stderr)

    if failed:
        print(f"{failed}/{len(cases)} case(s) failed.", file=sys.stderr)
        sys.exit(1)
    print("All cases passed.")


if __name__ == "__main__":
    main()
