#!/usr/bin/env python3
"""
Benchmark paged KV (`block_table` + `actual_seq_lengths` + `block_size`) for
`torch_npu.npu_incre_flash_attention` on GQA decode shapes.

KV layout (page attention, BSH-compatible): `(num_blocks, block_size, num_kv_heads * head_dim)`.
`block_table`: `[batch, max_logical_blocks]` int32, physical block id for each logical block slot.

Correctness: compare against dense BSH `[batch, kv_seq, num_kv_heads * head_dim]` on the same tensors.

Throughput uses the same FLOP and byte models as `bench_ifa_gpa.py` (logical Q+K+V+O).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass

import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bench_ifa_gpa import benchmark_with_events, gqa_decode_matmul_flops, gqa_tensor_bytes_bsh


def pack_dense_kv_bsh_to_paged(
    k_dense: torch.Tensor,
    v_dense: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pack dense `[B, L, Hkv*D]` into paged `[B*nb, block_size, Hkv*D]` with contiguous batch blocks.

    Requires `L % block_size == 0` so each logical block is full (typical when aligning lengths).
    """
    if k_dense.dim() != 3 or v_dense.shape != k_dense.shape:
        raise ValueError("k/v must be [B, L, hkv_dim] and match")
    bsz, L, _ = k_dense.shape
    if L % block_size != 0:
        raise ValueError(f"L={L} must be divisible by block_size={block_size}")
    nb = L // block_size
    hdim = k_dense.shape[-1]
    k_page = k_dense.view(bsz, nb, block_size, hdim).reshape(bsz * nb, block_size, hdim).contiguous()
    v_page = v_dense.view(bsz, nb, block_size, hdim).reshape(bsz * nb, block_size, hdim).contiguous()
    device = k_dense.device
    block_table = torch.arange(nb, dtype=torch.int32, device=device).unsqueeze(0).expand(bsz, -1).clone()
    block_table = block_table + (torch.arange(bsz, dtype=torch.int32, device=device) * nb).unsqueeze(1)
    return k_page, v_page, block_table, nb


def run_incre_flash_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    scale: float,
):
    return torch_npu.npu_incre_flash_attention(
        q,
        k,
        v,
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        input_layout="BSH",
        scale_value=scale,
    )


def run_incre_flash_paged(
    q: torch.Tensor,
    k_page: torch.Tensor,
    v_page: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    scale: float,
    block_table: torch.Tensor,
    actual_seq_lengths: list[int],
    block_size: int,
):
    return torch_npu.npu_incre_flash_attention(
        q,
        k_page,
        v_page,
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        input_layout="BSH",
        scale_value=scale,
        block_table=block_table,
        actual_seq_lengths=actual_seq_lengths,
        block_size=block_size,
    )


def verify_paged_matches_dense(
    batch: int,
    kv_seq: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    dtype: torch.dtype,
    device: str,
) -> None:
    """Sanity: paged IFA matches dense IFA (fp16/bf16 fused softmax may differ slightly)."""
    scale = 1.0 / math.sqrt(float(head_dim))
    q_seq = 1
    q = torch.randn(batch, q_seq, num_heads * head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, kv_seq, num_kv_heads * head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, kv_seq, num_kv_heads * head_dim, dtype=dtype, device=device)

    o_dense = run_incre_flash_dense(q, k, v, num_heads, num_kv_heads, scale)
    k_p, v_p, bt, _ = pack_dense_kv_bsh_to_paged(k, v, block_size)
    o_page = run_incre_flash_paged(
        q,
        k_p,
        v_p,
        num_heads,
        num_kv_heads,
        scale,
        bt,
        [kv_seq] * batch,
        block_size,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(o_dense, o_page, rtol=5e-3, atol=5e-3)


@dataclass(frozen=True)
class PagedCase:
    name: str
    batch: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    kv_seq: int
    block_size: int


def default_paged_cases() -> list[PagedCase]:
    """Qwen3-8B GQA (32/8) + block_size 128 (vLLM-style); kv_seq divisible by 128."""
    return [
        PagedCase("paged_b1_h32_kv8_kv2048_bs128", 1, 32, 8, 128, 2048, 128),
        PagedCase("paged_b4_h32_kv8_kv2048_bs128", 4, 32, 8, 128, 2048, 128),
        PagedCase("paged_b8_h32_kv8_kv2048_bs128", 8, 32, 8, 128, 2048, 128),
        PagedCase("paged_b16_h32_kv8_kv2048_bs128", 16, 32, 8, 128, 2048, 128),
        PagedCase("paged_b32_h32_kv8_kv2048_bs128", 32, 32, 8, 128, 2048, 128),
        PagedCase("paged_b64_h32_kv8_kv2048_bs128", 64, 32, 8, 128, 2048, 128),
        PagedCase("paged_b1_h32_kv8_kv4096_bs128", 1, 32, 8, 128, 4096, 128),
        PagedCase("paged_b32_h32_kv8_kv4096_bs128", 32, 32, 8, 128, 4096, 128),
        # block_size 256 (still kv_seq % 256 == 0)
        PagedCase("paged_b32_h32_kv8_kv2048_bs256", 32, 32, 8, 128, 2048, 256),
    ]


def run_one_paged(case: PagedCase, dtype: torch.dtype, warmup_iters: int, benchmark_iters: int) -> None:
    b = case.batch
    nq = case.num_heads
    nkv = case.num_kv_heads
    d = case.head_dim
    s_kv = case.kv_seq
    bs = case.block_size
    q_seq = 1
    if s_kv % bs != 0:
        raise ValueError(f"kv_seq {s_kv} must be divisible by block_size {bs} for this benchmark")

    scale = 1.0 / math.sqrt(float(d))
    device = "npu"

    q = torch.randn(b, q_seq, nq * d, dtype=dtype, device=device)
    k_dense = torch.randn(b, s_kv, nkv * d, dtype=dtype, device=device)
    v_dense = torch.randn(b, s_kv, nkv * d, dtype=dtype, device=device)
    k_page, v_page, block_table, _nb = pack_dense_kv_bsh_to_paged(k_dense, v_dense, bs)

    def forward():
        return run_incre_flash_paged(
            q,
            k_page,
            v_page,
            nq,
            nkv,
            scale,
            block_table,
            [s_kv] * b,
            bs,
        )

    out = forward()
    torch.npu.synchronize()
    assert out.shape == (b, q_seq, nq * d)

    ms = benchmark_with_events(forward, warmup_iters=warmup_iters, benchmark_iters=benchmark_iters)
    t_s = ms * 1e-3
    flops = gqa_decode_matmul_flops(b, nq, q_seq, s_kv, d)
    elem_size = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    nbytes = gqa_tensor_bytes_bsh(b, q_seq, s_kv, nq, nkv, d, elem_size)
    tflops = flops / t_s / 1e12
    gibs = (nbytes / t_s) / (1024**3)
    ai = flops / nbytes
    ratio = nq / nkv
    print(
        f"{case.name}: {ms:.4f} ms | {tflops:.4f} TFLOP/s | {gibs:.4f} GiB/s (logical Q+K+V+O) | "
        f"AI={ai:.4f} F/B | Hq/Hkv={ratio:.1f} | block_size={bs}"
    )


def _resolve_npu_device_id(cli_device: int | None) -> int:
    if cli_device is not None:
        return cli_device
    env = os.environ.get("ASCEND_DEVICE_ID", "").strip()
    if env.isdigit():
        return int(env)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="IFA GQA paged-KV benchmark (block_table).")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--skip-verify", action="store_true", help="Skip dense vs paged correctness check.")
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        metavar="N",
        help="NPU device id (default: ASCEND_DEVICE_ID env or 0).",
    )
    args = parser.parse_args()

    if torch_npu is None or not torch.npu.is_available():
        print("NPU / torch_npu required.", file=sys.stderr)
        sys.exit(1)

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    npu_id = _resolve_npu_device_id(args.device)
    torch.npu.set_device(npu_id)

    print("npu_incre_flash_attention **paged KV** (block_table) — timer: benchmark_with_events")
    print(f"npu:{npu_id} dtype={dtype} warmup={args.warmup} benchmark_iters={args.iters}")

    if not args.skip_verify:
        print("Correctness: dense BSH vs paged KV (small + large batch)...")
        verify_paged_matches_dense(2, 2048, 32, 8, 128, 128, dtype, "npu")
        verify_paged_matches_dense(8, 2048, 32, 8, 128, 128, dtype, "npu")
        print("  assert_close passed (rtol=5e-3, atol=5e-3).")
        print("---")

    for c in default_paged_cases():
        try:
            run_one_paged(c, dtype, args.warmup, args.iters)
        except RuntimeError as e:
            print(f"{c.name}: FAILED — {e}")


if __name__ == "__main__":
    main()
