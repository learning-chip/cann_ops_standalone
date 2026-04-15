# MLA Prefill

## Build and run

```bash
bash ./compile.sh
python ./test_mla_prefill.py
```

Expected output:
```
mla_prefill call finished.
output mean abs: 0.110291
output finite: True
output changed: True
mean abs diff vs ref: 0.000083
max abs diff vs ref: 0.000854
tor example: 0.072169
```
<<<<<<< Updated upstream
=======

## TFLOP/s comparison (measured)

`python ./test_mla_prefill.py` runs benchmarks and writes `benchmark_mla_prefill.csv`. The three APIs are timed on the **same** random half-precision BNSD tensors (`fq`, `fk`, `fv` with head size 128): the MLA path pads Q/K to 192 with zeros in the rope slots and uses the same softmax scale `1/sqrt(128)` as `torch.nn.functional.scaled_dot_product_attention` and `torch_npu.npu_fused_infer_attention_score`. Throughput is reported as **TFLOP/s using a shared FLOP count** (`flops_theory_attn128`: standard two matmuls at head size 128), so the three columns are directly comparable for the same logical attention workload.

Representative numbers from a full run on this repo’s benchmark matrix (see the CSV for exact rows):

| Role | TFLOP/s range (attn-128 basis) | Notes |
|------|----------------------------------|--------|
| MLA prefill kernel | ~0.23–92 | Highest on several smaller or mid-sized shapes (e.g. B4×H16 up to S768); often leads **SDPA** and **`npu_fused_infer_attention_score`** there. |
| SDPA (NPU) | ~0.17–136 | Catches up or exceeds MLA on larger sequences (e.g. S1024, and many B8/B16 cases in this run). |
| `npu_fused_infer_attention_score` | ~0.17–138 | Close to SDPA on many rows; best TFLOP/s on some high-head cases (e.g. B8×H32 S512 in this CSV). |

**Tiny IFA-style point** (B1, H32, Q1, KV2048): MLA ~0.23 TFLOP/s vs SDPA and fused ~0.17 TFLOP/s on the same tensors (same run).

**Caveats:** Figures depend on device, CANN/torch_npu build, and clock; rerun locally for your hardware. The MLA kernel still performs extra arithmetic in the 192-wide Q/K path; `flops_theory_mla_hw` in the CSV is the heavier FLOP estimate for that implementation, while the three TFLOP/s columns use the shared attn-128 denominator for cross-API comparison.
>>>>>>> Stashed changes
