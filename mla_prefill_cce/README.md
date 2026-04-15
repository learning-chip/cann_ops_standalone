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

## TFLOP/s and accuracy (measured)

`python ./test_mla_prefill.py` writes `benchmark_mla_prefill.csv`. The **custom MLA** kernel and **`npu_fused_infer_attention_score`** use the same random half-precision inputs: BNSD `fq` / `fk` / `fv` (head size 128), with `query_rope` / `key_rope` (64-wide, zero-padded) matching the packed MLA Q/K layout. Throughput uses a shared **`flops_theory_attn128`** (two matmuls at head size 128) for both `mla_tflops_attn128` and `npu_fused_tflops_attn128`. **`flops_theory_mla_hw`** records the heavier MLA Q/K (192) + V (128) FLOP count for the custom kernel.

**FLOPs and comparability.** In a practical sense, the two kernels are **quite comparable**: they use the **same input sizes and shapes** (batch, heads, Q/KV lengths, 128-dim value head), the **same softmax scale** (`1/√128` on the nope path), and the **same overall attention math** (QK with nope+rope, softmax, PV). The reported TFLOP/s therefore uses **one shared** `flops_theory_attn128` so both paths are measured against the same nominal workload. **Minor gaps** are still possible: **layout** differs (packed 192-dim Q/K for MLA vs split BNSD `query`/`key` + `query_rope`/`key_rope` for fused), and the implementations may use **slightly different internal ops** (e.g. fused online softmax/LSE vs MLA’s workspace, extra rope fusion inside CANN). Treat `flops_theory_mla_hw` as the closer estimate for **custom MLA** arithmetic alone; treat `flops_theory_attn128` as a **common headline** for cross-kernel throughput, not a cycle-accurate match.

Numerical columns **`mean_abs_err_mla_vs_fused`** and **`max_abs_err_mla_vs_fused`** compare the MLA output tensor to the fused op’s attention output on those same inputs.

Benchmark cases favor **large B × H × sequence** to saturate the device and report higher TFLOP/s (see `cases` in `test_mla_prefill.py`).

### Performance table (representative run)

Rounded TFLOP/s from `benchmark_mla_prefill.csv`; exact floats are in the CSV. **Custom CCE MLA** is this repo’s hand-written kernel (`mla_prefill`); **CANN built-in MLA** is `torch_npu.npu_fused_infer_attention_score` with split nope/rope inputs (same tensors as the benchmark).

| Benchmark case | B | H | Q len | KV len | Custom CCE MLA (TFLOP/s) | CANN built-in MLA (TFLOP/s) |
|----------------|--:|--:|------:|-------:|-------------------------:|----------------------------:|
| IFA-style | 1 | 32 | 1 | 2048 | 0.24 | 0.17 |
| prefill_b4_h16_s512 | 4 | 16 | 512 | 512 | 47.4 | 34.9 |
| prefill_b4_h16_s768 | 4 | 16 | 768 | 768 | 78.7 | 68.9 |
| prefill_b4_h16_s1024 | 4 | 16 | 1024 | 1024 | 92.9 | 83.2 |
| prefill_b4_h16_s2048 | 4 | 16 | 2048 | 2048 | **117.2** | **118.5** |
| prefill_b8_h16_s768 | 8 | 16 | 768 | 768 | 82.9 | 75.7 |
| prefill_b8_h16_s1024 | 8 | 16 | 1024 | 1024 | 96.4 | 92.9 |
| prefill_b8_h32_s512 | 8 | 32 | 512 | 512 | 69.2 | 74.1 |
| prefill_b8_h32_s768 | 8 | 32 | 768 | 768 | 77.6 | 80.1 |
| prefill_b8_h32_s1024 | 8 | 32 | 1024 | 1024 | 92.2 | 99.2 |
| prefill_b16_h16_s512 | 16 | 16 | 512 | 512 | 68.8 | 75.6 |
| prefill_b16_h16_s1024 | 16 | 16 | 1024 | 1024 | 92.9 | 98.2 |
| prefill_b16_h32_s768 | 16 | 32 | 768 | 768 | 77.3 | 73.4 |
| prefill_b16_h32_s1024 | 16 | 32 | 1024 | 1024 | 93.3 | 97.1 |

On this run, peak **~117 TFLOP/s** (attn-128 basis) appears at **B4×H16, Q=KV=2048** for both custom CCE MLA and CANN built-in MLA; large **B16×H32×1024** remains in the low‑90s TFLOP/s range at ~2.8–2.9 ms latency.

Figures depend on device, CANN/torch_npu build, and clock; rerun locally for your hardware.
