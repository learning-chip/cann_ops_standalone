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

Numerical columns **`mean_abs_err_mla_vs_fused`** and **`max_abs_err_mla_vs_fused`** compare the MLA output tensor to the fused op’s attention output on those same inputs.

Figures depend on device, CANN/torch_npu build, and clock; rerun locally for your hardware.
