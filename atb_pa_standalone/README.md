# atb_pa_standalone — Standalone Paged Attention Kernel

This directory contains a self-contained extraction of the **paged attention
decode kernel** from
[ascend-transformer-boost](../../../ascend-transformer-boost), compiled and
tested entirely outside the ATB framework.

---

## Contents

| File | Role |
|------|------|
| `op_kernel/paged_attention_mask_mix.cce` | Main CCE kernel entry point |
| `op_kernel/paged_attention_decoder_nd_common.cce` | AiC/AiV compute logic (included by the above) |
| `paged_attention_wrapper.cpp` | Thin C++ host wrapper (`call_kernel`, `get_ffts_info`) |
| `compile.sh` | Single-step bisheng build script → `pa_lib.so` |
| `pa_tiling.py` | Python port of the C++ tiling logic |
| `test_atb_pa_standalone.py` | Numerical correctness tests (vs ATB v2 reference) |
| `bench_pa_standalone.py` | Throughput benchmark (standalone vs ATB v2) |

---

## Build

```bash
cd /workdir/cann_ops_standalone/atb_pa_standalone
bash compile.sh          # produces pa_lib.so
```

Requirements:
- CANN 8.5.1 with `bisheng` in `PATH`
- Headers from `ascend-transformer-boost` (referenced via `-I` in `compile.sh`)

---

## Correctness Tests

Each test case runs in a subprocess to guarantee a clean FFTS hardware state.
The FFTS (Fast Functional Task Synchronization) counters are global NPU hardware
state that accumulates across kernel invocations. Mixing different kernel
configurations (different `eff_bd` or tiling keys) in the same process can put
the counters at phases that cause premature `WaitFlagDev` returns and thus data
races. Subprocess isolation is the simplest reliable fix.

Within each subprocess, correctness is ensured by a **4× shape-specific warmup**
(the FFTS counter cycle length), followed immediately by the custom kernel
(no intervening NPU operations), and then the ATB v2 reference for comparison.

```bash
python test_atb_pa_standalone.py
```

Expected output:

```
Device: npu:0  cube_cores=24

Running fp16 tests (each case in an isolated subprocess):
  PASS  b1_h32_kv8_s128_bs128  mean_err=0.00000  max_err=0.00000
  PASS  b4_h32_kv8_s512_bs128  mean_err=0.00000  max_err=0.00012
  PASS  b2_h8_kv8_s256_bs128   mean_err=0.00000  max_err=0.00000
  PASS  b8_h32_kv8_s1024_bs128 mean_err=0.00000  max_err=0.00012
  PASS  b1_h32_kv8_s2048_bs128 mean_err=0.00000  max_err=0.00006
  PASS  b4_h64_kv8_s1024_bs128 mean_err=0.00000  max_err=0.00012

All fp16 cases PASSED.
```

Tolerances: `rtol=5e-3, atol=2e-2`.

---

## Performance Benchmark

```bash
python bench_pa_standalone.py [--warmup 5] [--iters 20] [--bf16]
```

Results on Ascend 910B (24 cube cores, fp16, `npu:0`, warmup=5, iters=20).
Same shapes as [`atb_pa/README.md`](../atb_pa/README.md) (ifa-gpa suite, `block_size=128`).
ATB v1 numbers are from `atb_pa/README.md` (measured on `npu:3`).

| Case | Standalone (ms) | Standalone (GiB/s) | ATB v2 (ms) | ATB v2 (GiB/s) | ATB v1 (GiB/s) |
|------|-----------------|--------------------|-------------|----------------|----------------|
| Qwen3-0.6B b1 h16/kv8 kv2048  | 0.081 |  96.6 | 0.128 |  61.3 |  49.1 |
| Qwen3-1.7B b1 h16/kv8 kv4096  | 0.075 | 207.3 | 0.114 | 137.0 | 101.3 |
| Qwen3-4B   b1 h32/kv8 kv2048  | 0.075 | 103.9 | 0.106 |  74.2 |  59.8 |
| Qwen3-8B   b1 h32/kv8 kv4096  | 0.074 | 211.8 | 0.097 | 162.1 | 110.7 |
| Qwen3-8B   b1 h32/kv8 kv8192  | 0.081 | 385.2 | 0.095 | 330.0 | 213.9 |
| Qwen3-14B  b1 h40/kv8 kv2048  | 0.074 | 106.5 | 0.092 |  85.5 |  59.9 |
| Qwen3-32B  b1 h64/kv8 kv2048  | 0.072 | 109.3 | 0.092 |  85.3 |  57.6 |
| MHA        b1 h32/kv32 kv2048 | 0.071 | 437.7 | 0.094 | 334.2 | 222.8 |
| Qwen3-8B   b4 h32/kv8 kv2048  | 0.073 | 430.8 | 0.103 | 304.5 | 215.2 |
| Qwen3-8B   b8 h32/kv8 kv2048  | 0.092 | 678.8 | 0.119 | 528.6 | 405.6 |
| Qwen3-8B  b16 h32/kv8 kv2048  | 0.123 |  1016 | 0.150 | 834.9 | 791.9 |
| Qwen3-8B  b32 h32/kv8 kv2048  | 0.266 |   941 | 0.246 |  1020 |  1051 |
| Qwen3-8B  b64 h32/kv8 kv2048  | 0.482 |  1039 | 0.479 |  1046 |  1085 |

**Key observations:**

- **b1–b16**: standalone is **1.2–1.6× faster** than ATB v2 and **1.5–2.0×** vs
  ATB v1, by eliminating framework dispatch overhead (~0.04–0.08 ms per call).
- **b32–b64**: all three converge; the kernel is fully compute/bandwidth-bound
  and framework overhead is negligible.
- **Bandwidth ceiling**: 910B HBM peak is ~820 GB/s. b16 standalone reaches
  ~1016 GiB/s (≈1090 GB/s), near the hardware limit; b32/b64 all saturate at
  ~1040–1090 GiB/s.
- **ATB v1 vs v2**: ATB v1 is faster than ATB v2 at small batch; the standalone
  path exceeds both by skipping the op-graph and tiling layers.

---

## Design Notes

### Tiling (`pa_tiling.py`)
Ports `paged_attention_tiling.cpp` / `paged_attention_tiling_dependency.cpp`.
Key functions:
- `make_pa_nd_decode_tiling` — builds the `tiling_para_gm` buffer (uint32 array
  with header + per-batch entries) and returns `eff_bd` (effective block dim).
- `workspace_sizes` — returns byte sizes for the 8 scratch buffers.

### FFTS State
The kernel uses hardware FFTS counters (`SetFftsBaseAddr`, `FftsCrossCoreSync`,
`WaitFlagDev`) for AiC↔AiV synchronisation within each kernel launch. These
counters are shared hardware state at a fixed address returned by
`rtGetC2cCtrlAddr`. They accumulate across invocations and are not resettable
from the host. Different tiling configurations (different `eff_bd` or tiling
keys) use different counter-offset patterns, and switching between them in a
single process can cause counter parity errors.

Mitigations used in tests:
1. **Subprocess isolation** — each test case runs in a fresh OS process.
2. **4× shape-specific warmup** — runs the target shape 4 times (= FFTS cycle
   length) before the measurement, guaranteeing counter phase 0.
3. **Custom before reference** — the custom kernel call precedes the ATB
   reference call so no intervening NPU ops disrupt the phase.

For benchmarking, FFTS phase only affects output correctness, not latency — the
kernel always completes in the same wall time regardless of phase.

### `g_tilingKey`
In production, the ATB MKI framework sets `g_tilingKey` before dispatching the
kernel, enabling `TILING_KEY_IS()` conditional compilation. In standalone mode
the kernel reads the tiling key from the tiling buffer directly (patched in
`paged_attention_mask_mix.cce`).
