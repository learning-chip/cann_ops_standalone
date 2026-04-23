# PTO port: `pa_gqaonly_pto` (paged attention GQA decode)

## Goal

Match `atb_pa_gqaonly_cce` behavior and performance while moving implementation toward **PTO-ISA** APIs (`pto::TLOAD`, `pto::TMATMUL`, `pto::TADD`, `pto::TSTORE`, …), aligned with `pto-kernels/csrc` and `pto-isa-master/kernels/manual/a2a3/flash_atten`.

## Skills / references

- Requested skills path: `/workdir/pto-kernels/.skills/npu_kernel_general/skills.md` — **not present** in this workspace; mapping was taken from `pto-isa-master/include/pto/common/pto_instr.hpp`, `pto-kernels/csrc/kernel/kernel_simple_matmul.cpp`, and `pto-isa-master/include/pto/npu/a2a3/TStore.hpp`.

## Current status (2026-04-23)

| Milestone | State |
|-----------|--------|
| Layout under `pa_gqaonly_pto/` (tiling Python, tests, bench, host wrapper) | Done — same surface as CCE tree |
| `bisheng` shared library `pa_lib.so` | Done — `bash compile.sh` |
| `test_pa_accuracy.py` | **PASS** all fp16 cases on NPU (`ASCEND_DEVICE_ID=0`, post–L0C-store port) |
| `bench_pa_performance.py` | Runs; IFA comparison lines print as before |
| PTO-ISA headers on cube TU | Done — `-I${PTO_ISA_ROOT}/include` + `kernel/pa_gqa_pto_tile_helpers.hpp` |
| PTO-ISA headers on vector TU | Done — same header under `__DAV_C220_VEC__` |
| Cube QK / PV matmul: `mad` → `pto::TMatmul` | **Done** — `pa_pto::tmatmul_fp32acc<IN_DTYPE>` |
| Cube L0C → GM FP32 ND | **Done** — `pa_pto::tstore_l0c_fp32_nd` wraps `pto::TSTORE` (`TileAccCompact` + `GlobalTensor`); oversize `n` uses AscendC fallback (see below) |
| Replace `copy_gm_to_cbuf` / `load_cbuf_to_*` with `pto::TLOAD` / `pto::TMOV` | **Not done** — still AscendC helpers in `pa_kernel.cce` |
| Vector softmax / mask (`vadd`, `vexp`, …) → `pto::TADD` / `TEXP` / … | **Not done** — UB repeat/strided blocks; mirror `pto_macro_fa_softmax.hpp` patterns |

## L0C → GM implementation notes

- Entry point: `pa_pto::tstore_l0c_fp32_nd` in `kernel/pa_gqa_pto_tile_helpers.hpp` (cube TU only).
- Uses `TileAccCompact<float, kPaPtoMadMaxM, kPaPtoL0cStoreMaxN, DYNAMIC, DYNAMIC>` so **fractal source stride** follows PTO Normal-compact rules (aligned with legacy `RoundUp<16>(m)` behavior).
- PTO `CheckAcc2gm` requires static Acc tile `Cols` in **[1, 4095]** and fractal column alignment, so **`kPaPtoL0cStoreMaxN = 4080`** (largest 16-aligned value ≤ 4095). Matmul caps stay **`kPaPtoMadMaxN = 4096`**.
- If `mActual > kPaPtoMadMaxM` or `nActual > kPaPtoL0cStoreMaxN`, the helper **falls back** to the previous `set_nd_para` + `copy_matrix_cc_to_gm` sequence so correctness is preserved for extreme tilings.
- Call sites in `pa_kernel.cce` no longer use `pa_l0c_to_gm_nd_fp32` (removed).

## AscendC intrinsic → PTO-ISA mapping (target)

| AscendC / kernel_operator (current) | PTO-ISA direction | Notes |
|-------------------------------------|-------------------|--------|
| `copy_gm_to_cbuf` / `pa_gm_to_l1_nd_nd` | `pto::TLOAD` | … |
| `pa_gm_to_l1_nd_nz` | `pto::TLOAD` (NZ) | … |
| `load_cbuf_to_ca` / `pa_l1_to_l0_a_vector` | `pto::TMOV` | … |
| `load_cbuf_to_cb` / `pa_l1_to_l0_b_vector` | `pto::TMOV` | Transpose paths still AscendC |
| `mad` (QK and PV) | **`pto::TMatmul` (done)** | `pa_pto::tmatmul_fp32acc` |
| `copy_matrix_cc_to_gm` / L0C→GM ND FP32 | **`pto::TSTORE` (done; fallback above)** | `pa_pto::tstore_l0c_fp32_nd` |
| `vadd`, `vexp`, `vcmax`, … | `pto::TADD`, `pto::TEXP`, … | **Not started** on vector core |

## Incremental migration plan (suggested order)

1. ~~Cube QK/PV matmul~~ — **Done.**
2. ~~Cube L0C → GM~~ — **Done** (`tstore_l0c_fp32_nd`).
3. **Cube data path** — `pa_gm_to_l1_*` / `pa_l1_to_l0_*` → `TLOAD` / `TMOV` where layouts match PTO; keep AscendC for odd transpose paths until parity harness exists.
4. **Vector softmax** — replace contiguous-friendly blocks first; keep `pipe_barrier(PIPE_V)` like `pto_macro_fa_softmax.hpp` for A2A3 vec.
5. **Split-KV / MHA paths** — tiling keys 16/17; keep tiling Python identical (`pa_tiling.py`).

## Repro commands

```bash
export PTO_ISA_ROOT=/workdir/pto-isa-master   # optional; default in compile.sh
cd /workdir/cann_ops_standalone/paged_attention/pa_gqaonly_pto
bash compile.sh
export ASCEND_DEVICE_ID=0   # or 4–7 per cluster policy
python3 test_pa_accuracy.py
python3 bench_pa_performance.py --device 0 --warmup 5 --iters 20
```

## Session log

- Established `pa_gqaonly_pto` from `atb_pa_gqaonly_cce` sources; added PTO include path and cube-side `pa_gqa_pto_tile_helpers.hpp`.
- Verified `test_pa_accuracy.py` all fp16 cases PASS on NPU.
- Verified `bench_pa_performance.py` runs vs IFA baseline.
- Routed both cube `mad` sites through `pa_pto::tmatmul_fp32acc`.
- **2026-04-23 (this session):** Implemented `pa_pto::tstore_l0c_fp32_nd` using `pto::TSTORE` + `TileAccCompact` + dynamic `GlobalTensor`/`TileShape2D` ND; documented PTO Cols≤4095 limit and `n>4080` fallback; removed `pa_l0c_to_gm_nd_fp32`; re-ran `compile.sh`, full fp16 accuracy suite, and `bench_pa_performance.py` — all green.
