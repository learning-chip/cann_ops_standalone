# Pull request: Standalone `chunk_gdn` + extraction guide

## How to verify

```bash
cd chunk_gdn
bash ./compile.sh
bash ./compile_stage1.sh && python test_stage1.py
bash ./compile_stage2.sh && python test_stage2.py
bash ./compile_stage3.sh && python test_stage3.py
python test_chunk_gdn.py
python benchmark_chunk_gdn.py
python benchmark_stage_kernels.py
```

Set `NPU_ID` if not using device `0`.

### Measured performance (snapshot)

From `python benchmark_chunk_gdn.py` on Ascend **910B2**, `NPU_ID=7`, **2026-04-03**:

| Case | T | nk/nv | dk/dv | custom ms | custom TFLOP/s | custom GiB/s | torch ref ms | torch ref TFLOP/s | torch ref GiB/s | speedup |
|------|---|-------|-------|-----------|----------------|--------------|--------------|-------------------|-----------------|---------|
| gdn_b1_s4096_h4 | 4096 | 4 / 4 | 64 / 64 | 0.555 | 2.17 | 108.74 | 47.87 | 0.0252 | 1.26 | 86.2x |
| gdn_b1_s16384_h4 | 16384 | 4 / 4 | 64 / 64 | 1.930 | 2.50 | 122.82 | 153.01 | 0.0316 | 1.55 | 79.3x |
| gdn_b1_s65536_h4 | 65536 | 4 / 4 | 64 / 64 | 8.208 | 2.35 | 114.95 | 860.39 | 0.0225 | 1.10 | 104.8x |

These end-to-end custom numbers are for the **working staged custom pipeline** (`stage1_lib.so` -> `stage2_lib.so` -> `stage3_lib.so` launched from Python), not the unresolved single-launch fused `chunk_gdn_lib.so`.

Details: `chunk_gdn/README.md`, `benchmark_chunk_gdn.csv`.

Per-stage custom-kernel performance (TFLOP/s and operand GiB/s): `python benchmark_stage_kernels.py` → `benchmark_stage_kernels.csv`. Best observed utilization from the current broad sweep:

- Stage1: up to **3.45 TFLOP/s**, **219.65 GiB/s**
- Stage2: up to **14.62 TFLOP/s**, **268.39 GiB/s**
- Stage3: up to **10.63 TFLOP/s**, **273.62 GiB/s**

Fused `chunk_gdn_lib.so` runtime issues vs working stages: `chunk_gdn/remaining_issue.md`.

## Summary

Adds a self-contained **chunk gated delta rule** example under `cann_ops_standalone/chunk_gdn/` that builds with **bisheng** into shared libraries and runs **on-device** tests via **ctypes** and **torch_npu**, without depending on the full `ops-transformer` build. Includes working **staged** Stage1 / Stage2 / Stage3 probe binaries, a staged end-to-end benchmark path, shared Python helpers, and a short extraction guide for similar kernels.

## Motivation

- Run and debug the kernel outside the full ops-transformer CMake graph.
- Iterate with small `.so` targets and pytest-style scripts (similar to `mla_prefill`, `matmul_cce`, etc.).
- Keep kernel sources **vendored** under the example tree so CI or other checkouts do not depend on a sibling repo path.

## What changed

### `chunk_gdn/` — standalone build

- **`op_kernel/`**: Vendored AscendC kernel headers and `chunk_gated_delta_rule.cpp` (no `../../ops-transformer/...` includes).
- **Compile scripts**: `compile.sh`, `compile_stage{1,2,3}.sh` add `-I"${SCRIPT_DIR}/op_kernel"` plus existing CANN include paths.
- **Entry points**: `chunk_gdn_wrapper.cpp` + `chunk_gdn_kernel_entry.cpp` expose `call_kernel`; stage wrappers expose `call_stage1`, `call_stage2`, `call_stage3` with **FFTS** setup (`rtGetC2cCtrlAddr` on host, `SetSyncBaseAddr` / `SetAtomicNone` / `SetMaskNorm` on device).
- **Shims** (unchanged behavior, comments clarified): `kernel_tiling/kernel_tiling.h`, `lib/matmul_intf.h`, `kernel_vec_intf.h`, `kernel_cube_intf.h`, `kernel_operator_list_tensor_intf.h`.
- **`chunk_gdn_common.py`**: Shared helpers (`check_close` with optional `mean_tol`, tiling helpers, workspace size helpers). `ai_core_num_from_device()` now uses the full **`cube_core_num`**.
- **Tests**: `test_chunk_gdn.py`, `test_stage1.py`, `test_stage2.py`, `test_stage3.py`. `test_chunk_gdn.py` now validates the **staged custom end-to-end path** against the bf16 torch reference. Staged tests **L2-normalize** Q/K (avoids NaNs in Stage1 intermediates); Stage3 asserts both **max** and **mean** absolute error vs reference.
- **Benchmarks**:
  - `benchmark_chunk_gdn.py`: working end-to-end timing for the staged custom pipeline vs torch reference
  - `benchmark_stage_kernels.py`: broader validated stage-kernel sweep across `B`, `nk/nv`, and sequence length
- **Gamma propagation fix**: `stage2_kernel.cpp` and `stage3_kernel.cpp` now honor `tilingData.hasGamma`, which fixed the gamma-enabled staged end-to-end path.
- **`gen_chunk_gdn_tiling.cpp`**: Include path updated to local `chunk_gated_delta_rule_tiling_data.h`; build note for `-I.../op_kernel`.
- **`README.md`**: Build/run instructions and notes (device id, FFTS, normalization).

### Documentation

- **`doc/extract_from_cann_ops.md`**: Checklist and lessons for extracting similar standalone Ascend examples (vendoring, shims, workspace/tiling, FFTS/mix-mode, staged debugging, numerics).
- **`chunk_gdn/remaining_issue.md`**: Current status of the unresolved single-launch fused `chunk_gdn_lib.so` issue, plus concrete next steps.

- **`test_chunk_gdn.py`**: Comment wording updated (“upstream” instead of naming another repo).

## Notes for reviewers

- Kernel logic in `op_kernel/*.h` is **copied** from upstream; functional changes there should stay minimal.
- The supported end-to-end path in this standalone tree is currently the **host-orchestrated staged pipeline**, not the unresolved single-launch fused kernel.
- Broad benchmarking showed the best-performing stable region on this host is still `dk=dv=64`, `chunk=64`, with utilization improving mainly by increasing `nk=nv`.

## Follow-ups (non-blocking)

- Fix the single-launch fused `chunk_gdn_lib.so` path (`507057`) using real op-host tiling and/or deeper runtime parity checks.
- Tighten staged correctness thresholds if the bf16 cast envelope is reduced later.
- Optional: compile `gen_chunk_gdn_tiling.cpp` from a small `compile_gen.sh` if host-side tiling generation is needed in CI.

---

*Temporary PR description file; remove or relocate after merge.*
