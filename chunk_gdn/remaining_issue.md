# Remaining issue: single-launch fused `chunk_gdn_lib.so`

## What works

- **`stage1_lib.so`**, **`stage2_lib.so`**, **`stage3_lib.so`**: each launches from Python via ctypes, uses **`rtGetC2cCtrlAddr` + `SetSyncBaseAddr`** (FFTS) and **`KERNEL_TYPE_MIX_AIC_1_2`** where applicable. On-device tests **`test_stage1.py`**, **`test_stage2.py`**, **`test_stage3.py`** complete successfully with **custom AscendC kernels**.
- **Staged end-to-end path**: **`test_chunk_gdn.py`** now runs a host-orchestrated **Stage1 -> Stage2 -> Stage3** sequence using the working stage `.so` files instead of the fused single-launch kernel. That path passes on-device correctness checks against the bf16 torch reference.
- **`benchmark_chunk_gdn.py`**: the end-to-end benchmark now measures the staged custom pipeline successfully and reports strong speedups vs the PyTorch reference.
- **`benchmark_stage_kernels.py`**: stage-isolated timing works on a broader validated shape matrix and reports per-stage TFLOP/s / operand GiB/s in **`benchmark_stage_kernels.csv`**.
- **Core-count fix**: `ai_core_num_from_device()` now uses the full **`cube_core_num`**. That materially improved measured stage and end-to-end throughput.

## What fails

- **`chunk_gdn_lib.so`** (the fused single-launch `chunk_gated_delta_rule` in **`chunk_gdn_wrapper.cpp`** / **`chunk_gdn_kernel_entry.cpp`**) is still not stable on the same host where all staged kernels succeed. Direct launch still fails with runtime error **`507057`**, typically surfacing at **`torch.npu.synchronize()`** after the kernel launch.

So: **the supported end-to-end benchmark path works today, but only by chaining the three working stage kernels from host code.** The unresolved issue is specifically the **single-launch fused kernel**.

## Symptom details (for plog / vendor triage)

- Error class: **SUSPECT REMOTE ERROR**, code **507057** (`0x7B9C1` / variants in logs).
- Staged and fused builds use the same **`--npu-arch=dav-2201`** bisheng flags in **`compile*.sh`**.
- Fused kernel entry was aligned with the stages (FFTS base, **`KERNEL_TYPE_MIX_AIC_1_2`**, **`SetAtomicNone` / `SetMaskNorm`**), and host tiling **`build_tiling_and_workspace`** was adjusted so matmul **`singleCoreM/N/K`** match **`default_matmul_tiling`**.
- The fused single-launch path still fails even on tiny stable shapes such as **`B=1, T=64, nk=nv=dk=dv=64, chunk=64`**, so this does **not** look like only a large-shape utilization/resource problem.
- Additional fused experiments tried and still failed: varying **`blockDim`**, re-initializing matmul objects after pipe resets, prebuilding masks from host-side workspace, and replacing the fused body with a one-kernel Stage1 -> Stage2 -> Stage3 composition.

## Likely directions (hypotheses for future work)

1. **Resource / code size**: The fused object links **Stage1 + Stage2 + Stage3 + full `CGDR` pipeline** in one binary. It may exceed limits for **kernel cache**, **launch args**, or **on-chip** resources where small stage `.so` files succeed. *Mitigation to try:* split fusion boundaries, reduce live ranges, or match upstream’s exact **blockDim** / **task type** split.
2. **Workspace / tiling parity with upstream op_host**: Standalone **`build_tiling_and_workspace`** still uses a handcrafted tiling blob. *Mitigation:* diff against real **`GetTiling`** output from CANN for the same shapes and drive the fused path with true op-host tiling.
3. **FFTS / cross-core lifecycle**: Stages set FFTS once per process; fused kernel sets it inside the entry. If the runtime expects a different **order** or **per-context** setup, mix-mode sync could fault. *Mitigation:* compare with upstream full-op launch sequence (host runtime).
4. **CANN / driver version**: **507057** is generic; collect **`plog`** as ACL suggests and test another **CANN** drop on the same hardware.
5. **Wrapper semantics vs full op semantics**: The working stage wrappers are still simplified (for example, Stage2 / Stage3 effectively process one sequence/state at a time). That is good enough for the current staged benchmark path, but it means the single-launch fused kernel still needs a more faithful host/runtime integration than the standalone wrappers currently provide.

## Concrete next steps

1. Reproduce with **`ASCEND_LAUNCH_BLOCKING=1`** and **`ASCEND_GLOBAL_LOG_LEVEL=1`** (or your site’s ACL debug flags); capture **device plog** around the fault.
2. Generate or extract **real op-host tiling** for the fused kernel and compare it with the current standalone **`build_tiling_and_workspace`** output for the same shapes.
3. Compare **`.o` / `.so` size** and kernel metadata between **`chunk_gdn_lib.so`** and the stage libraries (e.g. `nm`, vendor tools, cache metadata if available).
4. If the fused path starts passing, benchmark it directly against the staged path to see whether single-launch fusion beats the current host-orchestrated Stage1 -> Stage2 -> Stage3 flow.

## Files to read first

| Area | File |
|------|------|
| Fused entry | `chunk_gdn_kernel_entry.cpp`, `chunk_gdn_wrapper.cpp` |
| Full operator | `op_kernel/chunk_gated_delta_rule.h` |
| Host tiling | `test_chunk_gdn.py` → `build_tiling_and_workspace` |
| Working staged path | `test_chunk_gdn.py`, `stage1_kernel.cpp`, `stage2_kernel.cpp`, `stage3_kernel.cpp` |
| Benchmarks | `benchmark_chunk_gdn.py`, `benchmark_stage_kernels.py` |
