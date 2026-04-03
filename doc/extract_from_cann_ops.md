# Extracting standalone Ascend examples from a large CANN ops repo

This note summarizes patterns that worked when turning an `ops-transformer` kernel (and tests) into a **self-contained** tree under `atb_standalone/` that **bisheng**-compiles to a `.so` and runs **on-device** tests via **ctypes** + **torch_npu**—without linking the full ops project build.

Use it as a checklist for similar extractions (other attention ops, matmul wrappers, etc.).

---

## 1. What “standalone” means

- **Compile**: One or more `bisheng` invocations with `-fPIC -shared -xcce`, NPU arch (e.g. `--npu-arch=dav-2201`), and the same `-I` layout as working examples (`mla_prefill`, `sync_c2v`, …).
- **Run**: Python loads the `.so`, passes device pointers + stream, compares to a **torch reference** on NPU (or CPU golden).
- **No** dependency on the ops repo **include path** or **CMake**—kernel sources should live **inside** the standalone directory (e.g. `op_kernel/*.h`).

---

## 2. Vendor the kernel sources

- Copy the **op_kernel** (and any strictly local headers) into the standalone package, e.g. `chunk_gdn/op_kernel/`.
- Add **`-I"${SCRIPT_DIR}/op_kernel"`** (and `-I"${SCRIPT_DIR}"` for shims) so includes like `"chunk_*.h"` resolve.
- Do **not** leave `#include "../../ops-transformer/..."` in standalone code; refresh vendored files when upstream changes.

---

## 3. Shim headers for CANN layout mismatches

Upstream kernels often include names that match the **in-tree** layout, not necessarily CANN 8.x install paths. Common shims:

| Expected include | Typical shim |
|------------------|--------------|
| `kernel_tiling/kernel_tiling.h` | Re-export `adv_api/kernel_tiling.h` + device-side `GET_TILING_DATA` (use `__builtin_memcpy` from GM, not `memcpy` / bad `reinterpret_cast` across address spaces). |
| `lib/matmul_intf.h` | `#include "adv_api/matmul/matmul_intf.h"` |
| `kernel_vec_intf.h` / `kernel_cube_intf.h` | Forward to `kernel_operator.h` or the matching `basic_api/...` headers. |
| `kernel_operator_list_tensor_intf.h` | Forward to `basic_api/kernel_operator_list_tensor_intf.h` |

Keep shims **minimal**—only what the extracted kernel actually includes.

---

## 4. Host launch surface (`ctypes`)

- Expose **`extern "C"`** entry points: e.g. `call_kernel(blockDim, stream, ...)` that launches `<<<blockDim, nullptr, stream>>>`.
- Pointers: host code often uses `void*` / `uint8_t*`; cast to `(__gm__ uint8_t*)` at the launch site if the compiler requires it.
- **`tilingGM`**: allocate device memory with correct **alignment** (e.g. `int64` tensor) and copy struct bytes from a `ctypes` layout matching the device struct **exactly** (pack, field order, `TCubeTiling` size).
- **`__CCE_KT_TEST__`**: not required for PyTorch-driven launches (see simple standalone kernels that omit it).

---

## 5. Workspace and tiling

- **System workspace**: Ascend runtime often expects a **~16 MB prefix** before “user” workspace; size buffers as `16MB + inter + stage` if the kernel uses `GetUserWorkspace`.
- Fill **`TCubeTiling`** with non-zero, coherent shape fields (`M`, `N`, `Ka`, `Kb`, `singleCore*`, `base*`, `shareL0CSize`, `transLength`, batch fields, etc.)—empty tiling is a frequent source of device faults.
- If the upstream project has **`op_host` tiling** code, mirror its matmul tiling or reuse a small **host-side** generator (see `gen_chunk_gdn_tiling.cpp` pattern) when you need parity.

---

## 6. FFTS / mix-mode (AIC + AIV)

Kernels using **`CrossCoreWaitFlag` / `CrossCoreSetFlag`** need a valid sync base:

- **Host**: `rtGetC2cCtrlAddr` (from runtime) to get the control address.
- **Device** (start of kernel): `SetSyncBaseAddr((unsigned long)fftsAddr)`, often plus `SetAtomicNone()`, `SetMaskNorm()`.

**Block dimensions**: for the mix kernel (`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`), use **`blockDim = ai_core_num`** (one block per AI core). Wrong `blockDim` can yield **partial execution** or odd `GetBlockIdx()` mapping—validate with a tiny “meta” kernel if needed.

---

## 7. Kernel mode macros

- Prefer **`KERNEL_TASK_TYPE_DEFAULT(...)`** inside the kernel entry if you must force AIV-only vs mix mode—**`AscendC::SetKernelMode`** may not be available in all bisheng link environments.
- **`ASCEND_IS_AIC` / `ASCEND_IS_AIV`**: branches are coupled; do not “split” them into separate tiny kernels without preserving sync—instead split by **phase** and write intermediates to GM for checks.

---

## 8. Debugging strategy: staged probes

For large kernels, avoid debugging the full op first:

1. **Smoke**: copy/cast or a few vector ops on GM (compile + run).
2. **Phase wrappers**: one `.cpp` per stage that constructs the same `GlobalTensor`s / `TPipe` / matmul objects as upstream and calls **`StageN::Init` / `Process`** unchanged.
3. **Golden**: torch reference for **that phase only** (or chain references).
4. Expand until the **full** `chunk_gated_delta_rule`-style entry matches.

This isolates tiling, FFTS, and numerical issues faster than a single monolithic test.

---

## 9. Numerical and stability notes

- **BF16 vs FP32 golden**: expect larger **max** error on outputs; use **mean abs** error too to ensure only a few outliers blow the max (bf16 cast can add ~0.5 max abs on rare elements while mean stays small).
- **Intermediate NaNs**: raw random Q/K can destabilize chunk attention math; **L2-normalize** Q/K like upstream golden tests if you see NaNs in `k_cum_decay` / `v_inner` while other tensors look fine.
- **`ASCENDC_CPU_DEBUG`**: ignore for NPU-only extraction.

---

## 10. Documentation and hygiene

- **`README.md`**: build command, `NPU_ID`, and what each `.so` exports.
- **`compile*.sh`**: pin **all** `-I` lines so a clean machine can reproduce.
- Prefer **one** place for ctypes struct definitions (e.g. `test_chunk_gdn.py`) + small **shared** test helpers (`chunk_gdn_common.py`) to avoid drift.

---

## 11. Quick verification order

1. `bisheng` completes with **no** missing includes.
2. `python` test: no **507057** / **507015** class errors after FFTS + `blockDim` fixes.
3. Staged tests pass, then full-kernel test against torch golden.

This order saves time: link/include errors first, then sync/scheduling, then numerical tuning.

## 12. Find free NPU devices

Command `npu-smi info` prints out the usage of current NPUs. Avoid using NPUs that are being used by other processes. Switch to a free device using, for example, `torch.npu.set_device("npu:7")` (assume NPU:7 is free).
