# Progress Report: `atb_pa_gqaonly_cce` Raw Intrinsic Port

## Goal

Single-source-style kernel build for paged attention: `paged_attention_mask_mix.cce` includes
`paged_attention_decoder_nd_common.cce`, with shared helpers in `op_kernel/pa_kernel_inline.h` (raw
`__gm__` / `__cbuf__` / `__ubuf__` / `__ca__` / `__cb__` / `__cc__` pointers, no `LocalTensorView` /
`RawAddrTensorView`, thin wrappers inlined to intrinsics where appropriate).

---

## Current Status (2026-04-22)

**Done.** `bash compile.sh` succeeds. All six fp16 cases in `test_atb_pa_standalone.py` pass
(`max_err` well under 0.02). `bench_pa_standalone.py` runs successfully.

The directory `kernels/utils/kernel` has been **removed** from this package; nothing in `compile.sh`
or the `.cce` sources referenced it anymore.

---

## Correctness fixes (chronological)

### UB→GM align dispatch (earlier port)

`ub_to_gm_align` must use `copy_ubuf_to_gm_align_b8` / `b16` / `b32` (byte-oriented block lengths),
not plain `copy_ubuf_to_gm`, or GM gets corrupted on padded paths.

### L0C→GM base pointer (this session)

`ProcessQK` / `ProcessPV` are called with **already-offset** GM bases (`s_gm + block/…`,
`o_tmp_gm + block/…`). After the raw-pointer port, `pa_l0c_to_gm_nd_fp32` incorrectly used the
class members `s_gm` / `o_tmp_gm` (root pointers), dropping the per-call offset. That wrote QK
scores and partial outputs to the wrong GM region → large numerical errors.

**Fix:** write with `s_gm_tensor + s_gm_offset` and `o_tmp_gm_tensor + o_temp_gm_offset`, matching
the original `GlobalTensor` indexing on the parameter tensors.

---

## Verification commands

Prefer a free device (see `npu-smi info`), e.g.:

```bash
cd /workdir/cann_ops_standalone/atb_pa_gqaonly_cce
bash compile.sh
ASCEND_DEVICE_ID=2 python3 test_atb_pa_standalone.py
ASCEND_DEVICE_ID=2 python3 bench_pa_standalone.py
```

---

## File map

| Area | Path |
|------|------|
| Inlined helpers | `op_kernel/pa_kernel_inline.h` |
| Cube + vec logic | `op_kernel/paged_attention_decoder_nd_common.cce` |
| Entry + include decoder | `op_kernel/paged_attention_mask_mix.cce` |
| Host launch | `paged_attention_wrapper.cpp`, `compile.sh` |

---

## Maintenance notes

- When porting UB↔GM pad/align AscendC APIs, match the **exact** intrinsic family
  (`copy_*_align_*` vs `copy_*`) and byte vs element length conventions.
- When a kernel function takes a sliced `__gm__ *` base, all GM writes inside that function must use
  that parameter, not a separate global “root” pointer, unless the offset is reapplied consistently.
