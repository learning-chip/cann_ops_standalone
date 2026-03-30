# Call Stack Notes: `mla_prefill`, `fa_mla`, and `mla`

This document captures practical call stacks from Python unit tests to the final CCE kernel file, so future tracing is faster.

Verified on branch http://gitcode.com/cann/ascend-transformer-boost/blob/br_release_cann_8.5.0_20260527/

## 1) Example A: `mla_prefill` path (`test_mla_prefill_rope.py`)

### Test entry
- File: `tests/apitest/kernelstest/mix/test_mla_prefill_rope.py`
- The test sets:
  - `OP_NAME = "MLAOperation"`
  - `OP_PARAM["type"] = 1` (prefill mode)
- Then calls `self.execute(...)` from `OpTest`.

### Python wrapper -> Torch custom class
- File: `tests/apitest/kernelstest/op_test.py`
- `set_param()` builds JSON op description and constructs:
  - `torch.classes.MkiTorch.MkiTorch(json.dumps(op_desc))`
- `execute()` moves tensors to NPU and calls:
  - `self.mki.execute(in_tensors_npu, out_tensors_npu)`

### Ops runner builds op node
- File: `src/ops/ops_infer/multi_latent_attention/multi_latent_attention_ops_runner_prefill.cpp`
- Creates node:
  - `mlaNode.opDesc = {0, "MLAOperation", asdParam}`
  - `asdParam.type = AtbOps::OpParam::MLA::PREFILL_SPLIT_CACHE`

### Operation chooses kernel name
- File: `src/kernels/mixkernels/multi_latent_attention/mla_operation.cpp`
- In `GetBestKernel(...)`:
  - `PREFILL_SPLIT_CACHE` + fp16 -> `GetKernelByName("MLAPrefillKernel")`
  - bf16 or causal-mask prefill -> `GetKernelByName("MLAPrefillBF16Kernel")`

### Kernel name -> CCE file binding
- File: `src/kernels/mixkernels/multi_latent_attention/CMakeLists.txt`
- Mapping:
  - `MLAPrefillKernel` -> `op_kernel/mla_prefill.cce`
  - `MLAPrefillBF16Kernel` -> `op_kernel/mla_prefill_high_precision.cce`

### Final CCE entry symbol
- File: `src/kernels/mixkernels/multi_latent_attention/op_kernel/mla_prefill.cce`
- Entry function:
  - `extern "C" __global__ __aicore__ void mla_prefill(...)`

### Conclusion
- For fp16 prefill tests in `test_mla_prefill_rope.py`, the execution path reaches:
  - `src/kernels/mixkernels/multi_latent_attention/op_kernel/mla_prefill.cce`

---

## 2) Example B: `fa_mla` path (`test_fa_mla.py`)

### Test entry
- File: `tests/apitest/kernelstest/mix/test_fa_mla.py`
- This test file uses:
  - `OP_NAME = "UnpadFlashAttentionOperation"`
  - `OP_PARAM["type"]` is typically `2008` or `2010` in this file
    - `2008 = MULTI_LATENT_ATTENTION_COMBINE_CACHE`
    - `2010 = MULTI_LATENT_ATTENTION_HIGH_PRECISION_COMBINE_CACHE`
- Then calls `self.execute(...)` (same `OpTest` wrapper path as above).

### Operation chooses kernel name
- File: `src/kernels/mixkernels/unpad_flash_attention/unpad_flash_attention_operation.cpp`
- In `GetBestKernel(...)`, for:
  - `type == MULTI_LATENT_ATTENTION_COMBINE_CACHE` or
  - `type == MULTI_LATENT_ATTENTION_HIGH_PRECISION_COMBINE_CACHE`
- It returns:
  - `GetKernelByName("MultiLatentAttentionEncoderCombineCacheKernel")`

### Type values source
- File: `src/kernels/include/atbops/params/unpad_flash_attention.h`
- Enum confirms:
  - `MULTI_LATENT_ATTENTION_COMBINE_CACHE = 2008`
  - `MULTI_LATENT_ATTENTION_HIGH_PRECISION_COMBINE_CACHE = 2010`

### Kernel name -> CCE file binding
- File: `src/kernels/mixkernels/unpad_flash_attention/CMakeLists.txt`
- Mapping:
  - `MultiLatentAttentionEncoderCombineCacheKernel` ->
    `op_kernel/multi_latent_attention_mix.cce`

### Final CCE entry symbol
- File: `src/kernels/mixkernels/unpad_flash_attention/op_kernel/multi_latent_attention_mix.cce`
- Entry function:
  - `extern "C" __global__ __aicore__ void multi_latent_attention_mix(...)`

### Important clarification
- `test_fa_mla.py` does **not** go to:
  - `src/kernels/mixkernels/multi_latent_attention/op_kernel/mla.cce`
- It uses the `UnpadFlashAttentionOperation` family and lands in:
  - `src/kernels/mixkernels/unpad_flash_attention/op_kernel/multi_latent_attention_mix.cce`
  - (or related unpad MLA CCE files for other op types/kernel choices).

---

## 3) Example C: `mla.cce` path (`test_paged_attention_mla_split.py`)

### Which unit test file calls `mla.cce`?
- A direct kernelstest example is:
  - `tests/apitest/kernelstest/mix/test_paged_attention_mla_split.py`
- This file contains many `MLAOperation` test cases with:
  - `OP_PARAM["type"] = 0` (split-cache mode)

### Test entry
- The test sets:
  - `OP_NAME = "MLAOperation"`
  - `OP_PARAM["type"] = 0`
- Then calls `self.execute(...)` from `OpTest`.

### Operation chooses kernel name
- File: `src/kernels/mixkernels/multi_latent_attention/mla_operation.cpp`
- In `GetBestKernel(...)`:
  - `type == OpParam::MLA::SPLIT_CACHE` -> `GetKernelByName("MLAKernel")`

### Type value source
- File: `src/kernels/include/atbops/params/mla.h`
- Enum confirms:
  - `SPLIT_CACHE = 0`
  - `PREFILL_SPLIT_CACHE = 1`

### Kernel name -> CCE file binding
- File: `src/kernels/mixkernels/multi_latent_attention/CMakeLists.txt`
- Mapping:
  - `MLAKernel` -> `op_kernel/mla.cce`

### Final CCE entry symbol
- File: `src/kernels/mixkernels/multi_latent_attention/op_kernel/mla.cce`
- This is the final kernel source selected for `MLAOperation` with `type=0`.

### Conclusion
- `test_paged_attention_mla_split.py` is a concrete unit test file that drives execution to:
  - `src/kernels/mixkernels/multi_latent_attention/op_kernel/mla.cce`

---

## Quick mental model for future tracing

1. Start from test `OP_NAME` + `OP_PARAM["type"]`.
2. Find `Operation::GetBestKernel(...)` for that `OP_NAME`.
3. Read CMake `add_kernel(... <cce> <KernelClassName>)` mapping.
4. Confirm final CCE entry symbol (`extern "C" ...`).

If `OP_NAME` differs (`MLAOperation` vs `UnpadFlashAttentionOperation`), expect different CCE directories and kernel files even when both are "MLA-related".
