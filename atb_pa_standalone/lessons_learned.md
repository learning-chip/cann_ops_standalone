# Lessons Learned: Extracting a Standalone AscendC Kernel

This document records hard-won lessons from extracting, compiling, and testing
the paged-attention kernel from `ascend-transformer-boost` as a standalone
`ctypes`-callable shared library on Ascend 910B.

---

## 1. FFTS Hardware State Is Global and Persistent

### What FFTS is
FFTS (Fast Functional Task Synchronization) is a hardware mechanism on the
Ascend 910B for AiC↔AiV (Cube↔Vector) core synchronization within a single
CCE kernel launch. The kernel uses three primitives:
- `SetFftsBaseAddr(addr)` — points the hardware at the counter buffer
- `FftsCrossCoreSync<PIPE, N>(flag_id)` — atomically increments counter `flag_id`
- `WaitFlagDev(flag_id)` — polls until `flag_id` reaches an expected value

The counter buffer address is obtained via `rtGetC2cCtrlAddr` and is a fixed
hardware address shared by all kernels on that device.

### The accumulation problem
FFTS counters are **not reset between kernel launches**. Every kernel call
increments certain counters, and `WaitFlagDev` compares against an absolute
(not relative) target. After enough prior calls, the counter may already exceed
the target when the next kernel starts, causing `WaitFlagDev` to return
immediately before the peer cores finish — producing silent data races and wrong
output (or NaN from uninitialized workspace reads).

### Counter cycle period
In practice the relevant flags cycle with period **4**: out of 4 consecutive
calls of the same kernel shape, exactly 1 produces correct output and the others
produce wrong or NaN results. Running 4 warmup calls of the **exact** target
shape resets the counter to phase 0, making the 5th call (the measurement)
always correct.

### Path-specific flag reuse
Different tiling configurations share some flag indices. For example,
`QK_READY_DECODER = 3` is used both by the standard BN path (non-split-KV) as
an AiC→AiV signal and by the BNS path (split-KV) as an all-cores barrier. This
means running a BN-path kernel contaminates the flag 3 state for the BNS-path
kernel, and vice versa.

### What does NOT work
| Attempted fix | Outcome |
|---------------|---------|
| `rtMemset(ffts_addr, ...)` on host | Segmentation fault — hardware FFTS memory is not writable from CPU |
| Pass Python-allocated NPU buffer as `sync` argument | Hardware hang — the CCE instructions require the specific hardware address |
| Re-ordering tests within one process | Still fails; any intervening NPU op (even `torch.zeros`) shifts the counter phase |
| Single generic warmup (different shape) | Not sufficient; shape changes alter the counter increment pattern |

### The reliable fix
1. **Subprocess isolation**: run each test case in a fresh OS process
   (`subprocess.run`). Each subprocess starts with fresh FFTS state.
2. **4× shape-specific warmup**: within each subprocess, run the exact target
   kernel shape 4 times with zero inputs. This exhausts one full counter cycle
   and lands at phase 0.
3. **Measure before reference**: call the custom kernel immediately after warmup
   with no intervening NPU operations, then run the reference kernel for
   comparison.

---

## 2. `torch.zeros` / `torch.randn` Are NPU Operations

Any tensor construction on an NPU device (`torch.zeros(...)`, `torch.randn(...)`,
`.zero_()`, `.clone()`) enqueues an NPU command. These commands contribute to
the FFTS counter accumulation just like kernel launches. Specifically:
- Tensor creation between the last warmup call and the measurement call can
  shift the counter phase and break correctness.
- **Fix**: allocate all test tensors *before* the warmup loop, synchronize with
  `torch.npu.synchronize()`, then run the 4× warmup, then immediately launch
  the kernel under test.

---

## 3. `g_tilingKey` Must Be Initialized in Standalone Mode

The ATB/MKI framework sets the `[[workgroup_local]] __gm__ uint64_t g_tilingKey`
variable before dispatching the kernel. This variable is checked by
`TILING_KEY_IS(n)` macros. In standalone mode (direct kernel launch without MKI)
this variable is never initialized, so all `TILING_KEY_IS` checks return false
and the kernel is a no-op.

**Fix** (one line added to `paged_attention_mask_mix.cce`):
```cpp
g_tilingKey = (uint64_t)(uint32_t)(*((__gm__ int32_t *)tiling_para_gm + AtbOps::TILING_KEY_ID));
```
Read the tiling key directly from the tiling buffer before any dispatch.
Also replace the `TILING_KEY_IS(n)` macro chain with explicit
`tiling_key_val == n` comparisons so the compiler doesn't need the MKI context.

---

## 4. Workspace Buffer Sizing

The kernel uses 8 global memory scratch buffers:

| Buffer | Purpose | Size formula |
|--------|---------|--------------|
| `s_gm` | QK score scratch (AiC) | `block_dim × 64KB × 4 bytes` |
| `p_gm` | Softmax numerator scratch | `block_dim × 64KB × 2 bytes` |
| `o_tmp_gm` | Output accumulation | `block_dim × 64KB × 8 bytes` |
| `go_gm` | Output scratch (AiV) | `block_dim × 64KB × 4 bytes` |
| `o_core_tmp_gm` | Per-core partial O (split-KV) | `⌊block_dim × 0.8⌋ × nq × block_dim × d × 4` |
| `l_gm` | Log-sum-exp (split-KV) | `⌊block_dim × 0.8⌋ × nq × block_dim × 4` |
| `gm_k16` | K dequant buffer (INT8 quant only) | `2 × block_dim × 256 × nq × d × 2` |
| `gm_v16` | V dequant buffer (INT8 quant only) | `2 × block_dim × 256 × nq × d × 2` |

`k16`/`v16` are **only used for INT8 quant paths** (tiling keys 8, 9, 24, 25).
For fp16/bf16 (tiling keys 0, 1, 16, 17) they are passed but never accessed.

**Oversized buffers cause NaN**: reusing workspace buffers sized for a larger
shape (e.g. nq=64 from warmup) for a smaller shape (nq=32 in test) produces NaN
even after zeroing the entire buffer. The safe approach is exact-size allocation
per call, or the 4× same-shape warmup which ensures the wscache is populated
with correct-sized buffers from the start.

---

## 5. Tiling Path Selection (BN vs BNS)

The tiling has two main strategies:

| Path | Condition | `eff_bd` | `kvCN` | Tiling key (fp16) |
|------|-----------|----------|--------|-------------------|
| BN (split by Batch×Head) | `batch × nq ≥ block_dim × 0.8` | `min(block_dim, batch×⌈nq/headSplit⌉)` | 1 | 0 |
| BNS (split by Batch×Head×KVseq) | otherwise | `min(block_dim, ...)` | > 1 | 16 |

The BNS (split-KV) path introduces `o_core_tmp_gm` and `l_gm` partial-result
reduction via an all-AIV barrier on FFTS flag 3. This flag is also used by the
BN path as `QK_READY_DECODER`, so mixing paths poisons flag 3 for BNS.

**Practical rule**: test cases that would exercise the BNS path (small
`batch × nq`) must be isolated from BN-path test cases within the same process.

---

## 6. Compiling Standalone AscendC Kernels with bisheng

### Include path order matters
The standalone build needs kernel-local headers first, then the toolkit:
```bash
bisheng++ -x cce ... \
    -I ./op_kernel \
    -I /path/to/ascend-transformer-boost/src/kernels \
    -I ${ASCEND_TOOLKIT}/aarch64-linux/include \
    ...
```
If the toolkit headers come first, local overrides (like patched `.cce` files)
are ignored.

### Namespace conflicts in tiling headers
Tiling helper files from the ATB repo may reference names like `TILING_KEY_ID`
without namespace qualification. When included outside the ATB build system the
compiler cannot find them. Fix: qualify as `AtbOps::TILING_KEY_ID`.

### `__global__ __aicore__` and `<<<>>>` syntax
These are bisheng extensions. The kernel entry point must be declared as:
```cpp
extern "C" __global__ __aicore__ void my_kernel(...) { ... }
```
and launched as:
```cpp
my_kernel<<<block_dim, nullptr, stream>>>(...);
```
Standard `g++` cannot compile these files; always use `bisheng++`.

---

## 7. Calling Ascend Kernels from Python via ctypes

### Stream handling
```python
stream = torch.npu.current_stream()._as_parameter_
lib.call_kernel(block_dim, stream, ...)
```
Pass `ctypes.c_void_p` for the stream. Do NOT call `torch.npu.synchronize()`
between workspace setup and the kernel launch — it adds an extra NPU event
that shifts the FFTS counter.

### Pointer arguments
All GM buffer arguments must be `ctypes.c_void_p`:
```python
def as_ptr(t): return ctypes.c_void_p(t.data_ptr())
```
Passing a Python int directly instead of `c_void_p` will silently truncate on
some platforms.

### Synchronization after kernel
Always synchronize before reading the output tensor:
```python
lib.call_kernel(...)
torch.npu.synchronize()
result = output_tensor.cpu()
```

---

## 8. NPU "Cold Start" / First-Call NaN

The very first kernel invocation on a freshly initialized NPU device may
produce NaN or all-zeros output. This is independent of the FFTS issue. Always
run at least one warmup call (even with zero inputs) before any measurement or
correctness check.

---

## 9. Workspace Caching Strategy

When reusing workspace buffers across multiple calls:
- If the buffer is **too large** (from a prior larger shape): zero the entire
  buffer with `.zero_()` and return it. The kernel will only use the first N
  bytes it needs, but stale data beyond those bytes could be read on corner
  cases (e.g. split-KV path with different `kvCN`).
- If the buffer is **too small**: reallocate fresh zeros.
- **Safest**: always allocate exactly the right size for the current shape.
  Use a cache keyed by `(batch, nq, head_dim, block_dim)` to avoid redundant
  allocations across multiple calls with the same shape.

---

## 10. Reference API for Correctness Comparison

`torch_npu.atb._npu_paged_attention_v2` is the most reliable reference:
```python
torch_npu.atb._npu_paged_attention_v2(
    q, k_page, block_table, context_lens,
    value_cache=v_page, mask=None,
    num_kv_heads=nkv, num_heads=nq,
    scale_value=scale, mask_type=0, out=output,
)
```
Input tensor shapes:
- `q`: `[batch, nq, head_dim]`
- `k_page` / `v_page`: `[total_blocks, block_size, nkv, head_dim]`
- `block_table`: `[batch, max_blocks_per_seq]`, dtype `int32`
- `context_lens`: `[batch]`, dtype `int32`, on CPU

Tolerances for fp16: `rtol=5e-3, atol=2e-2`.

---

## 11. Performance Expectations

Calling a CCE kernel directly (standalone) versus through the ATB framework:
- **ATB overhead** (~0.04–0.08 ms): op graph dispatch, tiling recomputation,
  memory manager, profiling hooks.
- **Standalone**: pre-computed tiling, direct `rtKernelLaunch` call.
- Observed speedup: **1.3–2.2× faster** on 910B for GQA decode shapes
  (batch 1–16, nq 8–128, kv_seq 128–2048).
- Speedup is larger for small batches (where ATB overhead dominates) and for
  large batches (where ATB's memory manager adds contention).
