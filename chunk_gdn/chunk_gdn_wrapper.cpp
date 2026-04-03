// Standalone wrapper to launch `chunk_gated_delta_rule` via ctypes.
//
// This file intentionally includes only the required kernel source
// (vendored under op_kernel/, plus local shims).

#include <stdint.h>

// AscendC types/keywords (bfloat16_t, __gm__, etc) are pulled in by the kernel include below.
#include "chunk_gdn_kernel_entry.cpp"
#include "runtime/rt.h"

extern "C" void call_kernel(
    uint32_t blockDim,
    void *stream,
    uint8_t *query,      // GM bfloat16
    uint8_t *key,        // GM bfloat16
    uint8_t *value,      // GM bfloat16
    uint8_t *beta,       // GM bfloat16
    uint8_t *initialState,  // GM bfloat16, shape (B, Nv, Dv, Dk)
    uint8_t *seqlens,    // GM int32, shape (B,)
    uint8_t *gOptional,  // GM float, shape (T, Nv); pass nullptr for hasGamma=0
    uint8_t *out,        // GM bfloat16, shape (T, Nv, Dv)
    uint8_t *finalState, // GM bfloat16, shape (B, Nv, Dv, Dk)
    uint8_t *workspaceGM,  // GM uint8 workspace (must include 16MB system workspace)
    uint8_t *tilingGM      // GM uint8 tiling struct bytes
) {
    // Same FFTS base as `call_stage1` / `call_stage2` / `call_stage3` (CrossCoreWait in full CGDR).
    void *ffts_addr = nullptr;
    uint32_t ffts_len = 0;
    rtGetC2cCtrlAddr(reinterpret_cast<uint64_t *>(&ffts_addr), &ffts_len);

    chunk_gated_delta_rule<<<blockDim, nullptr, stream>>>(
        (__gm__ uint8_t *)ffts_addr,
        (__gm__ uint8_t *)query,
        (__gm__ uint8_t *)key,
        (__gm__ uint8_t *)value,
        (__gm__ uint8_t *)beta,
        (__gm__ uint8_t *)initialState,
        (__gm__ uint8_t *)seqlens,
        (__gm__ uint8_t *)gOptional,
        (__gm__ uint8_t *)out,
        (__gm__ uint8_t *)finalState,
        (__gm__ uint8_t *)workspaceGM,
        (__gm__ uint8_t *)tilingGM
    );
}

