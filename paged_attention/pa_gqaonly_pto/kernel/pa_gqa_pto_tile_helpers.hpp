/**
 * PTO-ISA bridge for the paged-attention cube path.
 *
 * The production kernel still uses AscendC intrinsics (copy_gm_to_cbuf, mad, …)
 * for full tiling coverage; this header pulls in PTO-ISA so new code and
 * incremental replacements can use pto::TLOAD / pto::TMATMUL / pto::TSTORE
 * in the same translation unit. See PORT_PROGRESS.md for the migration map.
 */
#pragma once

#if defined(__DAV_C220_CUBE__) && \
    ((defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || defined(__CHECK_FEATURE_AT_PRECOMPILE__))
#include <pto/pto-inst.hpp>

namespace pa_pto {

// Upper bounds for one QK / PV matmul tile (must cover runtime m,k,n from tiling).
constexpr int kPaPtoMadMaxM = 256;
constexpr int kPaPtoMadMaxK = 1024;
constexpr int kPaPtoMadMaxN = 4096;

// `CheckAcc2gm` requires static Acc Cols in [1, 4095] and fractal alignment; 4080 is the largest 16-aligned fit.
constexpr int kPaPtoL0cStoreMaxN = 4080;

/**
 * FP16/BF16 matmul with FP32 accumulator using PTO-ISA (`pto::TMatmul` → hardware `mad`).
 * When m==1, uses GEMV-style `isGemv=true` so behavior matches a raw `mad(..., m=1, ...)`
 * (the generic PTO matmul path would otherwise widen m to 16 on A2A3).
 */
template <typename InDtype>
__aicore__ inline void tmatmul_fp32acc(__cc__ float *c_ptr, __ca__ InDtype *a_ptr, __cb__ InDtype *b_ptr, uint16_t m,
    uint16_t k, uint16_t n)
{
    using namespace pto;
    using TL = TileLeft<InDtype, kPaPtoMadMaxM, kPaPtoMadMaxK, DYNAMIC, DYNAMIC>;
    using TR = TileRight<InDtype, kPaPtoMadMaxK, kPaPtoMadMaxN, DYNAMIC, DYNAMIC>;
    using TC = TileAcc<float, kPaPtoMadMaxM, kPaPtoMadMaxN, DYNAMIC, DYNAMIC>;

    TL a_tile(static_cast<size_t>(m), static_cast<size_t>(k));
    TR b_tile(static_cast<size_t>(k), static_cast<size_t>(n));
    TC c_tile(static_cast<size_t>(m), static_cast<size_t>(n));

    TASSIGN(a_tile, reinterpret_cast<uintptr_t>(a_ptr));
    TASSIGN(b_tile, reinterpret_cast<uintptr_t>(b_ptr));
    TASSIGN(c_tile, reinterpret_cast<uintptr_t>(c_ptr));

    const bool kDir = GetKDirectionAlign(a_tile, b_tile);
    if (m == 1) {
        TMatmul<AccPhase::Unspecified, TC, TL, TR, false, true, true>(
            c_tile.data(), a_tile.data(), b_tile.data(), 1, k, n, kDir);
    } else {
        TMatmul<AccPhase::Unspecified, TC, TL, TR, false, true, false>(
            c_tile.data(), a_tile.data(), b_tile.data(), m, k, n, kDir);
    }
}

/**
 * FP32 L0C → GM ND store via PTO-ISA (`pto::TSTORE` → `TStoreAcc` → `copy_matrix_cc_to_gm`).
 * `dstLeadingDim` is the row stride in **elements** (same as legacy `dstStride` to `copy_matrix_cc_to_gm`).
 * Uses `TileAccCompact` so fractal source stride matches `RoundUp<16>(m)` like the AscendC path.
 * When `nActual` exceeds `kPaPtoL0cStoreMaxN` (PTO `TSTORE` Acc tile limits), falls back to `copy_matrix_cc_to_gm`.
 */
__aicore__ inline void tstore_l0c_fp32_nd(__gm__ float *gm, __cc__ float *cc, uint32_t mActual, uint32_t nActual,
    uint32_t dstLeadingDim)
{
    const uint32_t srcStrideFractal = static_cast<uint32_t>((mActual + 15U) / 16U * 16U);
    if (mActual > static_cast<uint32_t>(kPaPtoMadMaxM) || nActual > static_cast<uint32_t>(kPaPtoL0cStoreMaxN)) {
        set_nd_para((uint64_t)1);
        pipe_barrier(PIPE_FIX);
        copy_matrix_cc_to_gm(gm, cc, 0, nActual, mActual, dstLeadingDim, srcStrideFractal, 0, QuantMode_t::NoQuant, 0,
            false, true);
        return;
    }

    using namespace pto;
    using GmShape = TileShape2D<float, DYNAMIC, DYNAMIC, Layout::ND>;
    using GmStride = Stride<1, 1, 1, DYNAMIC, 1>;
    using GlobalOut = GlobalTensor<float, GmShape, GmStride, Layout::ND>;
    using TileC = TileAccCompact<float, kPaPtoMadMaxM, kPaPtoL0cStoreMaxN, DYNAMIC, DYNAMIC>;

    TileC c_tile(static_cast<size_t>(mActual), static_cast<size_t>(nActual));
    TASSIGN(c_tile, reinterpret_cast<uintptr_t>(cc));

    const GmShape shape(static_cast<int64_t>(mActual), static_cast<int64_t>(nActual));
    const GmStride stride(static_cast<int64_t>(dstLeadingDim));
    GlobalOut g_out(gm, shape, stride);
    TASSIGN(g_out, gm);

    pipe_barrier(PIPE_FIX);
    TSTORE(g_out, c_tile);
}

} // namespace pa_pto

#elif defined(__DAV_C220_VEC__) && \
    ((defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || defined(__CHECK_FEATURE_AT_PRECOMPILE__))
// Vector core: pull PTO-ISA headers for incremental migration (TADD/TEXP/…) alongside AscendC.
#include <pto/pto-inst.hpp>
#endif
