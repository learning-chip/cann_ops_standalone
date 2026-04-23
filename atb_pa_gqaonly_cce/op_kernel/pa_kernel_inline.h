/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: MulanPSL-2.0
 *
 * Inlined Ascend C220 helpers for paged attention standalone kernels.
 * Replaces legacy kernels/utils/kernel helpers for this directory only.
 */
#ifndef ATB_PA_PA_KERNEL_INLINE_H
#define ATB_PA_PA_KERNEL_INLINE_H

#include <limits>
#include <type_traits>

#include "kernel_operator.h"

#ifndef __force_inline__
#define __force_inline__ inline __attribute__((always_inline))
#endif

template <uint32_t ALIGN, typename T = uint32_t>
inline __aicore__ T RoundUp(const T val)
{
    static_assert(ALIGN != 0, "align must not be zero");
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    T align = ALIGN;
    if (val + align - 1 < val) {
        return val;
    }
    return (val + align - 1) / align * align;
}

template <typename T>
inline __aicore__ T RoundUp(const T val, const T align)
{
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    if (align == 0 || val + align - 1 < val) {
        return val;
    }
    return (val + align - 1) / align * align;
}

template <uint32_t DIVISOR, typename T = uint32_t>
inline __aicore__ T CeilDiv(const T dividend)
{
    static_assert(DIVISOR != 0, "divisor must not be zero");
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    T divisor = DIVISOR;
    if (dividend + divisor - 1 < dividend) {
        return dividend;
    }
    return (dividend + divisor - 1) / divisor;
}

template <typename T>
constexpr T T_MAX = std::numeric_limits<T>::max();

template <typename T>
inline __aicore__ T CeilDiv(const T dividend, const T divisor)
{
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    if (divisor == 0 || dividend + divisor - 1 < dividend) {
        return T_MAX<T>;
    }
    return (dividend + divisor - 1) / divisor;
}

constexpr Order_t ORDER_ONLY_VALUE = ONLY_VALUE;

template <typename DTypeIn, typename DTypeOut>
__aicore__ inline void conv_v(__ubuf__ DTypeOut *dst, __ubuf__ DTypeIn *src, uint8_t repeat, uint16_t dstBlockStride,
    uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride)
{
    if constexpr (std::is_same<DTypeIn, float>::value && std::is_same<DTypeOut, __bf16>::value) {
        vconv_f322bf16r((__ubuf__ __bf16 *)dst, (__ubuf__ float *)src, repeat, dstBlockStride, srcBlockStride,
            dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<DTypeIn, float>::value && std::is_same<DTypeOut, half>::value) {
        vconv_f322f16((__ubuf__ half *)dst, (__ubuf__ float *)src, repeat, dstBlockStride, srcBlockStride,
            dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<DTypeIn, half>::value && std::is_same<DTypeOut, float>::value) {
        vconv_f162f32((__ubuf__ float *)dst, (__ubuf__ half *)src, repeat, dstBlockStride, srcBlockStride,
            dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<DTypeIn, __bf16>::value && std::is_same<DTypeOut, float>::value) {
        vconv_bf162f32((__ubuf__ float *)dst, (__ubuf__ __bf16 *)src, repeat, dstBlockStride, srcBlockStride,
            dstRepeatStride, srcRepeatStride);
    } else {
        static_assert(!std::is_same<DTypeIn, DTypeIn>::value, "Unsupported conv_v dtype combination.");
    }
}

template <pipe_t pipe, uint8_t mode>
__aicore__ inline void FftsCrossCoreSync(uint16_t flagId)
{
    uint64_t config = 1ULL | (static_cast<uint64_t>(mode) << 4) | (static_cast<uint64_t>(flagId) << 8);
    ffts_cross_core_sync(pipe, config);
}

constexpr uint32_t PA_L1L0_BLOCK_BYTES = 32;
constexpr uint32_t PA_GM_ND2NZ_STRIDE_LIMIT = 65536;

template <typename DataType>
__aicore__ inline void pa_gm_to_l1_nd_nd(__cbuf__ DataType *l1, __gm__ DataType *gm, uint32_t nTileActual,
    uint32_t nTileCeil, uint32_t nVal, uint32_t dTileActual, uint32_t dTileCeil, uint32_t dVal)
{
    (void)nVal;
    (void)dTileCeil;
    static constexpr uint32_t BLOCK_SIZE = PA_L1L0_BLOCK_BYTES / sizeof(DataType);
    copy_gm_to_cbuf(l1, gm, 0, 1, CeilDiv<BLOCK_SIZE>(nTileActual * dTileActual), 0, 0, PAD_NONE);
}

template <typename DataType>
__aicore__ inline void pa_gm_to_l1_nd_nz(__cbuf__ DataType *l1, __gm__ DataType *gm, uint32_t nTileActual,
    uint32_t nTileCeil, uint32_t nVal, uint32_t dTileActual, uint32_t dTileCeil, uint32_t dVal)
{
    (void)nVal;
    (void)dTileCeil;
    static constexpr uint32_t BLOCK_SIZE = PA_L1L0_BLOCK_BYTES / sizeof(DataType);
    if (dVal < PA_GM_ND2NZ_STRIDE_LIMIT) {
        if constexpr (sizeof(DataType) == 4) {
            copy_gm_to_cbuf_multi_nd2nz_b32s(l1, gm, 0, 1, nTileActual, dTileActual, 0, dVal, nTileCeil, 1, 0);
        } else {
            copy_gm_to_cbuf_multi_nd2nz_b16(l1, gm, 0, 1, nTileActual, dTileActual, 0, dVal, nTileCeil, 1, 0);
        }
    } else {
        for (uint32_t i = 0; i < nTileActual; i++) {
            if constexpr (sizeof(DataType) == 4) {
                copy_gm_to_cbuf_multi_nd2nz_b32s(l1 + i * BLOCK_SIZE, gm + i * dVal, 0, 1, 1, dTileActual, 0, 0,
                    nTileCeil, 0, 0);
            } else {
                copy_gm_to_cbuf_multi_nd2nz_b16(l1 + i * BLOCK_SIZE, gm + i * dVal, 0, 1, 1, dTileActual, 0, 0,
                    nTileCeil, 0, 0);
            }
        }
    }
}

template <typename DataType, bool IsTranspose>
__aicore__ inline void pa_l1_to_l0_a_vector(__ca__ DataType *l0, __cbuf__ DataType *l1, uint32_t mTileCeil,
    uint32_t kPartCeil, uint32_t mSrcStride, uint32_t kSrcStride, uint32_t mDstStride, uint32_t kDstStride)
{
    (void)mTileCeil;
    (void)mSrcStride;
    (void)mDstStride;
    if constexpr (IsTranspose) {
        load_cbuf_to_ca(l0, l1, 0, kPartCeil, kSrcStride, kDstStride, 0, 1, (addr_cal_mode_t)0);
    } else {
        load_cbuf_to_ca(l0, l1, 0, kPartCeil, kSrcStride, kDstStride, 0, 0, (addr_cal_mode_t)0);
    }
}

template <typename DataType, bool IsTranspose>
__aicore__ inline void pa_l1_to_l0_b_vector(__cb__ DataType *l0, __cbuf__ DataType *l1, uint32_t nTileCeil,
    uint32_t kPartCeil, uint32_t nSrcStride, uint32_t kSrcStride, uint32_t nDstStride, uint32_t kDstStride)
{
    (void)nTileCeil;
    (void)nSrcStride;
    (void)nDstStride;
    if constexpr (IsTranspose) {
        load_cbuf_to_cb(l0, l1, 0, kPartCeil, kSrcStride, kDstStride, 0, 1, (addr_cal_mode_t)0);
    } else {
        load_cbuf_to_cb(l0, l1, 0, kPartCeil, kSrcStride, kDstStride, 0, 0, (addr_cal_mode_t)0);
    }
}

__aicore__ inline void pa_l0c_to_gm_nd_fp32(__gm__ float *gm, __cc__ float *cc, uint32_t mTileActual,
    uint32_t nTileActual, uint32_t srcStride, uint32_t dstStride, uint8_t unitFlag = 0)
{
    set_nd_para((uint64_t)1);
    pipe_barrier(PIPE_FIX);
    copy_matrix_cc_to_gm(gm, cc, 0, nTileActual, mTileActual, dstStride, srcStride, unitFlag, QuantMode_t::NoQuant, 0,
        false, true);
}

template <typename DType>
__aicore__ inline void pa_gm_to_ub(__ubuf__ DType *dst, __gm__ DType *src, uint8_t sid, uint16_t nBurst,
    uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride)
{
    copy_gm_to_ubuf(dst, src, sid, nBurst, lenBurst, srcStride, dstStride);
}

template <typename DType>
__aicore__ inline void pa_gm_to_ub_align(__ubuf__ DType *dst, __gm__ DType *src, uint8_t sid, uint16_t nBurst,
    uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum, uint32_t srcGap, uint32_t dstGap)
{
    if constexpr (sizeof(DType) == 1) {
        copy_gm_to_ubuf_align_b8(dst, src, sid, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
    } else if constexpr (sizeof(DType) == 2) {
        copy_gm_to_ubuf_align_b16(dst, src, sid, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
    } else {
        copy_gm_to_ubuf_align_b32(dst, src, sid, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
    }
}

template <typename DType>
__aicore__ inline void pa_ub_to_ub(__ubuf__ DType *dst, __ubuf__ DType *src, uint8_t sid, uint16_t nBurst,
    uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride)
{
    copy_ubuf_to_ubuf(dst, src, sid, nBurst, lenBurst, srcStride, dstStride);
}

template <typename DataType>
__aicore__ inline void pa_ub_to_gm(__gm__ DataType *dst, __ubuf__ DataType *src, uint8_t sid, uint16_t nBurst,
    uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride)
{
    copy_ubuf_to_gm(dst, src, sid, nBurst, lenBurst, srcStride, dstStride);
}

template <typename DType>
__aicore__ inline void pa_ub_to_gm_align(__gm__ DType *dst, __ubuf__ DType *src, uint8_t sid, uint16_t nBurst,
    uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum, uint32_t srcGap, uint32_t dstGap)
{
    if constexpr (sizeof(DType) == 1) {
        copy_ubuf_to_gm_align_b8(dst, src, sid, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
    } else if constexpr (sizeof(DType) == 2) {
        copy_ubuf_to_gm_align_b16(dst, src, sid, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
    } else {
        copy_ubuf_to_gm_align_b32(dst, src, sid, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
    }
}

template <typename ElementA, typename ElementB, typename AccDTypeC>
__aicore__ inline void pa_mmad(__cc__ AccDTypeC *l0c, __ca__ ElementA *l0a, __cb__ ElementB *l0b,
    uint32_t mTileActual, uint32_t nTileActual, uint32_t kPartActual, bool initC, uint8_t unitFlag = 0)
{
    mad(l0c, l0a, l0b, mTileActual, kPartActual, nTileActual, unitFlag, false, false, initC);
}

template <typename T>
__aicore__ inline __ubuf__ T *Ub(__ubuf__ T *p)
{
    return p;
}

#endif /* ATB_PA_PA_KERNEL_INLINE_H */
