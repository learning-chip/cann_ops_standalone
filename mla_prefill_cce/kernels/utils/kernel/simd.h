/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INCLUDE_SIMD_H
#define INCLUDE_SIMD_H

#include "hardware.h"
#include "kernel_operator.h"
constexpr uint32_t DUP_REPEAT_SIZE = 256;
/////////////////////////////////////////////////////
// vcgadd
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void cgadd_v(__ubuf__ DType* dst,
                               __ubuf__ DType* src,
                               const int32_t repeat,
                               const int32_t dstRepStride,
                               const int32_t srcBlkStride,
                               const int32_t srcRepStride)
{
    vcgadd(dst, src, repeat, dstRepStride, srcBlkStride, srcRepStride);
}

/////////////////////////////////////////////////////
// vadd
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void add_v(__ubuf__ DType* dst,
                             __ubuf__ DType* src0,
                             __ubuf__ DType* src1,
                             uint8_t repeat,
                             uint8_t dstBlockStride,
                             uint8_t src0BlockStride,
                             uint8_t src1BlockStride,
                             uint8_t dstRepeatStride,
                             uint8_t src0RepeatStride,
                             uint8_t src1RepeatStride)
{
    vadd(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
         src0RepeatStride, src1RepeatStride);
}

/////////////////////////////////////////////////////
// vadds
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void adds_v(__ubuf__ DType* dst,
                              __ubuf__ DType* src,
                              DType scalarValue,
                              uint8_t repeat,
                              uint8_t dstBlockStride,
                              uint8_t srcBlockStride,
                              uint8_t dstRepeatStride,
                              uint8_t srcRepeatStride)
{
    vadds(dst, src, scalarValue, repeat, static_cast<uint16_t>(dstBlockStride),
          static_cast<uint16_t>(srcBlockStride), dstRepeatStride, srcRepeatStride);
}

/////////////////////////////////////////////////////
// vcadd
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void cadd_v(__ubuf__ DType* dst,
                              __ubuf__ DType* src,
                              uint8_t repeat,
                              uint16_t dstRepeatStride,
                              uint16_t srcBlockStride,
                              uint16_t srcRepeatStride)
{
    vcadd(dst, src, repeat, dstRepeatStride, srcBlockStride, srcRepeatStride, 0);
}
/////////////////////////////////////////////////////
// vbrcb
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void brcb_v(__ubuf__ DType* dst,
                              __ubuf__ DType* src,
                              uint16_t dstBlockStride,
                              uint16_t dstRepeatStride,
                              uint8_t repeat)
{
    if constexpr (sizeof(DType) == 2) {
        vbrcb((__ubuf__ uint16_t*)dst, (__ubuf__ uint16_t*)src, dstBlockStride, dstRepeatStride, repeat);
    } else {
        vbrcb((__ubuf__ uint32_t*)dst, (__ubuf__ uint32_t*)src, dstBlockStride, dstRepeatStride, repeat);
    }
}

/////////////////////////////////////////////////////
// vcmax
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType, Order_t OrderType>
__aicore__ inline void cmax_v(__ubuf__ DType* dst,
                              __ubuf__ DType* src,
                              uint8_t repeat,
                              uint16_t dstRepeatStride,
                              uint16_t srcBlockStride,
                              uint16_t srcRepeatStride)
{
#if defined(__DAV_C220_VEC__)
    vcmax(dst,
          src,
          repeat,
          dstRepeatStride,
          srcBlockStride,
          srcRepeatStride,
          static_cast<Order_t>(OrderType));
#else
    vcmax(dst, src, repeat, dstRepeatStride, srcBlockStride, srcRepeatStride);
#endif
}

/////////////////////////////////////////////////////
// vconv
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DTypeIn, typename DTypeOut>
__aicore__ inline void conv_v(__ubuf__ DTypeOut* dst,
                              __ubuf__ DTypeIn* src,
                              uint8_t repeat,
                              uint16_t dstBlockStride,
                              uint16_t srcBlockStride,
                              uint16_t dstRepeatStride,
                              uint16_t srcRepeatStride)
{
    if constexpr (std::is_same<DTypeIn, float>::value && std::is_same<DTypeOut, __bf16>::value) {
        vconv_f322bf16r((__ubuf__ __bf16*)dst,
                        (__ubuf__ float*)src,
                        repeat,
                        dstBlockStride,
                        srcBlockStride,
                        dstRepeatStride,
                        srcRepeatStride);
    } else if constexpr (std::is_same<DTypeIn, float>::value && std::is_same<DTypeOut, half>::value) {
        vconv_f322f16((__ubuf__ half*)dst,
                      (__ubuf__ float*)src,
                      repeat,
                      dstBlockStride,
                      srcBlockStride,
                      dstRepeatStride,
                      srcRepeatStride);
    } else if constexpr (std::is_same<DTypeIn, half>::value && std::is_same<DTypeOut, float>::value) {
        vconv_f162f32((__ubuf__ float*)dst,
                      (__ubuf__ half*)src,
                      repeat,
                      dstBlockStride,
                      srcBlockStride,
                      dstRepeatStride,
                      srcRepeatStride);
    } else if constexpr (std::is_same<DTypeIn, __bf16>::value && std::is_same<DTypeOut, float>::value) {
        vconv_bf162f32((__ubuf__ float*)dst,
                       (__ubuf__ __bf16*)src,
                       repeat,
                       dstBlockStride,
                       srcBlockStride,
                       dstRepeatStride,
                       srcRepeatStride);
    } else {
        static_assert(!std::is_same<DTypeIn, DTypeIn>::value, "Unsupported conv_v dtype combination.");
    }
}

/////////////////////////////////////////////////////
// vconv_f322bf16r
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DTypeIn, typename DTypeOut>
__aicore__ inline void convr_v(__ubuf__ DTypeOut* dst,
                               __ubuf__ DTypeIn* src,
                               uint8_t repeat,
                               uint16_t dstBlockStride,
                               uint16_t srcBlockStride,
                               uint16_t dstRepeatStride,
                               uint16_t srcRepeatStride)
{
    if constexpr (std::is_same<DTypeIn, float>::value && std::is_same<DTypeOut, __bf16>::value) {
        vconv_f322bf16r((__ubuf__ __bf16*)dst,
                        (__ubuf__ float*)src,
                        repeat,
                        dstBlockStride,
                        srcBlockStride,
                        dstRepeatStride,
                        srcRepeatStride);
    } else if constexpr (std::is_same<DTypeIn, float>::value && std::is_same<DTypeOut, half>::value) {
        vconv_f322f16r((__ubuf__ half*)dst,
                       (__ubuf__ float*)src,
                       repeat,
                       dstBlockStride,
                       srcBlockStride,
                       dstRepeatStride,
                       srcRepeatStride);
    } else {
        static_assert(!std::is_same<DTypeIn, DTypeIn>::value, "Unsupported convr_v dtype combination.");
    }
}

/////////////////////////////////////////////////////
// vdiv
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void div_v(__ubuf__ DType* dst,
                             __ubuf__ DType* src0,
                             __ubuf__ DType* src1,
                             uint8_t repeat,
                             uint8_t dstBlockStride,
                             uint8_t src0BlockStride,
                             uint8_t src1BlockStride,
                             uint8_t dstRepeatStride,
                             uint8_t src0RepeatStride,
                             uint8_t src1RepeatStride)
{
    vdiv(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
         src0RepeatStride, src1RepeatStride);
}

/////////////////////////////////////////////////////
// vexp
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void exp_v(__ubuf__ DType* dst,
                             __ubuf__ DType* src,
                             uint8_t repeat,
                             uint16_t dstBlockStride,
                             uint16_t srcBlockStride,
                             uint16_t dstRepeatStride,
                             uint16_t srcRepeatStride)
{
    vexp(dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}

/////////////////////////////////////////////////////
// vmax
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void max_v(__ubuf__ DType* dst,
                             __ubuf__ DType* src0,
                             __ubuf__ DType* src1,
                             uint8_t repeat,
                             uint8_t dstBlockStride,
                             uint8_t src0BlockStride,
                             uint8_t src1BlockStride,
                             uint8_t dstRepeatStride,
                             uint8_t src0RepeatStride,
                             uint8_t src1RepeatStride)
{
    vmax(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
         src0RepeatStride, src1RepeatStride);
}

/////////////////////////////////////////////////////
// vmul
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void mul_v(__ubuf__ DType* dst,
                             __ubuf__ DType* src0,
                             __ubuf__ DType* src1,
                             uint8_t repeat,
                             uint8_t dstBlockStride,
                             uint8_t src0BlockStride,
                             uint8_t src1BlockStride,
                             uint8_t dstRepeatStride,
                             uint8_t src0RepeatStride,
                             uint8_t src1RepeatStride)
{
    vmul(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
         src0RepeatStride, src1RepeatStride);
}

/////////////////////////////////////////////////////
// vmuls
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void muls_v(__ubuf__ DType* dst,
                              __ubuf__ DType* src0,
                              DType src1,
                              uint8_t repeat,
                              uint16_t dstBlockStride,
                              uint16_t srcBlockStride,
                              uint16_t dstRepeatStride,
                              uint16_t srcRepeatStride)
{
    vmuls(dst, src0, src1, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}

/////////////////////////////////////////////////////
// vsub
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void sub_v(__ubuf__ DType* dst,
                             __ubuf__ DType* src0,
                             __ubuf__ DType* src1,
                             uint8_t repeat,
                             uint8_t dstBlockStride,
                             uint8_t src0BlockStride,
                             uint8_t src1BlockStride,
                             uint8_t dstRepeatStride,
                             uint8_t src0RepeatStride,
                             uint8_t src1RepeatStride)
{
    vsub(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
         src0RepeatStride, src1RepeatStride);
}

/////////////////////////////////////////////////////
// vmaxs
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void maxs_v(__ubuf__ DType* dst,
                              __ubuf__ DType* src0,
                              DType src1,
                              uint8_t repeat,
                              uint16_t dstBlockStride,
                              uint16_t srcBlockStride,
                              uint16_t dstRepeatStride,
                              uint16_t srcRepeatStride)
{
    vmaxs(dst, src0, src1, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}

/////////////////////////////////////////////////////
// vmins
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void mins_v(__ubuf__ DType* dst,
                              __ubuf__ DType* src0,
                              DType src1,
                              uint8_t repeat,
                              uint16_t dstBlockStride,
                              uint16_t srcBlockStride,
                              uint16_t dstRepeatStride,
                              uint16_t srcRepeatStride)
{
    vmins(dst, src0, src1, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}

/////////////////////////////////////////////////////
// vsqrt
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void sqrt_v(__ubuf__ DType* dst,
                              __ubuf__ DType* src,
                              uint8_t repeat,
                              uint16_t dstBlockStride,
                              uint16_t srcBlockStride,
                              uint16_t dstRepeatStride,
                              uint16_t srcRepeatStride)
{
    vsqrt(dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}

/////////////////////////////////////////////////////
// vln
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void ln_v(__ubuf__ DType* dst,
                            __ubuf__ DType* src,
                            uint8_t repeat,
                            uint16_t dstBlockStride,
                            uint16_t srcBlockStride,
                            uint16_t dstRepeatStride,
                            uint16_t srcRepeatStride)
{
    vln(dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}

/////////////////////////////////////////////////////
// vtranspose
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void tranpose_v(__ubuf__ DType* dst, __ubuf__ DType* src)
{
    vtranspose(dst, src);
}

/////////////////////////////////////////////////////
// vcgmax
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void cgmax_v(__ubuf__ DType* dst,
                               __ubuf__ DType* src,
                               const int32_t repeat,
                               const int32_t dstRepStride,
                               const int32_t srcBlkStride,
                               const int32_t srcRepStride)
{
    vcgmax(dst, src, repeat, dstRepStride, srcBlkStride, srcRepStride);
}

template <ArchType ArchTag, typename DType>
__aicore__ inline void dup_v(__ubuf__ DType* dst,
                             DType src1,
                             const int32_t repeat)
{
    vector_dup(dst,
               src1,
               static_cast<uint8_t>(repeat * DUP_REPEAT_SIZE / sizeof(DType)),
               static_cast<uint16_t>(1),
               static_cast<uint16_t>(1),
               static_cast<uint8_t>(1),
               static_cast<uint8_t>(0));
}

#endif