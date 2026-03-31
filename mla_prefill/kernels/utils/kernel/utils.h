/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INCLUDE_UTILS_H
#define INCLUDE_UTILS_H

__aicore__ inline void SetFftsBaseAddr(uint64_t config)
{
    set_ffts_base_addr(config);
}

template <typename IN_DTYPE>
__aicore__ inline void SetPadding(IN_DTYPE padValue)
{
    if constexpr (sizeof(IN_DTYPE) == 8) {
        set_padding(static_cast<uint64_t>(padValue));
    } else if constexpr (sizeof(IN_DTYPE) == 4) {
        set_padding(static_cast<uint32_t>(padValue));
    } else if constexpr (sizeof(IN_DTYPE) == 2) {
        set_padding(static_cast<uint16_t>(padValue));
    } else {
        set_padding(static_cast<uint8_t>(padValue));
    }
}

__aicore__ inline void SetAtomicnone()
{
    set_atomic_none();
}

__aicore__ inline void SetMasknorm()
{
#if __CCE_AICORE__ == 100
    return;
#endif
    set_mask_norm();
}

__aicore__ inline void SetNdpara(uint16_t ndNum, uint16_t srcNdStride, uint16_t dstNdStride)
{
    uint64_t config = static_cast<uint64_t>(ndNum);
    config |= static_cast<uint64_t>(srcNdStride) << 16;
    config |= static_cast<uint64_t>(dstNdStride) << 32;
    set_nd_para(config);
}

template <typename IN_DTYPE>
__aicore__ inline void SetVectorMask(const uint64_t maskHigh, const uint64_t maskLow)
{
    (void)sizeof(IN_DTYPE);
    set_vector_mask(maskHigh, maskLow);
}

__aicore__ inline int64_t GetSubBlockidx()
{
    return get_subblockid();
}

__aicore__ inline void WaitFlagDev(uint16_t flagId)
{
    wait_flag_dev(flagId);
}

template <pipe_t pipe, uint8_t mode>
__aicore__ inline void FftsCrossCoreSync(uint16_t flagId)
{
    uint64_t config = 1ULL | (static_cast<uint64_t>(mode) << 4) | (static_cast<uint64_t>(flagId) << 8);
    ffts_cross_core_sync(pipe, config);
}

#endif