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

template <pipe_t pipe, uint8_t mode>
__aicore__ inline void FftsCrossCoreSync(uint16_t flagId)
{
    uint64_t config = 1ULL | (static_cast<uint64_t>(mode) << 4) | (static_cast<uint64_t>(flagId) << 8);
    ffts_cross_core_sync(pipe, config);
}

#endif
