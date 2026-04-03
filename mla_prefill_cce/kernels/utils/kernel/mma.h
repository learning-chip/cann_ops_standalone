/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INCLUDE_MMA_H
#define INCLUDE_MMA_H

#include "hardware.h"
#include "kernel_operator.h"

template <ArchType ArchTag, typename ElementA, typename ElementB, typename AccDTypeC, bool IsTransposeA = false>
__aicore__ inline void mmad_raw(
    __cc__ AccDTypeC *l0cTensor,
    __ca__ ElementA *l0aTensor,
    __cb__ ElementB *l0bTensor,
    uint32_t mTileActual,
    uint32_t nTileActual,
    uint32_t kPartActual,
    bool initC,
    uint8_t unitFlag = 0)
{
    (void)ArchTag;
    (void)IsTransposeA;
    mad(
        l0cTensor,
        l0aTensor,
        l0bTensor,
        mTileActual,
        kPartActual,
        nTileActual,
        unitFlag,
        false,
        0,
        initC);
}

#endif
