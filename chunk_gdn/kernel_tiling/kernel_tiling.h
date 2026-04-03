// Shim header for standalone build.
//
// These kernel sources include `kernel_tiling/kernel_tiling.h`,
// but this header is not present in CANN 8.5.0 at the expected path.
//
// We re-export CANN's `adv_api/kernel_tiling.h` and provide the minimal
// `GET_TILING_DATA` macro required by this kernel.

#pragma once

#include "adv_api/kernel_tiling.h"

// Minimal tiling fetch: copy fields from GM into a local tiling struct.
//
// We cannot use memcpy/struct reinterpret_cast between address spaces in AscendC here,
// so we load each member explicitly.
//
// Note: these offsets must match `ChunkGatedDeltaRuleTilingData` layout in this repo:
// - int64 fields: offsets 0..96 step 8
// - float scale: offset 104
// - TCubeTiling starts at offset 108 (verified via ctypes in test_chunk_gdn.py)
#define GET_TILING_DATA(tilingData, tilingGM)                                                                                          \
    ChunkGatedDeltaRuleTilingData tilingData;                                                                                          \
    tilingData.aiCoreNum = *reinterpret_cast<const __gm__ int64_t *>(tilingGM + 0);                                                   \
    tilingData.t = *reinterpret_cast<const __gm__ int64_t *>(tilingGM + 8);                                                         \
    tilingData.nk = *reinterpret_cast<const __gm__ int64_t *>(tilingGM + 16);                                                        \
    tilingData.dk = *reinterpret_cast<const __gm__ int64_t *>(tilingGM + 24);                                                        \
    tilingData.nv = *reinterpret_cast<const __gm__ int64_t *>(tilingGM + 32);                                                        \
    tilingData.dv = *reinterpret_cast<const __gm__ int64_t *>(tilingGM + 40);                                                        \
    tilingData.b = *reinterpret_cast<const __gm__ int64_t *>(tilingGM + 48);                                                         \
    tilingData.hasGamma = *reinterpret_cast<const __gm__ int64_t *>(tilingGM + 56);                                                  \
    tilingData.chunkSize = *reinterpret_cast<const __gm__ int64_t *>(tilingGM + 64);                                                \
    tilingData.maxGroupLength = *reinterpret_cast<const __gm__ int64_t *>(tilingGM + 72);                                       \
    tilingData.interWorkspaceSz = *reinterpret_cast<const __gm__ int64_t *>(tilingGM + 80);                                       \
    tilingData.stageWorkspaceSz = *reinterpret_cast<const __gm__ int64_t *>(tilingGM + 88);                                       \
    tilingData.stageOneParaNum = *reinterpret_cast<const __gm__ int64_t *>(tilingGM + 96);                                       \
    tilingData.scale = *reinterpret_cast<const __gm__ float *>(tilingGM + 104);                                                  \
    {                                                                                                                                  \
        auto mmPtr = reinterpret_cast<const __gm__ int32_t *>(tilingGM + 108);                                                      \
        tilingData.matmulTilingFp32.usedCoreNum = mmPtr[0];                                                                       \
        tilingData.matmulTilingFp32.M = mmPtr[1];                                                                                 \
        tilingData.matmulTilingFp32.N = mmPtr[2];                                                                                 \
        tilingData.matmulTilingFp32.Ka = mmPtr[3];                                                                                \
        tilingData.matmulTilingFp32.Kb = mmPtr[4];                                                                                \
        tilingData.matmulTilingFp32.singleCoreM = mmPtr[5];                                                                   \
        tilingData.matmulTilingFp32.singleCoreN = mmPtr[6];                                                                   \
        tilingData.matmulTilingFp32.singleCoreK = mmPtr[7];                                                                   \
        tilingData.matmulTilingFp32.baseM = mmPtr[8];                                                                             \
        tilingData.matmulTilingFp32.baseN = mmPtr[9];                                                                             \
        tilingData.matmulTilingFp32.baseK = mmPtr[10];                                                                            \
        tilingData.matmulTilingFp32.depthA1 = mmPtr[11];                                                                         \
        tilingData.matmulTilingFp32.depthB1 = mmPtr[12];                                                                         \
        tilingData.matmulTilingFp32.stepM = mmPtr[13];                                                                           \
        tilingData.matmulTilingFp32.stepN = mmPtr[14];                                                                           \
        tilingData.matmulTilingFp32.isBias = mmPtr[15];                                                                           \
        tilingData.matmulTilingFp32.transLength = mmPtr[16];                                                                      \
        tilingData.matmulTilingFp32.iterateOrder = mmPtr[17];                                                                    \
        tilingData.matmulTilingFp32.shareMode = mmPtr[18];                                                                       \
        tilingData.matmulTilingFp32.shareL1Size = mmPtr[19];                                                                     \
        tilingData.matmulTilingFp32.shareL0CSize = mmPtr[20];                                                                     \
        tilingData.matmulTilingFp32.shareUbSize = mmPtr[21];                                                                     \
        tilingData.matmulTilingFp32.batchM = mmPtr[22];                                                                          \
        tilingData.matmulTilingFp32.batchN = mmPtr[23];                                                                          \
        tilingData.matmulTilingFp32.singleBatchM = mmPtr[24];                                                                  \
        tilingData.matmulTilingFp32.singleBatchN = mmPtr[25];                                                                  \
        tilingData.matmulTilingFp32.stepKa = mmPtr[26];                                                                        \
        tilingData.matmulTilingFp32.stepKb = mmPtr[27];                                                                        \
        tilingData.matmulTilingFp32.depthAL1CacheUB = mmPtr[28];                                                              \
        tilingData.matmulTilingFp32.depthBL1CacheUB = mmPtr[29];                                                              \
        tilingData.matmulTilingFp32.dbL0A = mmPtr[30];                                                                            \
        tilingData.matmulTilingFp32.dbL0B = mmPtr[31];                                                                            \
        tilingData.matmulTilingFp32.dbL0C = mmPtr[32];                                                                            \
        tilingData.matmulTilingFp32.ALayoutInfoB = mmPtr[33];                                                                  \
        tilingData.matmulTilingFp32.ALayoutInfoS = mmPtr[34];                                                                  \
        tilingData.matmulTilingFp32.ALayoutInfoN = mmPtr[35];                                                                  \
        tilingData.matmulTilingFp32.ALayoutInfoG = mmPtr[36];                                                                  \
        tilingData.matmulTilingFp32.ALayoutInfoD = mmPtr[37];                                                                  \
        tilingData.matmulTilingFp32.BLayoutInfoB = mmPtr[38];                                                                  \
        tilingData.matmulTilingFp32.BLayoutInfoS = mmPtr[39];                                                                  \
        tilingData.matmulTilingFp32.BLayoutInfoN = mmPtr[40];                                                                  \
        tilingData.matmulTilingFp32.BLayoutInfoG = mmPtr[41];                                                                  \
        tilingData.matmulTilingFp32.BLayoutInfoD = mmPtr[42];                                                                  \
        tilingData.matmulTilingFp32.CLayoutInfoB = mmPtr[43];                                                                  \
        tilingData.matmulTilingFp32.CLayoutInfoS1 = mmPtr[44];                                                                 \
        tilingData.matmulTilingFp32.CLayoutInfoN = mmPtr[45];                                                                 \
        tilingData.matmulTilingFp32.CLayoutInfoG = mmPtr[46];                                                                 \
        tilingData.matmulTilingFp32.CLayoutInfoS2 = mmPtr[47];                                                                 \
        tilingData.matmulTilingFp32.BatchNum = mmPtr[48];                                                                     \
        tilingData.matmulTilingFp32.mxTypePara = mmPtr[49];                                                                     \
    }


