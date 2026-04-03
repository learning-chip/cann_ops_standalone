// Standalone Stage1 wrapper: real `Stage1` (see op_kernel/chunk_gated_delta_rule_stage1.h).
#include "chunk_gated_delta_rule.h"
#include "runtime/rt.h"

using namespace AscendC;
using namespace ChunkGatedDeltaRule;
using namespace matmul;

extern "C" __global__ __aicore__ void stage1_kernel(
    GM_ADDR fftsAddr, GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta, GM_ADDR gOptional,
    GM_ADDR stageOneMask, GM_ADDR qktOut, GM_ADDR gCumExpOut, GM_ADDR kCumDecayOut,
    GM_ADDR vInnerOut, GM_ADDR qPrimeOut, GM_ADDR kgOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    SetSyncBaseAddr((unsigned long)fftsAddr);
    SetAtomicNone();
    SetMaskNorm();

    REGISTER_TILING_DEFAULT(ChunkGatedDeltaRuleTilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    TPipe pipe;
    StageOneMT mm;
    if ASCEND_IS_AIC {
        mm.Init(&tilingData.matmulTilingFp32, &pipe);
    }

    GlobalTensor<bfloat16_t> queryGm;
    GlobalTensor<bfloat16_t> keyGm;
    GlobalTensor<bfloat16_t> valueGm;
    GlobalTensor<bfloat16_t> betaGm;
    GlobalTensor<float> gGm;
    GlobalTensor<float> maskGm;
    GlobalTensor<float> qktGm;
    GlobalTensor<float> gCumExpGm;
    GlobalTensor<float> kCumDecayGm;
    GlobalTensor<float> vInnerGm;
    GlobalTensor<float> qPrimeGm;
    GlobalTensor<float> kgGm;

    queryGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(query), tilingData.t * tilingData.nk * tilingData.dk);
    keyGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(key), tilingData.t * tilingData.nk * tilingData.dk);
    valueGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(value), tilingData.t * tilingData.nv * tilingData.dv);
    betaGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(beta), tilingData.t * tilingData.nv);
    if (gOptional != nullptr) {
        gGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gOptional), tilingData.t * tilingData.nv);
    }
    maskGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(stageOneMask),
                           tilingData.chunkSize * tilingData.chunkSize * tilingData.aiCoreNum * TASK_RATIO);
    qktGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(qktOut), tilingData.nv * tilingData.maxGroupLength * tilingData.chunkSize);
    gCumExpGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gCumExpOut), tilingData.nv * tilingData.maxGroupLength);
    kCumDecayGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(kCumDecayOut), tilingData.nv * tilingData.maxGroupLength * tilingData.dk);
    vInnerGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(vInnerOut), tilingData.nv * tilingData.maxGroupLength * tilingData.dv);
    qPrimeGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(qPrimeOut), tilingData.nv * tilingData.maxGroupLength * tilingData.dk);
    kgGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(kgOut), tilingData.nv * tilingData.maxGroupLength * tilingData.dk);

    ChunkGroup cg;
    cg.startPos = 0;
    cg.length = tilingData.t;
    cg.chunkSize = tilingData.chunkSize;
    cg.coreStart = 0;
    cg.coreEnd = 0;

    GDRStageOneInitParams params{queryGm, keyGm, valueGm, betaGm, gGm,
                                 gCumExpGm, kCumDecayGm, vInnerGm, qPrimeGm, kgGm, qktGm,
                                 workspaceGM, maskGm, cg, (gOptional != nullptr)};

    Stage1 stage1(mm);
    stage1.Init(params, &pipe, &tilingData);
    stage1.Process();
}

extern "C" void call_stage1(
    uint32_t blockDim, void *stream,
    uint8_t *query, uint8_t *key, uint8_t *value, uint8_t *beta, uint8_t *gOptional,
    uint8_t *stageOneMask, uint8_t *qktOut, uint8_t *gCumExpOut, uint8_t *kCumDecayOut,
    uint8_t *vInnerOut, uint8_t *qPrimeOut, uint8_t *kgOut, uint8_t *workspaceGM, uint8_t *tilingGM)
{
    void *ffts_addr = nullptr;
    uint32_t ffts_len = 0;
    rtGetC2cCtrlAddr(reinterpret_cast<uint64_t *>(&ffts_addr), &ffts_len);

    stage1_kernel<<<blockDim, nullptr, stream>>>(
        (__gm__ uint8_t *)ffts_addr, (__gm__ uint8_t *)query, (__gm__ uint8_t *)key, (__gm__ uint8_t *)value, (__gm__ uint8_t *)beta,
        (__gm__ uint8_t *)gOptional, (__gm__ uint8_t *)stageOneMask, (__gm__ uint8_t *)qktOut, (__gm__ uint8_t *)gCumExpOut,
        (__gm__ uint8_t *)kCumDecayOut, (__gm__ uint8_t *)vInnerOut, (__gm__ uint8_t *)qPrimeOut, (__gm__ uint8_t *)kgOut,
        (__gm__ uint8_t *)workspaceGM, (__gm__ uint8_t *)tilingGM);
}
