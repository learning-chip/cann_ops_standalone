// Standalone Stage2 wrapper: real `Stage2` from chunk_gated_delta_rule_stage2.h
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "chunk_gated_delta_rule_stage2.h"
#include "runtime/rt.h"

using namespace AscendC;
using namespace ChunkGatedDeltaRule;
using namespace matmul;

extern "C" __global__ __aicore__ void stage2_kernel(
    GM_ADDR fftsAddr,
    GM_ADDR qPrime, GM_ADDR vInner, GM_ADDR gCumExp, GM_ADDR kCumDecay,
    GM_ADDR curState, GM_ADDR kg, GM_ADDR attnInter,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    SetSyncBaseAddr((unsigned long)fftsAddr);
    SetAtomicNone();
    SetMaskNorm();

    REGISTER_TILING_DEFAULT(ChunkGatedDeltaRuleTilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    TPipe pipe;
    StageTwoMT mm;
    if ASCEND_IS_AIC {
        mm.Init(&tilingData.matmulTilingFp32, &pipe);
    }

    int64_t sp = (tilingData.t + tilingData.chunkSize - 1) / tilingData.chunkSize * tilingData.chunkSize;

    GlobalTensor<float> qPrimeGm;
    GlobalTensor<float> vInnerGm;
    GlobalTensor<float> gCumExpGm;
    GlobalTensor<float> kCumDecayGm;
    GlobalTensor<float> curStateGm;
    GlobalTensor<float> kgGm;
    GlobalTensor<float> attnInterGm;

    qPrimeGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(qPrime), tilingData.nv * sp * tilingData.dk);
    vInnerGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(vInner), tilingData.nv * sp * tilingData.dv);
    gCumExpGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gCumExp), tilingData.nv * sp);
    kCumDecayGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(kCumDecay), tilingData.nv * sp * tilingData.dk);
    curStateGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(curState), tilingData.nv * tilingData.dv * tilingData.dk);
    kgGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(kg), tilingData.nv * sp * tilingData.dk);
    attnInterGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(attnInter), tilingData.nv * sp * tilingData.dv);

    ChunkGroup cg;
    cg.startPos = 0;
    cg.length = tilingData.t;
    cg.chunkSize = tilingData.chunkSize;
    cg.coreStart = 0;
    cg.coreEnd = 0;

    StageTwoParams params{
        qPrimeGm, vInnerGm, gCumExpGm, kCumDecayGm, curStateGm, kgGm,
        attnInterGm, workspaceGM, &mm, &pipe, &cg,
        tilingData.nv, tilingData.nk, tilingData.dv, tilingData.dk, tilingData.hasGamma != 0};

    Stage2 op;
    op.Init(&params, static_cast<int32_t>(tilingData.aiCoreNum));
    op.Process();
}

extern "C" void call_stage2(
    uint32_t blockDim, void *stream,
    uint8_t *qPrime, uint8_t *vInner, uint8_t *gCumExp, uint8_t *kCumDecay,
    uint8_t *curState, uint8_t *kg, uint8_t *attnInter,
    uint8_t *workspaceGM, uint8_t *tilingGM)
{
    void *ffts_addr = nullptr;
    uint32_t ffts_len = 0;
    rtGetC2cCtrlAddr(reinterpret_cast<uint64_t *>(&ffts_addr), &ffts_len);

    stage2_kernel<<<blockDim, nullptr, stream>>>(
        (__gm__ uint8_t *)ffts_addr,
        (__gm__ uint8_t *)qPrime, (__gm__ uint8_t *)vInner, (__gm__ uint8_t *)gCumExp, (__gm__ uint8_t *)kCumDecay,
        (__gm__ uint8_t *)curState, (__gm__ uint8_t *)kg, (__gm__ uint8_t *)attnInter,
        (__gm__ uint8_t *)workspaceGM, (__gm__ uint8_t *)tilingGM);
}
