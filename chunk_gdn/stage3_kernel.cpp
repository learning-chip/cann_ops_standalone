// Standalone Stage3 wrapper: real `Stage3` from chunk_gated_delta_rule_stage3.h
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "chunk_gated_delta_rule_stage3.h"
#include "runtime/rt.h"

using namespace AscendC;
using namespace ChunkGatedDeltaRule;
using namespace matmul;

extern "C" __global__ __aicore__ void stage3_kernel(
    GM_ADDR fftsAddr,
    GM_ADDR qkt, GM_ADDR gCumExp, GM_ADDR attnInter, GM_ADDR vInner,
    GM_ADDR stageThreeMaskBase, GM_ADDR attnOutBf16,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    SetSyncBaseAddr((unsigned long)fftsAddr);
    SetAtomicNone();
    SetMaskNorm();

    REGISTER_TILING_DEFAULT(ChunkGatedDeltaRuleTilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    TPipe pipe;
    StageThreeMT mm3;
    if ASCEND_IS_AIC {
        mm3.Init(&tilingData.matmulTilingFp32, &pipe);
    }

    int64_t sp = (tilingData.t + tilingData.chunkSize - 1) / tilingData.chunkSize * tilingData.chunkSize;

    GlobalTensor<float> qktGm;
    GlobalTensor<float> gCumExpGm;
    GlobalTensor<float> attnInterGm;
    GlobalTensor<float> vInnerGm;
    GlobalTensor<float> maskGm;
    GlobalTensor<bfloat16_t> outGm;

    qktGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(qkt), tilingData.nv * sp * tilingData.chunkSize);
    gCumExpGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gCumExp), tilingData.nv * sp);
    attnInterGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(attnInter), tilingData.nv * sp * tilingData.dv);
    vInnerGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(vInner), tilingData.nv * sp * tilingData.dv);
    maskGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(stageThreeMaskBase),
                           tilingData.chunkSize * tilingData.chunkSize * tilingData.aiCoreNum * TASK_RATIO);
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(attnOutBf16), tilingData.t * tilingData.nv * tilingData.dv);

    ChunkGroup cg;
    cg.startPos = 0;
    cg.length = tilingData.t;
    cg.chunkSize = tilingData.chunkSize;
    cg.coreStart = 0;
    cg.coreEnd = 0;

    StageThreeParams params{
        qktGm,
        gCumExpGm,
        attnInterGm,
        vInnerGm,
        maskGm[(int64_t)(GetBlockIdx() / 2) * tilingData.chunkSize * tilingData.chunkSize],
        workspaceGM,
        outGm,
        &mm3,
        &pipe,
        &cg,
        tilingData.scale,
        tilingData.nv,
        tilingData.nk,
        tilingData.dv,
        tilingData.dk,
        tilingData.hasGamma != 0};

    Stage3 op;
    op.Init(&params, static_cast<int32_t>(tilingData.aiCoreNum));
    op.Process();
}

extern "C" void call_stage3(
    uint32_t blockDim, void *stream,
    uint8_t *qkt, uint8_t *gCumExp, uint8_t *attnInter, uint8_t *vInner,
    uint8_t *stageThreeMaskBase, uint8_t *attnOutBf16,
    uint8_t *workspaceGM, uint8_t *tilingGM)
{
    void *ffts_addr = nullptr;
    uint32_t ffts_len = 0;
    rtGetC2cCtrlAddr(reinterpret_cast<uint64_t *>(&ffts_addr), &ffts_len);

    stage3_kernel<<<blockDim, nullptr, stream>>>(
        (__gm__ uint8_t *)ffts_addr,
        (__gm__ uint8_t *)qkt, (__gm__ uint8_t *)gCumExp, (__gm__ uint8_t *)attnInter, (__gm__ uint8_t *)vInner,
        (__gm__ uint8_t *)stageThreeMaskBase, (__gm__ uint8_t *)attnOutBf16,
        (__gm__ uint8_t *)workspaceGM, (__gm__ uint8_t *)tilingGM);
}
