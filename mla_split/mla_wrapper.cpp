#include "kernel_operator.h"
#include "runtime/rt.h"
#include "mla.cce"

extern "C" __global__ __aicore__ void mla_direct(
    __gm__ uint8_t *__restrict__ sync,
    __gm__ uint8_t *__restrict__ q_gm,
    __gm__ uint8_t *__restrict__ q_rope_gm,
    __gm__ uint8_t *__restrict__ ctkv_gm,
    __gm__ uint8_t *__restrict__ ctkv_rope_gm,
    __gm__ uint8_t *__restrict__ block_tables_gm,
    __gm__ uint8_t *__restrict__ mask_gm,
    __gm__ uint8_t *__restrict__ deq_qk_gm,
    __gm__ uint8_t *__restrict__ deq_pv_gm,
    __gm__ uint8_t *__restrict__ o_gm,
    __gm__ uint8_t *__restrict__ lse_gm,
    __gm__ uint8_t *__restrict__ s_gm,
    __gm__ uint8_t *__restrict__ s_rope_out_gm,
    __gm__ uint8_t *__restrict__ p_gm,
    __gm__ uint8_t *__restrict__ o_tmp_gm,
    __gm__ uint8_t *__restrict__ go_gm,
    __gm__ uint8_t *__restrict__ o_core_tmp_gm,
    __gm__ uint8_t *__restrict__ l_gm,
    __gm__ uint8_t *__restrict__ tiling_para_gm)
{
    SetFftsBaseAddr((unsigned long)sync);
    SetAtomicnone();
    SetMasknorm();
#ifdef __DAV_C220_VEC__
    SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
#elif __DAV_C220_CUBE__
    SetPadding<uint64_t>(0);
    SetNdpara(1, 0, 0);
#endif
#ifdef __DAV_C220_CUBE__
    MLAttentionDecoderAic<TilingKeyType::TILING_HALF_DATA, half, half, half, half, InputFormat::ND_FORMAT> pa_aic_fp16 {};
    pa_aic_fp16.SetArgs(sync, q_gm, q_rope_gm, ctkv_gm, ctkv_rope_gm, block_tables_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, tiling_para_gm);
    pa_aic_fp16.Run();
#elif __DAV_C220_VEC__
    MLADecoderAiv<TilingKeyType::TILING_HALF_DATA, half, half> pa_aiv {};
    pa_aiv.SetArgs(sync, block_tables_gm, deq_qk_gm, deq_pv_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, go_gm, tiling_para_gm, mask_gm);
    pa_aiv.Run();
#else
    (void)q_gm;
    (void)q_rope_gm;
    (void)ctkv_gm;
    (void)ctkv_rope_gm;
    (void)block_tables_gm;
    (void)mask_gm;
    (void)deq_qk_gm;
    (void)deq_pv_gm;
    (void)o_gm;
    (void)lse_gm;
    (void)s_gm;
    (void)s_rope_out_gm;
    (void)p_gm;
    (void)o_tmp_gm;
    (void)go_gm;
    (void)o_core_tmp_gm;
    (void)l_gm;
    (void)tiling_para_gm;
#endif
}

extern "C" void call_kernel(
    uint32_t block_dim, void *stream,
    uint8_t *q_gm,
    uint8_t *q_rope_gm,
    uint8_t *ctkv_gm,
    uint8_t *ctkv_rope_gm,
    uint8_t *block_tables_gm,
    uint8_t *mask_gm,
    uint8_t *deq_qk_gm,
    uint8_t *deq_pv_gm,
    uint8_t *o_gm,
    uint8_t *lse_gm,
    uint8_t *s_gm,
    uint8_t *s_rope_out_gm,
    uint8_t *p_gm,
    uint8_t *o_tmp_gm,
    uint8_t *go_gm,
    uint8_t *o_core_tmp_gm,
    uint8_t *l_gm,
    uint8_t *tiling_para_gm)
{
    void *ffts_addr = nullptr;
    uint32_t ffts_len = 0;
    rtGetC2cCtrlAddr(reinterpret_cast<uint64_t *>(&ffts_addr), &ffts_len);

    mla_direct<<<block_dim, nullptr, stream>>>(
        (__gm__ uint8_t *)ffts_addr,
        (__gm__ uint8_t *)q_gm,
        (__gm__ uint8_t *)q_rope_gm,
        (__gm__ uint8_t *)ctkv_gm,
        (__gm__ uint8_t *)ctkv_rope_gm,
        (__gm__ uint8_t *)block_tables_gm,
        (__gm__ uint8_t *)mask_gm,
        (__gm__ uint8_t *)deq_qk_gm,
        (__gm__ uint8_t *)deq_pv_gm,
        (__gm__ uint8_t *)o_gm,
        (__gm__ uint8_t *)lse_gm,
        (__gm__ uint8_t *)s_gm,
        (__gm__ uint8_t *)s_rope_out_gm,
        (__gm__ uint8_t *)p_gm,
        (__gm__ uint8_t *)o_tmp_gm,
        (__gm__ uint8_t *)go_gm,
        (__gm__ uint8_t *)o_core_tmp_gm,
        (__gm__ uint8_t *)l_gm,
        (__gm__ uint8_t *)tiling_para_gm);
}
