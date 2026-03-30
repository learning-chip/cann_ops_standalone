#include "runtime/rt.h"
#include "multi_latent_attention_mix.cce"

extern "C" void call_kernel(
    uint32_t block_dim, void *stream,
    uint8_t *q_gm,
    uint8_t *kv_gm,
    uint8_t *layerID_gm,
    uint8_t *mask_gm,
    uint8_t *deq_qk_gm,
    uint8_t *off_qk_gm,
    uint8_t *deq_pv_gm,
    uint8_t *off_pv_gm,
    uint8_t *quant_p_gm,
    uint8_t *o_gm,
    uint8_t *s_gm,
    uint8_t *p_gm,
    uint8_t *o_tmp_gm,
    uint8_t *upo_tmp_gm,
    uint8_t *tiling_para_gm)
{
    void *ffts_addr = nullptr;
    uint32_t ffts_len = 0;
    rtGetC2cCtrlAddr(reinterpret_cast<uint64_t *>(&ffts_addr), &ffts_len);

    multi_latent_attention_mix<<<block_dim, nullptr, stream>>>(
        (__gm__ uint8_t *)ffts_addr,
        (__gm__ uint8_t *)q_gm,
        (__gm__ uint8_t *)kv_gm,
        (__gm__ uint8_t *)layerID_gm,
        (__gm__ uint8_t *)mask_gm,
        (__gm__ uint8_t *)deq_qk_gm,
        (__gm__ uint8_t *)off_qk_gm,
        (__gm__ uint8_t *)deq_pv_gm,
        (__gm__ uint8_t *)off_pv_gm,
        (__gm__ uint8_t *)quant_p_gm,
        (__gm__ uint8_t *)o_gm,
        (__gm__ uint8_t *)s_gm,
        (__gm__ uint8_t *)p_gm,
        (__gm__ uint8_t *)o_tmp_gm,
        (__gm__ uint8_t *)upo_tmp_gm,
        (__gm__ uint8_t *)tiling_para_gm);
}
