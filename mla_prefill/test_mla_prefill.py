import ctypes
import math
import os

import numpy as np
import torch
import torch_npu


def as_ptr(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def make_empty(device: str, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(0, device=device, dtype=dtype)


def load_kernel(lib_path: str):
    lib = ctypes.CDLL(lib_path)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,  # block dim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # q_gm
        ctypes.c_void_p,  # q_rope_gm
        ctypes.c_void_p,  # k_gm
        ctypes.c_void_p,  # k_rope_gm
        ctypes.c_void_p,  # v_gm
        ctypes.c_void_p,  # mask_gm
        ctypes.c_void_p,  # alibi_coeff_gm
        ctypes.c_void_p,  # deq_qk_gm
        ctypes.c_void_p,  # off_qk_gm
        ctypes.c_void_p,  # deq_pv_gm
        ctypes.c_void_p,  # off_pv_gm
        ctypes.c_void_p,  # quant_p_gm
        ctypes.c_void_p,  # logN_gm
        ctypes.c_void_p,  # o_gm
        ctypes.c_void_p,  # s_gm
        ctypes.c_void_p,  # p_gm
        ctypes.c_void_p,  # o_tmp_gm
        ctypes.c_void_p,  # upo_tmp_gm
        ctypes.c_void_p,  # tiling_para_gm
    ]
    lib.call_kernel.restype = None
    return lib


def split_qk_for_mla(q: torch.Tensor, k: torch.Tensor, heads: int, kv_heads: int):
    """Split nope/rope halves for MLA. Results must be contiguous: the kernel indexes GM with
    row stride q_heads*128 / kv_heads*128 (see mla_prefill.cce stride_qo / stride_k). Slices of
    the full [L, H*192] / [B,L,H*192] tensors have row stride 192 and would feed wrong rows to the kernel."""
    q_split1 = q[:, :128]
    q_split2 = q[:, 128:192]
    for i in range(1, heads):
        q_split1 = torch.cat([q_split1, q[:, i * 192 : i * 192 + 128]], dim=1)
        q_split2 = torch.cat([q_split2, q[:, i * 192 + 128 : (i + 1) * 192]], dim=1)

    k_split1 = k[:, :, :, :128]
    k_split2 = k[:, :, :, 128:192]
    for i in range(1, kv_heads):
        k_split1 = torch.cat([k_split1, k[:, :, :, i * 192 : i * 192 + 128]], dim=3)
        k_split2 = torch.cat([k_split2, k[:, :, :, i * 192 + 128 : (i + 1) * 192]], dim=3)
    return q_split1.contiguous(), q_split2.contiguous(), k_split1.contiguous(), k_split2.contiguous()


def _split_u64(v: int):
    return (v >> 32) & 0xFFFFFFFF, v & 0xFFFFFFFF


def make_prefill_tiling(
    batch: int,
    q_seqlens,
    kv_seqlens,
    q_heads: int,
    embed: int,
    kv_heads: int,
    embdv: int,
    tor: float,
    q_offset_elems: int,
    k_offset_elems: int,
    v_offset_elems: int,
    tiling_key: int,
    device: str,
) -> torch.Tensor:
    # Mirror essential fields from GetMLAPrefillTilingParam for one standalone case.
    TILING_HEAD_SIZE_PREFILL = 19
    TILING_PARA_SIZE_PREFILL = 27
    PP_MM = [16, 32, 48, 64, 80, 96, 112, 128]
    PP_NN = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256]
    BLOCK_SIZE = 16
    LONG_SEQ_LEN = 128

    max_seq = max(kv_seqlens)
    total_words = TILING_HEAD_SIZE_PREFILL + batch * TILING_PARA_SIZE_PREFILL + 16
    tiling = torch.zeros(total_words, dtype=torch.int32, device=device)

    tor_u32 = np.frombuffer(np.float32(tor).tobytes(), dtype=np.uint32)[0]

    total_q_blk_num = 0
    addr_q = q_offset_elems
    addr_k = k_offset_elems
    addr_v = v_offset_elems
    addr_o = 0

    for seq_idx in range(batch):
        q_seqlen = int(q_seqlens[seq_idx])
        kv_seqlen = int(kv_seqlens[seq_idx])
        q_seqlen_aligned = (q_seqlen + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
        kv_seqlen_aligned = (kv_seqlen + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE

        m_ubd = min(LONG_SEQ_LEN, q_seqlen_aligned)
        m_idx = 7 if m_ubd > PP_MM[7] else (m_ubd // 16 - 1)
        m_ubd = PP_MM[m_idx]
        n_ubd = min(LONG_SEQ_LEN, kv_seqlen_aligned)
        n_idx = 15 if n_ubd > PP_NN[15] else (n_ubd // 16 - 1)
        n_ubd = PP_NN[n_idx]

        if q_seqlen != 0 and kv_seqlen != 0:
            total_q_blk_num += (q_seqlen + m_ubd - 1) // m_ubd

        base = TILING_HEAD_SIZE_PREFILL + seq_idx * TILING_PARA_SIZE_PREFILL
        tiling[base + 0] = q_seqlen
        tiling[base + 1] = kv_seqlen
        tiling[base + 2] = m_ubd
        tiling[base + 3] = n_ubd
        q_hi, q_lo = _split_u64(addr_q)
        k_hi, k_lo = _split_u64(addr_k)
        v_hi, v_lo = _split_u64(addr_v)
        o_hi, o_lo = _split_u64(addr_o)
        tiling[base + 4] = q_hi
        tiling[base + 5] = q_lo
        tiling[base + 6] = k_hi
        tiling[base + 7] = k_lo
        tiling[base + 8] = v_hi
        tiling[base + 9] = v_lo
        tiling[base + 10] = o_hi
        tiling[base + 11] = o_lo
        tiling[base + 13] = total_q_blk_num
        tiling[base + 14] = 1

        # Match GetMLAPrefillTilingParam / PrefillTilingParam (embeddingSizeV for Q/K/V packed strides).
        addr_q += q_seqlen * q_heads * embdv
        addr_k += kv_seqlen * kv_heads * embdv
        addr_v += kv_seqlen * kv_heads * embdv
        addr_o += q_seqlen * q_heads * embdv

    tiling[0] = batch
    tiling[1] = max_seq
    tiling[2] = q_heads
    tiling[3] = embed
    tiling[4] = kv_heads
    tiling[5] = int(tor_u32)
    tiling[6] = 0
    tiling[7] = 0
    tiling[9] = total_q_blk_num
    tiling[10] = TILING_HEAD_SIZE_PREFILL
    tiling[11] = TILING_PARA_SIZE_PREFILL
    tiling[12] = tiling_key
    tiling[14] = max_seq
    tiling[16] = embdv
    tiling[18] = 0
    return tiling


def run_smoke():
    device = "npu"
    dtype = torch.float16
    torch.npu.set_device(device)

    here = os.path.dirname(os.path.abspath(__file__))
    lib = load_kernel(os.path.join(here, "mla_prefill_lib.so"))

    batch = 1
    heads = 1
    kv_heads = 1
    embd = 192
    embdv = 128
    max_seq = 128
    q_tokens = batch * max_seq

    q = torch.randn(q_tokens, heads * embd, dtype=dtype, device=device)
    k = torch.randn(1, batch, max_seq, kv_heads * embd, dtype=dtype, device=device)
    v = torch.randn(1, batch, max_seq, kv_heads * embdv, dtype=dtype, device=device)
    q_split1, q_split2, k_split1, k_split2 = split_qk_for_mla(q, k, heads, kv_heads)

    mask = make_empty(device, dtype)
    alibi = make_empty(device, torch.float32)
    deq_qk = make_empty(device, torch.float32)
    off_qk = make_empty(device, torch.int32)
    deq_pv = make_empty(device, torch.float32)
    off_pv = make_empty(device, torch.int32)
    quant_p = make_empty(device, torch.float32)
    log_n = make_empty(device, torch.float32)

    block_dim = heads  # num_heads * total_q_blk_num for this simple case
    o = torch.zeros(q_tokens, heads * embdv, dtype=dtype, device=device)
    # Keep workspace sizes aligned with kernel constants to avoid OOB writes.
    tmp_size = 32768 * 16
    s = torch.empty(block_dim * tmp_size, dtype=dtype, device=device)
    p = torch.empty(block_dim * tmp_size, dtype=dtype, device=device)
    o_tmp = torch.empty(block_dim * tmp_size, dtype=torch.float32, device=device)
    upo_tmp = make_empty(device, torch.float32)

    tor = 1.0 / math.sqrt(float(embd))
    tiling = make_prefill_tiling(
        batch=batch,
        q_seqlens=[max_seq],
        kv_seqlens=[max_seq],
        q_heads=heads,
        embed=embd,
        kv_heads=kv_heads,
        embdv=embdv,
        tor=tor,
        q_offset_elems=0,
        k_offset_elems=0,
        v_offset_elems=0,
        tiling_key=1,  # mask free
        device=device,
    )

    stream_ptr = torch.npu.current_stream()._as_parameter_
    o_before = o.clone()
    lib.call_kernel(
        block_dim,
        stream_ptr,
        as_ptr(q_split1),
        as_ptr(q_split2),
        as_ptr(k_split1[0]),
        as_ptr(k_split2[0]),
        as_ptr(v[0].contiguous()),
        as_ptr(mask),
        as_ptr(alibi),
        as_ptr(deq_qk),
        as_ptr(off_qk),
        as_ptr(deq_pv),
        as_ptr(off_pv),
        as_ptr(quant_p),
        as_ptr(log_n),
        as_ptr(o),
        as_ptr(s),
        as_ptr(p),
        as_ptr(o_tmp),
        as_ptr(upo_tmp),
        as_ptr(tiling),
    )
    torch.npu.synchronize()

    # Torch reference: same math as fused split matmul; SDPA matches float32 softmax + matmul path here.
    scale = 1.0 / math.sqrt(float(embd))
    q_bmh = q.view(1, max_seq, heads, embd).transpose(1, 2)
    k_bmh = k[0].view(1, max_seq, kv_heads, embd).transpose(1, 2)
    v_bmh = v[0].view(1, max_seq, kv_heads, embdv).transpose(1, 2)
    o_ref = torch.nn.functional.scaled_dot_product_attention(
        q_bmh, k_bmh, v_bmh, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale
    )
    o_ref = o_ref.transpose(1, 2).contiguous().view(q_tokens, heads * embdv)

    changed = bool(torch.any(o != o_before).item())
    diff = torch.mean(torch.abs(o.float() - o_ref.float()))
    max_diff = torch.max(torch.abs(o.float() - o_ref.float())).item()
    # fp16 attention vs SDPA: allow small numerical gap (kernel uses fused online softmax).
    torch.testing.assert_close(o, o_ref, rtol=5e-3, atol=5e-3, msg="mla_prefill output vs torch SDPA")
    print("mla_prefill call finished.")
    print(f"output mean abs: {o.abs().mean().item():.6f}")
    print(f"output finite: {bool(torch.isfinite(o).all().item())}")
    print(f"output changed: {changed}")
    print(f"mean abs diff vs ref: {diff.item():.6f}")
    print(f"max abs diff vs ref: {max_diff:.6f}")
    print(f"tor example: {tor:.6f}")


if __name__ == "__main__":
    run_smoke()
