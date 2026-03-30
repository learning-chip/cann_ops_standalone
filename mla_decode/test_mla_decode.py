import ctypes
import math
import os

import numpy as np
import torch
import torch_npu


def as_ptr(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def load_kernel(lib_path: str):
    lib = ctypes.CDLL(lib_path)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,  # block dim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # q_gm
        ctypes.c_void_p,  # kv_gm
        ctypes.c_void_p,  # layerID_gm
        ctypes.c_void_p,  # mask_gm
        ctypes.c_void_p,  # deq_qk_gm
        ctypes.c_void_p,  # off_qk_gm
        ctypes.c_void_p,  # deq_pv_gm
        ctypes.c_void_p,  # off_pv_gm
        ctypes.c_void_p,  # quant_p_gm
        ctypes.c_void_p,  # o_gm
        ctypes.c_void_p,  # s_gm
        ctypes.c_void_p,  # p_gm
        ctypes.c_void_p,  # o_tmp_gm
        ctypes.c_void_p,  # upo_tmp_gm
        ctypes.c_void_p,  # tiling_para_gm
    ]
    lib.call_kernel.restype = None
    return lib


def _split_u64(v: int):
    return (v >> 32) & 0xFFFFFFFF, v & 0xFFFFFFFF


def make_tiling(
    batch: int,
    q_seq_lens,
    kv_seq_lens,
    heads: int,
    embd: int,
    kv_heads: int,
    embdv: int,
    tor: float,
    device: str,
):
    # Mirrors encoder path in flash_attention_tiling_dependency.cpp for MLA combine cache.
    TILING_HEAD_SIZE = 37
    TILING_PARA_SIZE = 27
    BLOCK_SIZE = 16
    LONG_SEQ_LEN = 128
    PP_MM = [16, 32, 48, 64, 80, 96, 112, 128]
    PP_NN = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256]

    total_words = TILING_HEAD_SIZE + batch * TILING_PARA_SIZE + 16
    tiling = torch.zeros(total_words, dtype=torch.int32, device=device)
    tor_u32 = np.frombuffer(np.float32(tor).tobytes(), dtype=np.uint32)[0]

    total_q_blk_num = 0
    addr_q = 0
    addr_k = 0
    addr_v = 0
    addr_o = 0
    max_seq = max(kv_seq_lens)
    max_q_seq = max(q_seq_lens)

    for seq_idx in range(batch):
        q_seqlen = int(q_seq_lens[seq_idx])
        kv_seqlen = int(kv_seq_lens[seq_idx])
        q_aligned = (q_seqlen + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
        kv_aligned = (kv_seqlen + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE

        m_ubd = min(LONG_SEQ_LEN, q_aligned)
        m_idx = 7 if m_ubd > PP_MM[7] else (m_ubd // 16 - 1)
        m_ubd = PP_MM[m_idx]
        n_ubd = min(LONG_SEQ_LEN, kv_aligned)
        n_idx = 15 if n_ubd > PP_NN[15] else (n_ubd // 16 - 1)
        n_ubd = PP_NN[n_idx]

        if q_seqlen != 0 and kv_seqlen != 0:
            total_q_blk_num += (q_seqlen + m_ubd - 1) // m_ubd

        base = TILING_HEAD_SIZE + seq_idx * TILING_PARA_SIZE
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
        tiling[base + 14] = 1  # batch state

        addr_q += q_seqlen * heads * embd
        # MLA combine cache packs K and V in one kv buffer with stride_k*2.
        addr_k += kv_seqlen * kv_heads * embd * 2
        addr_v += kv_seqlen * kv_heads * embdv
        addr_o += q_seqlen * heads * embdv

    tiling[0] = batch
    tiling[1] = max_seq
    tiling[2] = heads
    tiling[3] = embd
    tiling[4] = kv_heads
    tiling[5] = int(tor_u32)
    tiling[6] = 0  # headStride for MASK_TYPE_NO_BATCH
    tiling[7] = 0  # maskStride for MASK_TYPE_NO_BATCH
    tiling[8] = 0  # isTriuMask
    tiling[9] = total_q_blk_num
    tiling[10] = 0  # isClamp
    tiling[13] = 0  # headStride duplicate
    tiling[14] = TILING_HEAD_SIZE
    tiling[15] = TILING_PARA_SIZE
    tiling[16] = 0  # MLA combine-cache fp16 (embedding > 256)
    tiling[17] = 0  # isLongSeq
    tiling[18] = max_seq
    tiling[20] = 2  # MASK_TYPE_NO_BATCH
    tiling[23] = embdv
    tiling[24] = 0  # quantType
    tiling[25] = 0  # dataShapeType ND
    tiling[26] = 0  # SCALE_TOR
    tiling[29] = max_q_seq
    return tiling, total_q_blk_num * heads


def run_smoke():
    device = "npu"
    dtype = torch.float16
    torch.npu.set_device(device)
    here = os.path.dirname(os.path.abspath(__file__))
    lib = load_kernel(os.path.join(here, "mla_decode_lib.so"))

    batch = 1
    heads = 1
    kv_heads = 1
    embd = 576
    embdv = 512
    max_seq = 128
    q_tokens = batch * max_seq

    q = torch.randn(q_tokens, heads * embd, dtype=dtype, device=device)
    kv = torch.randn(1, batch, max_seq, kv_heads * embd, dtype=dtype, device=device)
    layer_id = torch.tensor([], dtype=torch.int32, device=device)
    mask = torch.triu(
        torch.full((max_seq, max_seq), -10000.0, dtype=dtype, device=device),
        diagonal=1,
    )
    deq_qk = torch.empty(0, dtype=torch.float32, device=device)
    off_qk = torch.empty(0, dtype=torch.int32, device=device)
    deq_pv = torch.empty(0, dtype=torch.float32, device=device)
    off_pv = torch.empty(0, dtype=torch.int32, device=device)
    quant_p = torch.empty(0, dtype=torch.float32, device=device)

    tor = 1.0 / math.sqrt(float(embd))
    tiling, block_dim = make_tiling(
        batch=batch,
        q_seq_lens=[max_seq],
        kv_seq_lens=[max_seq],
        heads=heads,
        embd=embd,
        kv_heads=kv_heads,
        embdv=embdv,
        tor=tor,
        device=device,
    )

    o = torch.zeros(q_tokens, heads * embdv, dtype=dtype, device=device)
    tmp_size = 32768 * 4
    s = torch.empty(block_dim * tmp_size, dtype=torch.float32, device=device)
    p = torch.empty(block_dim * tmp_size, dtype=dtype, device=device)
    o_tmp = torch.empty(block_dim * tmp_size * 6, dtype=torch.float32, device=device)
    upo_tmp = torch.empty(0, dtype=torch.float32, device=device)

    stream_ptr = torch.npu.current_stream()._as_parameter_
    o_before = o.clone()
    lib.call_kernel(
        block_dim,
        stream_ptr,
        as_ptr(q),
        as_ptr(kv[0].contiguous()),
        as_ptr(layer_id),
        as_ptr(mask),
        as_ptr(deq_qk),
        as_ptr(off_qk),
        as_ptr(deq_pv),
        as_ptr(off_pv),
        as_ptr(quant_p),
        as_ptr(o),
        as_ptr(s),
        as_ptr(p),
        as_ptr(o_tmp),
        as_ptr(upo_tmp),
        as_ptr(tiling),
    )
    torch.npu.synchronize()

    q_bmh = q.view(1, max_seq, heads, embd).transpose(1, 2)
    k_bmh = kv[0].view(1, max_seq, kv_heads, embd).transpose(1, 2)
    v_bmh = kv[0][:, :, :embdv].contiguous().view(1, max_seq, kv_heads, embdv).transpose(1, 2)
    score = torch.matmul(q_bmh.float(), k_bmh.float().transpose(-2, -1)) * tor
    score = score + mask.view(1, 1, max_seq, max_seq).float()
    prob = torch.softmax(score, dim=-1).to(torch.float32)
    o_ref = torch.matmul(prob, v_bmh.float()).to(dtype)
    o_ref = o_ref.transpose(1, 2).contiguous().view(q_tokens, heads * embdv)

    changed = bool(torch.any(o != o_before).item())
    diff = torch.mean(torch.abs(o.float() - o_ref.float())).item()
    max_diff = torch.max(torch.abs(o.float() - o_ref.float())).item()
    torch.testing.assert_close(
        o, o_ref, rtol=6e-3, atol=6e-3, msg="multi_latent_attention_mix output vs torch reference"
    )
    print("multi_latent_attention_mix call finished.")
    print(f"output mean abs: {o.abs().mean().item():.6f}")
    print(f"output finite: {bool(torch.isfinite(o).all().item())}")
    print(f"output changed: {changed}")
    print(f"mean abs diff vs ref: {diff:.6f}")
    print(f"max abs diff vs ref: {max_diff:.6f}")
    print(f"tor example: {tor:.6f}")


if __name__ == "__main__":
    run_smoke()
