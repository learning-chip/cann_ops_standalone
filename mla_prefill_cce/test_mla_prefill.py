import ctypes
import csv
import math
import os
<<<<<<< Updated upstream
=======
import zlib
>>>>>>> Stashed changes

import numpy as np
import torch
import torch_npu

BLOCK_DIM = int(getattr(torch.npu.get_device_properties("npu:0"), "cube_core_num", 20))


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

    o = torch.zeros(q_tokens, heads * embdv, dtype=dtype, device=device)
    # Keep workspace sizes aligned with kernel constants to avoid OOB writes.
    tmp_size = 32768 * 16
    s = torch.empty(BLOCK_DIM * tmp_size, dtype=dtype, device=device)
    p = torch.empty(BLOCK_DIM * tmp_size, dtype=dtype, device=device)
    o_tmp = torch.empty(BLOCK_DIM * tmp_size, dtype=torch.float32, device=device)
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
        BLOCK_DIM,
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


def benchmark_with_events(fn, warmup_iters: int = 5, benchmark_iters: int = 20) -> float:
    start_events = [torch.npu.Event(enable_timing=True) for _ in range(benchmark_iters)]
    end_events = [torch.npu.Event(enable_timing=True) for _ in range(benchmark_iters)]
    for _ in range(warmup_iters):
        fn()
    torch.npu.synchronize()
    for i in range(benchmark_iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.npu.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return float(sum(times_ms) / len(times_ms))


def estimate_flops(batch: int, heads: int, q_seq: int, kv_seq: int, embd: int, embdv: int) -> float:
    qk = 2.0 * batch * heads * q_seq * kv_seq * embd
    pv = 2.0 * batch * heads * q_seq * kv_seq * embdv
    return qk + pv


<<<<<<< Updated upstream
=======
def estimate_flops_fused_infer(batch: int, heads: int, q_seq: int, kv_seq: int, head_dim: int) -> float:
    """FLOPs for standard MHA as in `torch_npu.npu_fused_infer_attention_score` (QK + PV matmuls, same head_dim)."""
    qk = 2.0 * batch * heads * q_seq * kv_seq * head_dim
    pv = 2.0 * batch * heads * q_seq * kv_seq * head_dim
    return qk + pv


# Rope slot width for MLA packed Q/K (128 nope + MLA_ROPE_DIM = 192 per head). Fused op uses the same split via query_rope/key_rope.
MLA_ROPE_DIM = 64


def bnsd128_to_mla_packed(
    fq: torch.Tensor,
    fk: torch.Tensor,
    fv: torch.Tensor,
    batch: int,
    heads: int,
    q_seq: int,
    kv_seq: int,
    dtype: torch.dtype,
    device: str,
):
    """Map shared BNSD (D=128) tensors to MLA packed Q/K (192 = 128 nope + rope) and V (128).

    Also returns `query_rope` / `key_rope` in BNSD (..., MLA_ROPE_DIM) — the tensors concatenated
    after `fq`/`fk` — so `npu_fused_infer_attention_score(..., query_rope=..., key_rope=...)` matches
    the custom MLA kernel layout.
    """
    embd = 192
    embdv = 128
    rope_pad_q = torch.zeros(batch, heads, q_seq, MLA_ROPE_DIM, dtype=dtype, device=device)
    rope_pad_k = torch.zeros(batch, heads, kv_seq, MLA_ROPE_DIM, dtype=dtype, device=device)
    q_h192 = torch.cat([fq, rope_pad_q], dim=-1)
    k_h192 = torch.cat([fk, rope_pad_k], dim=-1)
    q_flat = q_h192.permute(0, 2, 1, 3).contiguous().view(batch * q_seq, heads * embd)
    k_flat = k_h192.permute(0, 2, 1, 3).contiguous().view(1, batch, kv_seq, heads * embd)
    v_flat = fv.permute(0, 2, 1, 3).contiguous().view(1, batch, kv_seq, heads * embdv)
    return q_flat, k_flat, v_flat, rope_pad_q, rope_pad_k


>>>>>>> Stashed changes
def run_benchmarks():
    device = "npu"
    dtype = torch.float16
    torch.npu.set_device(device)
    here = os.path.dirname(os.path.abspath(__file__))
    lib = load_kernel(os.path.join(here, "mla_prefill_lib.so"))

    # Larger shapes for higher FLOP rates while staying within reasonable memory.
<<<<<<< Updated upstream
    # Additional cases push FLOPs higher; cases that fail numerical checks are skipped below.
    cases = [
=======
    # All APIs use the same random BNSD tensors (D=128); MLA path pads Q/K to 192 with zero rope slots.
    # Throughput uses a common FLOP count (128-dim attention) so TFLOP/s is comparable across kernels.
    attn_head_dim = 128
    cases = [
        {
            "name": "prefill_op_plugin_v3_b1_h32_q1_kv2048",
            "batch": 1,
            "heads": 32,
            "kv_heads": 32,
            "embd": 192,
            "embdv": 128,
            "q_seq": 1,
            "kv_seq": 2048,
        },
>>>>>>> Stashed changes
        {"name": "prefill_b4_h16_s256", "batch": 4, "heads": 16, "kv_heads": 16, "embd": 192, "embdv": 128, "q_seq": 256, "kv_seq": 256},
        {"name": "prefill_b4_h16_s512", "batch": 4, "heads": 16, "kv_heads": 16, "embd": 192, "embdv": 128, "q_seq": 512, "kv_seq": 512},
        {"name": "prefill_b4_h16_s768", "batch": 4, "heads": 16, "kv_heads": 16, "embd": 192, "embdv": 128, "q_seq": 768, "kv_seq": 768},
        {"name": "prefill_b4_h16_s1024", "batch": 4, "heads": 16, "kv_heads": 16, "embd": 192, "embdv": 128, "q_seq": 1024, "kv_seq": 1024},
        {"name": "prefill_b8_h16_s768", "batch": 8, "heads": 16, "kv_heads": 16, "embd": 192, "embdv": 128, "q_seq": 768, "kv_seq": 768},
        {"name": "prefill_b8_h32_s512", "batch": 8, "heads": 32, "kv_heads": 32, "embd": 192, "embdv": 128, "q_seq": 512, "kv_seq": 512},
        {"name": "prefill_b8_h32_s768", "batch": 8, "heads": 32, "kv_heads": 32, "embd": 192, "embdv": 128, "q_seq": 768, "kv_seq": 768},
        {"name": "prefill_b16_h16_s512", "batch": 16, "heads": 16, "kv_heads": 16, "embd": 192, "embdv": 128, "q_seq": 512, "kv_seq": 512},
        {"name": "prefill_b16_h32_s256", "batch": 16, "heads": 32, "kv_heads": 32, "embd": 192, "embdv": 128, "q_seq": 256, "kv_seq": 256},
    ]
    error_warn_threshold = 1.0e-2
    results = []
    skipped_cases = []

    for case in cases:
        batch = case["batch"]
        heads = case["heads"]
        kv_heads = case["kv_heads"]
        embd = case["embd"]
        embdv = case["embdv"]
        q_seq = case["q_seq"]
        kv_seq = case["kv_seq"]
        q_tokens = batch * q_seq

<<<<<<< Updated upstream
        q = torch.randn(q_tokens, heads * embd, dtype=dtype, device=device)
        k = torch.randn(1, batch, kv_seq, kv_heads * embd, dtype=dtype, device=device)
        v = torch.randn(1, batch, kv_seq, kv_heads * embdv, dtype=dtype, device=device)
=======
        gen = torch.Generator(device=device)
        gen.manual_seed(
            (zlib.crc32(case["name"].encode("utf-8")) & 0xFFFFFFFF) ^ (batch << 20) ^ (heads << 10) ^ q_seq ^ kv_seq
        )
        fq = torch.randn(batch, heads, q_seq, attn_head_dim, dtype=dtype, device=device, generator=gen)
        fk = torch.randn(batch, heads, kv_seq, attn_head_dim, dtype=dtype, device=device, generator=gen)
        fv = torch.randn(batch, heads, kv_seq, attn_head_dim, dtype=dtype, device=device, generator=gen)

        q, k, v, query_rope, key_rope = bnsd128_to_mla_packed(fq, fk, fv, batch, heads, q_seq, kv_seq, dtype, device)
>>>>>>> Stashed changes
        q_split1, q_split2, k_split1, k_split2 = split_qk_for_mla(q, k, heads, kv_heads)
        mask = make_empty(device, dtype)
        alibi = make_empty(device, torch.float32)
        deq_qk = make_empty(device, torch.float32)
        off_qk = make_empty(device, torch.int32)
        deq_pv = make_empty(device, torch.float32)
        off_pv = make_empty(device, torch.int32)
        quant_p = make_empty(device, torch.float32)
        log_n = make_empty(device, torch.float32)
<<<<<<< Updated upstream
        tor = 1.0 / math.sqrt(float(embd))
=======
        # Same softmax scaling as SDPA / fused on the 128-dim attention (padded Q/K have zero last-64).
        attn_scale = 1.0 / math.sqrt(float(attn_head_dim))
        tor = attn_scale
>>>>>>> Stashed changes
        tiling = make_prefill_tiling(
            batch=batch,
            q_seqlens=[q_seq] * batch,
            kv_seqlens=[kv_seq] * batch,
            q_heads=heads,
            embed=embd,
            kv_heads=kv_heads,
            embdv=embdv,
            tor=tor,
            q_offset_elems=0,
            k_offset_elems=0,
            v_offset_elems=0,
            tiling_key=1,
            device=device,
        )

        o = torch.zeros(q_tokens, heads * embdv, dtype=dtype, device=device)
        tmp_size = 32768 * 16
        s = torch.empty(BLOCK_DIM * tmp_size, dtype=dtype, device=device)
        p = torch.empty(BLOCK_DIM * tmp_size, dtype=dtype, device=device)
        o_tmp = torch.empty(BLOCK_DIM * tmp_size, dtype=torch.float32, device=device)
        upo_tmp = make_empty(device, torch.float32)
        stream_ptr = torch.npu.current_stream()._as_parameter_

        def run_custom():
            lib.call_kernel(
                BLOCK_DIM,
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

        def run_ref():
<<<<<<< Updated upstream
            scale = 1.0 / math.sqrt(float(embd))
            q_bmh = q.view(batch, q_seq, heads, embd).transpose(1, 2)
            k_bmh = k[0].view(batch, kv_seq, kv_heads, embd).transpose(1, 2)
            v_bmh = v[0].view(batch, kv_seq, kv_heads, embdv).transpose(1, 2)
            o_ref_local = torch.nn.functional.scaled_dot_product_attention(
                q_bmh, k_bmh, v_bmh, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale
            )
            return o_ref_local.transpose(1, 2).contiguous().view(q_tokens, heads * embdv)

=======
            o_ref_local = torch.nn.functional.scaled_dot_product_attention(
                fq,
                fk,
                fv,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=attn_scale,
            )
            return o_ref_local.transpose(1, 2).contiguous().view(q_tokens, heads * embdv)

        def run_npu_fused():
            # MLA-aligned split: query/key = nope (128), query_rope/key_rope = rope (64), same tensors as packed into `q`/`k`.
            # Omit key_rope_antiquant_scale for fp16 NO_QUANT (CANN tiling requires null; op-plugin tests sometimes pass a tensor).
            torch_npu.npu_fused_infer_attention_score(
                fq,
                fk,
                fv,
                query_rope=query_rope,
                key_rope=key_rope,
                num_heads=heads,
                input_layout="BNSD",
                scale=attn_scale,
                pre_tokens=65535,
                next_tokens=65535,
                softmax_lse_flag=True,
            )

>>>>>>> Stashed changes
        # NOTE: Some large prefill shapes can trigger runtime/device errors on certain environments.
        # TODO: Revisit these skipped cases after kernel/tiling stability is improved.
        try:
            run_custom()
            torch.npu.synchronize()
        except RuntimeError as e:
            print(f"WARNING[{case['name']}]: skipped due to runtime error during correctness run: {e}")
            skipped_cases.append(case["name"])
            continue
        o_ref = run_ref()
        mean_abs_err = torch.mean(torch.abs(o.float() - o_ref.float())).item()
        max_abs_err = torch.max(torch.abs(o.float() - o_ref.float())).item()
        if max_abs_err > error_warn_threshold:
            print(
                f"WARNING[{case['name']}]: skipped due to large error "
                f"(mean_abs_err={mean_abs_err:.6f}, max_abs_err={max_abs_err:.6f}, "
                f"threshold={error_warn_threshold:.6f})"
            )
            skipped_cases.append(case["name"])
            continue

        try:
            custom_ms = benchmark_with_events(run_custom)
        except RuntimeError as e:
            print(f"WARNING[{case['name']}]: skipped due to runtime error during timing run: {e}")
            skipped_cases.append(case["name"])
            continue
        ref_ms = benchmark_with_events(run_ref)
<<<<<<< Updated upstream
        flops = estimate_flops(batch, heads, q_seq, kv_seq, embd, embdv)
        custom_tflops = flops / (custom_ms * 1e-3) / 1e12
        ref_tflops = flops / (ref_ms * 1e-3) / 1e12
        print(
            f"[{case['name']}] custom={custom_ms:.3f} ms ({custom_tflops:.4f} TFLOP/s), "
            f"torch_ref={ref_ms:.3f} ms ({ref_tflops:.4f} TFLOP/s), "
            f"mean_abs_err={mean_abs_err:.6f}, max_abs_err={max_abs_err:.6f}"
        )
        results.append(
            {
                "case": case["name"],
                "batch": batch,
                "heads": heads,
                "kv_heads": kv_heads,
                "q_seq": q_seq,
                "kv_seq": kv_seq,
                "embd": embd,
                "embdv": embdv,
                "block_dim": BLOCK_DIM,
                "custom_ms": custom_ms,
                "torch_ref_ms": ref_ms,
                "custom_tflops": custom_tflops,
                "torch_ref_tflops": ref_tflops,
                "mean_abs_err": mean_abs_err,
                "max_abs_err": max_abs_err,
            }
        )

    if results:
        csv_path = os.path.join(here, "benchmark_mla_prefill.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
=======

        flops_attn128 = estimate_flops_fused_infer(batch, heads, q_seq, kv_seq, attn_head_dim)
        flops_mla_hw = estimate_flops(batch, heads, q_seq, kv_seq, embd, embdv)
        custom_tflops_attn128 = flops_attn128 / (custom_ms * 1e-3) / 1e12
        torch_ref_tflops_attn128 = flops_attn128 / (ref_ms * 1e-3) / 1e12

        npu_fused_ms = None
        npu_fused_tflops_attn128 = None
        try:
            run_npu_fused()
            torch.npu.synchronize()
        except RuntimeError as e:
            print(f"WARNING[{case['name']}]: npu_fused_infer_attention_score warmup failed: {e}")
        else:
            try:
                npu_fused_ms = benchmark_with_events(run_npu_fused)
                npu_fused_tflops_attn128 = flops_attn128 / (npu_fused_ms * 1e-3) / 1e12
            except RuntimeError as e:
                print(f"WARNING[{case['name']}]: npu_fused_infer_attention_score timing failed: {e}")

        fused_ms_s = f"{npu_fused_ms:.3f}" if npu_fused_ms is not None else "n/a"
        fused_tf_s = f"{npu_fused_tflops_attn128:.4f}" if npu_fused_tflops_attn128 is not None else "n/a"
        print(
            f"[{case['name']}] mla_kernel={custom_ms:.3f} ms ({custom_tflops_attn128:.4f} TFLOP/s @ attn128 FLOPs), "
            f"sdpa={ref_ms:.3f} ms ({torch_ref_tflops_attn128:.4f}), "
            f"npu_fused_infer={fused_ms_s} ms ({fused_tf_s}), "
            f"flops_attn128={flops_attn128:.6e}, flops_mla_hw={flops_mla_hw:.6e}, "
            f"mean_abs_err={mean_abs_err:.6f}, max_abs_err={max_abs_err:.6f}"
        )
        row = {
            "case": case["name"],
            "batch": batch,
            "heads": heads,
            "kv_heads": kv_heads,
            "q_seq": q_seq,
            "kv_seq": kv_seq,
            "embd": embd,
            "embdv": embdv,
            "attn_head_dim": attn_head_dim,
            "block_dim": BLOCK_DIM,
            "flops_theory_attn128": flops_attn128,
            "flops_theory_mla_hw": flops_mla_hw,
            "mla_ms": custom_ms,
            "sdpa_ms": ref_ms,
            "npu_fused_infer_ms": npu_fused_ms if npu_fused_ms is not None else "",
            "mla_tflops_attn128": custom_tflops_attn128,
            "sdpa_tflops_attn128": torch_ref_tflops_attn128,
            "npu_fused_tflops_attn128": npu_fused_tflops_attn128 if npu_fused_tflops_attn128 is not None else "",
            "mean_abs_err": mean_abs_err,
            "max_abs_err": max_abs_err,
        }
        results.append(row)

    if results:
        csv_path = os.path.join(here, "benchmark_mla_prefill.csv")
        csv_fields = [
            "case",
            "batch",
            "heads",
            "kv_heads",
            "q_seq",
            "kv_seq",
            "embd",
            "embdv",
            "attn_head_dim",
            "block_dim",
            "flops_theory_attn128",
            "flops_theory_mla_hw",
            "mla_ms",
            "sdpa_ms",
            "npu_fused_infer_ms",
            "mla_tflops_attn128",
            "sdpa_tflops_attn128",
            "npu_fused_tflops_attn128",
            "mean_abs_err",
            "max_abs_err",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
>>>>>>> Stashed changes
            writer.writeheader()
            writer.writerows(results)
        print(f"wrote benchmark csv: {csv_path}")
    else:
        print("WARNING: no successful prefill benchmark cases; csv not written.")
    if skipped_cases:
        print(f"NOTE: skipped prefill benchmark cases due to runtime error: {skipped_cases}")


if __name__ == "__main__":
    run_smoke()
    run_benchmarks()
