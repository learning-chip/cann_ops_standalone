/*
 * PTO-ISA cube-side helpers for MLA prefill (910B / DAV C220).
 * - matmul: pto::TMATMUL / TMATMUL_ACC (same lowering as mmad_raw → mad).
 * - Optional GM→L1 helpers (pto::TLOAD) are kept for experiments; the kernel uses
 *   Ascend C copy_gm_to_cbuf / copy_gm_to_cbuf_multi_nd2nz_b16 for loads (runtime strides).
 */
#ifndef MLA_PTO_CUBE_OPS_H
#define MLA_PTO_CUBE_OPS_H

#include <pto/pto-inst.hpp>

namespace mla_pto {

using namespace pto;

/// Half matmul into FP32 L0C via PTO (operand order matches mmad_raw: mTile, nTile, kPart).
template <int MaxM, int MaxN, int MaxK>
AICORE PTO_INLINE void matmul_hh_f32(uint64_t l0c_addr_u64, uint64_t l0a_addr_u64, uint64_t l0b_addr_u64, uint32_t m,
                                      uint32_t n, uint32_t k, bool init_c)
{
    using L0A = pto::Tile<TileType::Left, half, MaxM, MaxK, BLayout::RowMajor, pto::DYNAMIC, pto::DYNAMIC,
                          SLayout::RowMajor, 512, PadValue::Zero>;
    using L0B = pto::Tile<TileType::Right, half, MaxK, MaxN, BLayout::RowMajor, pto::DYNAMIC, pto::DYNAMIC,
                          SLayout::ColMajor, 512, PadValue::Zero>;
    using L0C = pto::TileAcc<float, MaxM, MaxN, pto::DYNAMIC, pto::DYNAMIC>;

    L0A l0a(static_cast<int>(m), static_cast<int>(k));
    L0B l0b(static_cast<int>(k), static_cast<int>(n));
    L0C l0c(static_cast<int>(m), static_cast<int>(n));

    TASSIGN(l0a, static_cast<int32_t>(l0a_addr_u64));
    TASSIGN(l0b, static_cast<int32_t>(l0b_addr_u64));
    TASSIGN(l0c, static_cast<int32_t>(l0c_addr_u64));

    if (init_c) {
        TMATMUL(l0c, l0a, l0b);
    } else {
        TMATMUL_ACC(l0c, l0a, l0b);
    }
}

/// Experimental: ND GM → NZ L1 via pto::TLOAD (same micro-op as copy_gm_to_cbuf_multi_nd2nz_b16).
template <int MaxRows, int MaxCols, typename T>
AICORE PTO_INLINE void tload_gm_to_l1_nd2nz(__gm__ T *src, int32_t l1_byte_addr, uint32_t n_rows, uint32_t d_cols,
                                            uint32_t gm_row_stride_elems)
{
    using TileNZ = Tile<TileType::Mat, T, MaxRows, MaxCols, BLayout::ColMajor, pto::DYNAMIC, pto::DYNAMIC,
                        SLayout::RowMajor, 512>;
    int vr = static_cast<int>(n_rows);
    int vc = static_cast<int>(d_cols);
    TileNZ dst_tile(vr, vc);
    TASSIGN(dst_tile, l1_byte_addr);

    Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC> dyn_shape;
    dyn_shape.shape[pto::GlobalTensorDim::DIM_3] = vr;
    dyn_shape.shape[pto::GlobalTensorDim::DIM_4] = vc;
    Stride<0, 0, 0, pto::DYNAMIC, 1> dyn_stride;
    dyn_stride.stride[pto::GlobalTensorDim::DIM_3] = static_cast<int64_t>(gm_row_stride_elems);

    GlobalTensor<T, Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>, Stride<0, 0, 0, pto::DYNAMIC, 1>, Layout::ND> g(
        src, dyn_shape, dyn_stride);
    TLOAD(dst_tile, g);
}

} // namespace mla_pto

#endif
