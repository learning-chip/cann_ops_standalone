# MLA Prefill (PTO-ISA cube matmul)

Port of `mla_prefill_cce` with **cube matmul** expressed through PTO-ISA (`pto::TMATMUL` / `TMATMUL_ACC` via `include/mla_pto_cube_ops.h`), following the same patterns as `pto-kernels/examples/jit_cpp` (e.g. `chunk_gdn/dynamic_bsnd/include/common.h` for tile typedefs and `TMATMUL` usage).

## What changed vs `mla_prefill_cce`

- **Cube `mmad_raw` → PTO**: `mla_pto::matmul_hh_f32<…>(l0c, l0a, l0b, m, n, k, init_c)` assigns `Tile`/`TileAcc` at the same L0 addresses and issues `TMATMUL` / `TMATMUL_ACC` (same `mad` lowering as before).
- **GM → L1 loads**: still use `copy_gm_to_cbuf` / `copy_gm_to_cbuf_multi_nd2nz_b16` (local `gm_to_l1_nd_nd_raw` / `gm_to_l1_nd_nz_raw` in `mla_prefill.cpp`) so tiling and **runtime ND strides** stay bit-identical to the original. An optional experimental `mla_pto::tload_gm_to_l1_nd2nz` is in the header for future migration to `pto::TLOAD`.
- **L1 → L0 / softmax / vec**: unchanged from the CCE tree (`../mla_prefill_cce` includes).
- **L0C → GM**: unchanged (`l0c_to_gm_nd_raw` + `copy_matrix_cc_to_gm`).

## Build and run

Set `PTO_ISA_ROOT` if PTO headers are not at `/workdir/pto-isa-master` (default in `compile.sh`).

```bash
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-8.5.1   # example
bash ./compile.sh
python ./test_mla_prefill.py
```

Expect smoke output similar to `mla_prefill_cce` (mean/max abs diff vs torch SDPA on the order of 1e-4).

## Dependencies

- CANN / bisheng (`--npu-arch=dav-2201`)
- PTO-ISA headers: `pto/pto-inst.hpp` (from `pto-isa-master` or your CANN SDK tree)
- Parent directory `../mla_prefill_cce` for `kernels/` and `mixkernels/` sources included by the kernel
