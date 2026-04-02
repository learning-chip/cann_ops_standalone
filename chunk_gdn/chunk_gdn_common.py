"""Shared helpers for standalone chunk_gdn stage probe tests."""

from __future__ import annotations

import ctypes

import numpy as np
import torch

from test_chunk_gdn import ChunkGatedDeltaRuleTilingData, TCubeTiling


def as_ptr(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def check_close(
    name: str,
    actual: torch.Tensor,
    ref: torch.Tensor,
    tol: float = 1e-5,
    mean_tol: float | None = None,
) -> None:
    diff = (actual - ref).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    print(f"{name}: max_abs={max_abs:.6e} mean_abs={mean_abs:.6e}")
    assert max_abs <= tol, f"{name} mismatch max_abs={max_abs}"
    if mean_tol is not None:
        assert mean_abs <= mean_tol, f"{name} mean_abs too large: {mean_abs} (max_abs={max_abs})"


def default_matmul_tiling(ai_core_num: int, dim: int) -> TCubeTiling:
    """FP32 matmul tiling used by StageOneMT / StageTwoMT / StageThreeMT Init."""
    mm = TCubeTiling()
    mm.usedCoreNum = ai_core_num
    mm.M = dim
    mm.N = dim
    mm.Ka = dim
    mm.Kb = dim
    mm.singleCoreM = dim
    mm.singleCoreN = dim
    mm.singleCoreK = dim
    mm.baseM = dim
    mm.baseN = dim
    mm.baseK = dim
    mm.depthA1 = 1
    mm.depthB1 = 1
    mm.stepM = 1
    mm.stepN = 1
    mm.isBias = 0
    mm.stepKa = 1
    mm.stepKb = 1
    mm.dbL0C = 1
    mm.transLength = dim * dim * 4
    mm.shareMode = 0
    mm.shareL1Size = 0
    mm.shareL0CSize = dim * dim * 4
    mm.shareUbSize = 0
    mm.batchM = 1
    mm.batchN = 1
    mm.singleBatchM = 1
    mm.singleBatchN = 1
    return mm


def tiling_to_device(tiling: ChunkGatedDeltaRuleTilingData, device: str) -> torch.Tensor:
    tiling_bytes = bytes(tiling)
    return torch.from_numpy(np.frombuffer(tiling_bytes, dtype=np.uint8).copy()).to(device=device)


def ai_core_num_from_device() -> int:
    try:
        cube_core_num = int(getattr(torch.npu.get_device_properties("npu"), "cube_core_num", 24))
    except Exception:
        cube_core_num = 24
    return max(1, cube_core_num // 3)


def stage1_workspace_bytes(ai_core_num: int, chunk: int, dk: int, dv: int) -> int:
    float_size = 4
    return (
        ai_core_num * chunk * dk * float_size
        + ai_core_num * chunk * chunk * float_size
        + ai_core_num * chunk * dv * float_size
        + ai_core_num * chunk * chunk * float_size
        + ai_core_num * chunk * dk * float_size
        + ai_core_num * chunk * dk * float_size
    )


def stage3_workspace_bytes(ai_core_num: int, chunk: int) -> int:
    """Stage3 tmpGM starts at ws + ai_core_num*chunk^2*4; reserve 2x that region."""
    return 2 * ai_core_num * chunk * chunk * 4
