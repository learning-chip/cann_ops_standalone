"""Shared helpers for standalone chunk_gdn stage probe tests."""

from __future__ import annotations

import ctypes

import numpy as np
import torch


class TCubeTiling(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("usedCoreNum", ctypes.c_int32),
        ("M", ctypes.c_int32),
        ("N", ctypes.c_int32),
        ("Ka", ctypes.c_int32),
        ("Kb", ctypes.c_int32),
        ("singleCoreM", ctypes.c_int32),
        ("singleCoreN", ctypes.c_int32),
        ("singleCoreK", ctypes.c_int32),
        ("baseM", ctypes.c_int32),
        ("baseN", ctypes.c_int32),
        ("baseK", ctypes.c_int32),
        ("depthA1", ctypes.c_int32),
        ("depthB1", ctypes.c_int32),
        ("stepM", ctypes.c_int32),
        ("stepN", ctypes.c_int32),
        ("isBias", ctypes.c_int32),
        ("transLength", ctypes.c_int32),
        ("iterateOrder", ctypes.c_int32),
        ("shareMode", ctypes.c_int32),
        ("shareL1Size", ctypes.c_int32),
        ("shareL0CSize", ctypes.c_int32),
        ("shareUbSize", ctypes.c_int32),
        ("batchM", ctypes.c_int32),
        ("batchN", ctypes.c_int32),
        ("singleBatchM", ctypes.c_int32),
        ("singleBatchN", ctypes.c_int32),
        ("stepKa", ctypes.c_int32),
        ("stepKb", ctypes.c_int32),
        ("depthAL1CacheUB", ctypes.c_int32),
        ("depthBL1CacheUB", ctypes.c_int32),
        ("dbL0A", ctypes.c_int32),
        ("dbL0B", ctypes.c_int32),
        ("dbL0C", ctypes.c_int32),
        ("ALayoutInfoB", ctypes.c_int32),
        ("ALayoutInfoS", ctypes.c_int32),
        ("ALayoutInfoN", ctypes.c_int32),
        ("ALayoutInfoG", ctypes.c_int32),
        ("ALayoutInfoD", ctypes.c_int32),
        ("BLayoutInfoB", ctypes.c_int32),
        ("BLayoutInfoS", ctypes.c_int32),
        ("BLayoutInfoN", ctypes.c_int32),
        ("BLayoutInfoG", ctypes.c_int32),
        ("BLayoutInfoD", ctypes.c_int32),
        ("CLayoutInfoB", ctypes.c_int32),
        ("CLayoutInfoS1", ctypes.c_int32),
        ("CLayoutInfoN", ctypes.c_int32),
        ("CLayoutInfoG", ctypes.c_int32),
        ("CLayoutInfoS2", ctypes.c_int32),
        ("BatchNum", ctypes.c_int32),
        ("mxTypePara", ctypes.c_int32),
    ]


class ChunkGatedDeltaRuleTilingData(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("aiCoreNum", ctypes.c_int64),
        ("t", ctypes.c_int64),
        ("nk", ctypes.c_int64),
        ("dk", ctypes.c_int64),
        ("nv", ctypes.c_int64),
        ("dv", ctypes.c_int64),
        ("b", ctypes.c_int64),
        ("hasGamma", ctypes.c_int64),
        ("chunkSize", ctypes.c_int64),
        ("maxGroupLength", ctypes.c_int64),
        ("interWorkspaceSz", ctypes.c_int64),
        ("stageWorkspaceSz", ctypes.c_int64),
        ("stageOneParaNum", ctypes.c_int64),
        ("scale", ctypes.c_float),
        ("matmulTilingFp32", TCubeTiling),
    ]


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
        cube_core_num = int(
            getattr(torch.npu.get_device_properties(torch.npu.current_device()), "cube_core_num", 24)
        )
    except Exception:
        cube_core_num = 24
    return max(1, cube_core_num)


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
