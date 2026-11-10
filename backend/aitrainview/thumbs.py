"""Down-sample weight matrices into small heatmaps for the 3D atlas."""
from __future__ import annotations

import math

import numpy as np

from .st_reader import FLOAT_DTYPES, SafetensorsCheckpoint, to_float32

SIGNED_MODES = {"mean", "sample"}
MODES = ("rms", "mean", "sample", "absmax")


def as_2d(shape) -> tuple[int, int]:
    shape = [int(s) for s in shape]
    if len(shape) == 0:
        return 1, 1
    if len(shape) == 1:
        return 1, shape[0]
    if len(shape) == 2:
        return shape[0], shape[1]
    if len(shape) == 3:  # fused experts etc.
        return shape[0] * shape[1], shape[2]
    return shape[0], int(np.prod(shape[1:]))  # conv kernels


def _edges(n: int, g: int) -> np.ndarray:
    e = np.floor(np.arange(g + 1) * (n / g)).astype(np.int64)
    e[-1] = n
    return e


def thumbnail(ckpt: SafetensorsCheckpoint, name: str, size: int, mode: str = "rms",
              want_stats: bool = False) -> tuple[np.ndarray, dict | None]:
    """Block-reduce a tensor to at most size x size cells.

    mode: rms (block RMS), mean (block mean, signed), sample (one real weight per
    block, signed), absmax (block max |w|).
    """
    e = ckpt.entries[name]
    if e.dtype not in FLOAT_DTYPES:
        raise ValueError(f"{e.dtype} の可視化は未対応です")
    R, C = as_2d(e.shape)
    raw = ckpt.raw(name).reshape(R, C)
    gh, gw = min(R, size), min(C, size)
    r_edges, c_edges = _edges(R, gh), _edges(C, gw)
    c_count = np.diff(c_edges).astype(np.float32)
    c_center = (c_edges[:-1] + c_edges[1:]) // 2
    c_starts = c_edges[:-1]
    out = np.empty((gh, gw), np.float32)

    n = 0
    s1 = 0.0
    s2 = 0.0
    mn, mx = math.inf, -math.inf
    stride = max(1, (R * C) // 65536)
    samples = []

    for i in range(gh):
        r0, r1 = int(r_edges[i]), int(r_edges[i + 1])
        if mode == "sample" and not want_stats:
            # one real row per block is all that is shown: don't page in the whole matrix
            row = to_float32(raw[r0 + (r1 - r0) // 2, c_center], e.dtype)
            out[i] = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
            continue
        chunk = to_float32(raw[r0:r1], e.dtype)
        if not np.isfinite(chunk).all():
            chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
        rb = r1 - r0
        if mode == "sample":
            out[i] = chunk[rb // 2, c_center]
        elif mode == "mean":
            out[i] = np.add.reduceat(chunk.sum(0), c_starts) / (c_count * rb)
        elif mode == "absmax":
            out[i] = np.maximum.reduceat(np.abs(chunk).max(0), c_starts)
        else:  # rms
            sq = np.einsum("ij,ij->j", chunk, chunk)
            out[i] = np.sqrt(np.add.reduceat(sq, c_starts) / (c_count * rb))
        if want_stats:
            flat = chunk.ravel()
            n += flat.size
            s1 += float(flat.sum(dtype=np.float64))
            s2 += float(np.dot(flat.astype(np.float64), flat.astype(np.float64))) if flat.size < 4_000_000 \
                else float(np.dot(flat, flat))
            mn = min(mn, float(flat.min()))
            mx = max(mx, float(flat.max()))
            samples.append(flat[::stride].copy())

    stats = None
    if want_stats:
        mean = s1 / max(n, 1)
        var = max(s2 / max(n, 1) - mean * mean, 0.0)
        smp = np.concatenate(samples) if samples else np.zeros(1, np.float32)
        lo, hi = (float(smp.min()), float(smp.max())) if smp.size else (0.0, 0.0)
        if hi <= lo:
            hi = lo + 1e-6
        counts, edges = np.histogram(smp, bins=64, range=(lo, hi))
        stats = {
            "mean": mean, "std": math.sqrt(var), "min": mn, "max": mx,
            "absmax": max(abs(mn), abs(mx)), "rms": math.sqrt(s2 / max(n, 1)),
            "p01": float(np.percentile(smp, 1)), "p99": float(np.percentile(smp, 99)),
            "hist": counts.tolist(), "hist_lo": lo, "hist_hi": hi,
            "zero_frac": float(np.mean(np.abs(smp) < 1e-8)),
        }
    return out, stats


def quantize(thumb: np.ndarray, signed: bool) -> tuple[np.ndarray, float, float]:
    """Map a float heatmap to uint8. Signed: 127.5 is zero. Returns (u8, lo, hi)."""
    if signed:
        a = np.abs(thumb)
        scale = float(np.percentile(a, 99.5)) if a.size > 64 else float(a.max(initial=0.0))
        if not math.isfinite(scale) or scale <= 0:
            scale = float(a.max(initial=0.0)) or 1.0
        q = np.clip(thumb / scale, -1.0, 1.0) * 127.5 + 127.5
        return np.rint(q).astype(np.uint8), -scale, scale
    if thumb.size > 64:
        lo, hi = (float(v) for v in np.percentile(thumb, [0.5, 99.5]))
    else:
        lo, hi = float(thumb.min()), float(thumb.max())
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi - lo <= 1e-12:
        lo, hi = float(thumb.min()), float(thumb.max()) + 1e-6
    q = np.clip((thumb - lo) / (hi - lo), 0.0, 1.0) * 255.0
    return np.rint(q).astype(np.uint8), lo, hi


def resize_nearest(a: np.ndarray, size: int) -> np.ndarray:
    ri = (np.arange(size) * a.shape[0]) // size
    ci = (np.arange(size) * a.shape[1]) // size
    return a[ri][:, ci]
