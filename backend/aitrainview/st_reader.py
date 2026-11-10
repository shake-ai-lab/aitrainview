"""Minimal safetensors reader that needs only numpy.

The header is parsed directly and tensor data is memory-mapped, so multi-GB
checkpoints can be inspected without loading them into RAM (and without torch).
"""
from __future__ import annotations

import glob
import json
import os
import struct
from dataclasses import dataclass

import numpy as np

# dtype name -> numpy storage dtype (bf16 / fp8 are stored as raw ints and decoded)
_STORAGE = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16, "BF16": np.uint16,
    "F8_E4M3": np.uint8, "F8_E5M2": np.uint8,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
    "U64": np.uint64, "U32": np.uint32, "U16": np.uint16, "U8": np.uint8, "BOOL": np.bool_,
}
FLOAT_DTYPES = {"F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2"}


def _fp8_table(exp_bits: int, man_bits: int, finite_only: bool) -> np.ndarray:
    bias = (1 << (exp_bits - 1)) - 1
    emax = (1 << exp_bits) - 1
    mmax = (1 << man_bits) - 1
    table = np.zeros(256, dtype=np.float32)
    for b in range(256):
        sign = -1.0 if b & 0x80 else 1.0
        e = (b >> man_bits) & emax
        m = b & mmax
        if finite_only:
            if e == emax and m == mmax:
                table[b] = np.nan
                continue
        elif e == emax:
            table[b] = sign * np.inf if m == 0 else np.nan
            continue
        if e == 0:
            v = (m / (1 << man_bits)) * 2.0 ** (1 - bias)
        else:
            v = (1 + m / (1 << man_bits)) * 2.0 ** (e - bias)
        table[b] = sign * v
    return table


_F8_E4M3 = _fp8_table(4, 3, True)
_F8_E5M2 = _fp8_table(5, 2, False)


def to_float32(raw: np.ndarray, dtype: str) -> np.ndarray:
    """Decode a raw storage array into float32 (copying)."""
    raw = np.asarray(raw)
    if dtype == "BF16":
        return (raw.astype(np.uint32) << 16).view(np.float32)
    if dtype == "F8_E4M3":
        return _F8_E4M3[raw]
    if dtype == "F8_E5M2":
        return _F8_E5M2[raw]
    return raw.astype(np.float32)


@dataclass
class TensorEntry:
    name: str
    dtype: str
    shape: tuple
    file: str
    offset: int  # absolute byte offset in the file
    nbytes: int

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= int(s)
        return n


class SafetensorsCheckpoint:
    """All *.safetensors shards of a HF-style model folder."""

    def __init__(self, model_dir: str):
        self.model_dir = os.path.abspath(model_dir)
        self.files = self._find_files()
        if not self.files:
            raise FileNotFoundError(
                f"{self.model_dir} に .safetensors ファイルが見つかりません"
                "（.bin / GGUF 形式は未対応です）"
            )
        self.entries: dict[str, TensorEntry] = {}
        self.metadata: dict = {}
        for f in self.files:
            self._read_header(f)
        self._maps: dict[str, np.memmap] = {}

    def _find_files(self) -> list[str]:
        index = os.path.join(self.model_dir, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index, encoding="utf-8") as fh:
                weight_map = json.load(fh).get("weight_map", {})
            files = sorted({os.path.join(self.model_dir, v) for v in weight_map.values()})
            missing = [os.path.basename(f) for f in files if not os.path.exists(f)]
            if missing:  # a half-copied / half-downloaded model would otherwise open with layers missing
                raise FileNotFoundError(
                    f"{self.model_dir} の重みファイルが {len(missing)} 個足りません: {', '.join(missing[:3])}"
                    + (" ほか" if len(missing) > 3 else "")
                )
            if files:
                return files
        return sorted(glob.glob(os.path.join(self.model_dir, "*.safetensors")))

    def _read_header(self, path: str) -> None:
        with open(path, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(n))
        base = 8 + n
        meta = header.pop("__metadata__", None)
        if isinstance(meta, dict):
            self.metadata.update(meta)
        for name, info in header.items():
            start, end = info["data_offsets"]
            self.entries[name] = TensorEntry(
                name, info["dtype"], tuple(info["shape"]), path, base + start, end - start
            )

    def fingerprint(self) -> str:
        parts = [self.model_dir]
        for f in self.files:
            st = os.stat(f)
            parts.append(f"{os.path.basename(f)}:{st.st_size}:{int(st.st_mtime)}")
        return "|".join(parts)

    def _map(self, path: str) -> np.memmap:
        mm = self._maps.get(path)
        if mm is None:
            mm = np.memmap(path, dtype=np.uint8, mode="r")
            self._maps[path] = mm
        return mm

    def raw(self, name: str) -> np.ndarray:
        """Zero-copy view of the tensor in its storage dtype."""
        e = self.entries[name]
        storage = _STORAGE.get(e.dtype)
        if storage is None:
            raise ValueError(f"未対応の dtype: {e.dtype}")
        buf = self._map(e.file)[e.offset:e.offset + e.nbytes]
        arr = buf.view(storage)
        return arr.reshape(e.shape) if e.shape else arr

    def close(self) -> None:
        # numpy closes a memmap when the last view of it is gone; closing it here
        # would break views that a running thumbnail job still holds
        self._maps.clear()
