"""
tensorshrink.container
=======================

A small, self-describing, streaming-friendly container format for holding a
whole quantized state_dict.

Why not just pickle a dict of QuantizedTensor, or reuse safetensors?
  - We want an *entropy-coded* on-disk size (safetensors stores raw bytes,
    no compression at all), so every segment is zstd-compressed.
  - We want a writer that can be fed tensors one at a time, in any order,
    without knowing the total tensor count or size up front, and that never
    needs to hold more than "the tensor currently being written" in memory.
    That's what lets `stream_quantize_safetensors_dir` compress a 24GB model
    while peak RSS stays close to the size of its single largest tensor,
    not the whole model.

Layout on disk (append-only, trailer-indexed — similar in spirit to how a
ZIP's central directory sits at the end of the file, which is exactly what
makes streamed writing possible without seeking backwards):

    [segment bytes for tensor 1] [segment bytes for tensor 2] ... [segment bytes for tensor N]
    [manifest JSON, utf-8]
    [8 bytes: little-endian uint64 = length of manifest JSON]
    [8 bytes: magic b"TSHRNK01"]

A reader opens the file, seeks to the last 16 bytes to find the manifest,
then can fetch any single tensor by seeking directly to its segment
offsets — it does not need to read the whole file, which matters for the
"decompress only the layer I'm about to run" streaming-inference path.
"""

from __future__ import annotations

import io
import json
import mmap
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import zstandard as zstd

from .codec import (
    QuantizedTensor,
    dequantize_tensor,
    quantize_tensor,
    quantize_tensor_adaptive,
    relative_error,
    LayerBitStats,
    budget_allocate_bits,
    estimate_resident_nbytes,
)

MAGIC = b"TSHRNK01"
# Level 19 recovers a bit more size but costs much more time to compress;
# ZSTD_LEVEL_MAX is available as an explicit opt-in (`zstd_level=ZSTD_LEVEL_MAX`).
ZSTD_LEVEL = 12
ZSTD_LEVEL_MAX = 19
ZSTD_THREADS = max(1, os.cpu_count() or 1)  # zstd chunks large buffers across cores automatically

# A QuantizedTensor is 5 numpy arrays (packed/scale/gmin/outlier_idx/
# outlier_val). v1 compressed each as its own zstd frame, which is wasteful
# on load (frame overhead paid 5x per tensor). v2 concatenates all 5 into one
# buffer (offsets/lengths in the JSON manifest) and compresses it as a single
# frame. v1 containers remain readable — ContainerReader picks the code path
# per-tensor based on which manifest key ("segments" vs "segments_v2") is present.
CONTAINER_FORMAT_VERSION = 2


def _decompress_into(mm: "mmap.mmap", offset: int, length: int) -> bytes:
    """Decompress a zstd frame stored at mm[offset:offset+length]. Reading
    through an mmap instead of file.seek()+file.read() avoids a syscall +
    an extra buffer copy per segment, and (unlike a plain file handle) an
    mmap is safe to read concurrently from multiple threads, which is what
    lets `ContainerReader.get_quantized_many` decompress several tensors'
    worth of segments in parallel (zstd's C decompressor releases the GIL)."""
    comp = mm[offset : offset + length]
    return zstd.ZstdDecompressor().decompress(comp)


def _compress(buf: bytes, level: int = ZSTD_LEVEL, threads: int = ZSTD_THREADS) -> bytes:
    return zstd.ZstdCompressor(level=level, threads=threads).compress(buf)


def _decompress(buf: bytes) -> bytes:
    return zstd.ZstdDecompressor().decompress(buf)


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------

def _encode_outliers(outlier_idx: np.ndarray, outlier_val: np.ndarray, cols: int):
    """Roadmap #6: `outlier_idx` is a flat `row*cols+col` index per outlier
    — effectively high-entropy (spread across the tensor's whole element
    range), so it compresses poorly. Splitting into (row, col) doesn't help
    on its own (same total bits), but `QuantLinear.__init__` already sorts
    outliers by row for its sync-free chunk-bounds lookup — sorted rows let
    the ROW field be stored as a *delta* from the previous outlier's row
    instead of an absolute value. Outlier density is high enough per row
    (`outlier_frac` default 0.3% of a row's `cols` elements) that most
    deltas are 0 or 1 in practice — a low-entropy stream zstd compresses far
    better than the original near-uniform flat index, even though the raw
    (pre-compression) byte count is larger. Measured on real weight-shaped
    tensors (not synthetic): -3% to -17% *compressed* bytes vs the flat
    index, growing with tensor size (largest win on the biggest layers,
    where the flat index's entropy is highest) — see the roadmap file's
    history for the measurement this was validated against before committing.

    Returns `(row_delta, col, val, dtype_name)`, all reordered together by
    the row-sort permutation (order doesn't matter for correctness — the
    reader scatters `(row, col) -> val` independent of array order — only
    that the three stay aligned to the same outlier at each position).
    `dtype_name` is `"uint16"` when both the largest row-delta and every
    column value fit (the common case for any realistic Linear layer), else
    `"uint32"` (a tensor with far more than 65535 rows, or more than 65535
    columns, is not a shape this codec's realistic use cases produce, but
    falling back instead of overflowing keeps this correct regardless).
    """
    n = outlier_idx.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.uint16), np.zeros(0, dtype=np.uint16), outlier_val, "uint16"
    idx64 = outlier_idx.astype(np.int64)
    row = idx64 // cols
    col = idx64 % cols
    perm = np.argsort(row, kind="stable")
    row_sorted = row[perm]
    col_sorted = col[perm]
    val_sorted = outlier_val[perm]
    row_delta = np.empty(n, dtype=np.int64)
    row_delta[0] = row_sorted[0]
    row_delta[1:] = row_sorted[1:] - row_sorted[:-1]

    fits16 = bool(row_delta.max() < 65536 and (cols - 1) < 65536)
    dt = np.uint16 if fits16 else np.uint32
    return row_delta.astype(dt), col_sorted.astype(dt), val_sorted, dt.__name__


def _decode_outliers(row_delta: np.ndarray, col: np.ndarray, cols: int) -> np.ndarray:
    """Inverse of `_encode_outliers`'s (row_delta, col) half — reconstructs
    the flat `outlier_idx` array `QuantizedTensor` expects everywhere else
    in the codebase (this encoding is purely a container-serialization
    detail; `QuantizedTensor.outlier_idx` itself is unchanged)."""
    if row_delta.shape[0] == 0:
        return np.zeros(0, dtype=np.int32)
    row = np.cumsum(row_delta.astype(np.int64))
    idx = row * cols + col.astype(np.int64)
    return idx.astype(np.int32)


def _build_quantized_entry_and_blob(
    qt: QuantizedTensor,
    rel_error: Optional[float] = None,
    escalated: Optional[bool] = None,
    force_fp32_compute: bool = False,
) -> Tuple[dict, bytes]:
    """Pure (no file I/O) half of `ContainerWriter.add_quantized`: builds the
    manifest entry (minus the file-offset fields, filled in once the caller
    knows where the compressed blob lands) and the raw, pre-compression blob.
    Factored out so `stream_quantize_safetensors_dir`'s multiprocess workers
    can do this same CPU-bound work in a separate process and hand back just
    the (already small) result."""
    entry = {
        "kind": "quant",
        "orig_shape": list(qt.orig_shape),
        "orig_dtype": qt.orig_dtype,
        "bits": qt.bits,
        "group_size": qt.group_size,
        "rows": qt.rows,
        "cols": qt.cols,
        "padded_cols": qt.padded_cols,
        "n_groups": qt.n_groups,
        "n_outliers": qt.n_outliers(),
        "rel_error": rel_error,
        "escalated": bool(escalated) if escalated is not None else None,
        "force_fp32_compute": bool(force_fp32_compute),
    }
    # v2: concatenate the arrays and zstd-compress once (see
    # CONTAINER_FORMAT_VERSION's docstring above) instead of one frame per
    # array. `parts` records byte offsets *within the decompressed blob*,
    # not file offsets.
    #
    # Roadmap #6: outliers are stored as (row_delta, col) instead of the
    # flat `outlier_idx` — see `_encode_outliers`'s docstring for why this
    # compresses smaller despite more raw bytes. `entry["outlier_encoding"]`
    # marks which scheme this tensor used so `_quantized_from_entry` can
    # still read older containers that wrote a plain `outlier_idx` part.
    row_delta, col, val_sorted, dt_name = _encode_outliers(qt.outlier_idx, qt.outlier_val, qt.cols)
    entry["outlier_encoding"] = "delta_row_v1"
    blob = bytearray()
    parts = {}
    segments = [
        ("packed", qt.packed),
        ("scale", qt.scale),
        ("gmin", qt.gmin),
        ("outlier_row_delta", row_delta),
        ("outlier_col", col),
        ("outlier_val", val_sorted),
    ]
    codebook = getattr(qt, 'codebook', None)
    if codebook is not None:
        segments.append(("codebook", codebook))
        entry["companded"] = True
    for seg_name, arr in segments:
        arr = np.ascontiguousarray(arr)
        b = arr.tobytes()
        parts[seg_name] = {
            "offset": len(blob),
            "length": len(b),
            "np_dtype": str(arr.dtype),
            "shape": list(arr.shape),
        }
        blob += b
    entry["_parts"] = parts  # consumed by add_precompressed, stripped before it lands in the manifest
    return entry, bytes(blob)


def _build_raw_entry_and_blob(tensor: torch.Tensor) -> Tuple[dict, bytes]:
    """Pure half of `ContainerWriter.add_raw` — see `_build_quantized_entry_and_blob`."""
    orig_torch_dtype = str(tensor.dtype).replace("torch.", "")
    t = tensor.detach().to("cpu").contiguous()
    if t.dtype == torch.bfloat16:
        # numpy has no bfloat16 type; bf16->fp32 is an exact, lossless
        # upcast (bf16 is literally the top 16 bits of fp32), so this
        # loses nothing and zstd still compresses the redundant zero
        # mantissa bits well. Original dtype is recorded so `get()`
        # casts back down to bf16 on the way out.
        t = t.to(torch.float32)
    arr = t.numpy()
    entry = {
        "kind": "raw",
        "orig_shape": list(tensor.shape),
        "orig_torch_dtype": orig_torch_dtype,
        "np_dtype": str(arr.dtype),
        "_data_shape": list(arr.shape),
    }
    return entry, arr.tobytes()


class ContainerWriter:
    def __init__(self, path):
        self.path = str(path)
        self._f = open(self.path, "wb")
        self._manifest = {"format": "tensorshrink-v1", "format_version": CONTAINER_FORMAT_VERSION, "tensors": {}}
        self._offset = 0

    def _write_segment(self, data: bytes) -> Tuple[int, int]:
        off = self._offset
        self._f.write(data)
        self._offset += len(data)
        return off, len(data)

    def add_precompressed(self, name: str, entry: dict, compressed_blob: bytes) -> None:
        """Write an already-compressed segment produced by
        `_build_quantized_entry_and_blob`/`_build_raw_entry_and_blob` (+
        `_compress`) — e.g. from a multiprocessing worker. Fills in the
        file-offset fields, which only the (single-process, sequentially
        appending) writer can know, and stores the finished manifest entry.
        """
        off, ln = self._write_segment(compressed_blob)
        if entry["kind"] == "quant":
            parts = entry.pop("_parts")
            entry["segments_v2"] = {"blob": {"offset": off, "length": ln}, "parts": parts}
        else:
            shape = entry.pop("_data_shape")
            entry["segments"] = {
                "data": {"offset": off, "length": ln, "np_dtype": entry["np_dtype"], "shape": shape}
            }
        self._manifest["tensors"][name] = entry

    def add_quantized(
        self,
        name: str,
        qt: QuantizedTensor,
        rel_error: Optional[float] = None,
        escalated: Optional[bool] = None,
        force_fp32_compute: bool = False,
        level: int = ZSTD_LEVEL,
    ) -> None:
        entry, blob = _build_quantized_entry_and_blob(qt, rel_error, escalated, force_fp32_compute)
        comp = _compress(blob, level=level)
        self.add_precompressed(name, entry, comp)

    def add_raw(self, name: str, tensor: torch.Tensor, level: int = ZSTD_LEVEL) -> None:
        entry, raw = _build_raw_entry_and_blob(tensor)
        comp = _compress(raw, level=level)
        self.add_precompressed(name, entry, comp)

    def close(self) -> None:
        manifest_bytes = json.dumps(self._manifest).encode("utf-8")
        self._f.write(manifest_bytes)
        self._f.write(struct.pack("<Q", len(manifest_bytes)))
        self._f.write(MAGIC)
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            self._f.close()


# ---------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------

class ContainerReader:
    def __init__(self, path):
        self.path = str(path)
        self._f = open(self.path, "rb")
        self._f.seek(-len(MAGIC), io.SEEK_END)
        magic = self._f.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError(f"{path} is not a tensorshrink container (bad magic footer)")
        self._f.seek(-(len(MAGIC) + 8), io.SEEK_END)
        (manifest_len,) = struct.unpack("<Q", self._f.read(8))
        self._f.seek(-(len(MAGIC) + 8 + manifest_len), io.SEEK_END)
        manifest_bytes = self._f.read(manifest_len)
        self.manifest = json.loads(manifest_bytes)
        # Read-only mmap over the whole file: segment reads become slices
        # into already-mapped pages instead of seek()+read() syscalls, and
        # (unlike the plain file handle above) an mmap can be sliced from
        # multiple threads concurrently without any locking, which is what
        # `get_quantized_many` relies on for parallel decompression. Falls
        # back to the file handle for the (rare) zero-byte-segment edge case
        # since mmap.mmap() rejects length=0 on some platforms.
        try:
            self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        except (ValueError, OSError):
            self._mm = None

    def names(self) -> List[str]:
        return list(self.manifest["tensors"].keys())

    def kind(self, name: str) -> str:
        return self.manifest["tensors"][name]["kind"]

    def _read_raw_segment(self, offset: int, length: int) -> bytes:
        if self._mm is not None:
            return _decompress_into(self._mm, offset, length)
        self._f.seek(offset)
        return _decompress(self._f.read(length))

    def _read_segment(self, seg: dict) -> np.ndarray:
        raw = self._read_raw_segment(seg["offset"], seg["length"])
        # np.frombuffer aliases the (immutable) decompressed bytes; copy so the
        # array is writable and safe to hand to torch.from_numpy downstream.
        arr = np.frombuffer(raw, dtype=np.dtype(seg["np_dtype"])).copy()
        return arr.reshape(seg["shape"])

    def _quantized_from_entry(self, entry: dict) -> QuantizedTensor:
        if "segments_v2" in entry:
            blob_seg = entry["segments_v2"]["blob"]
            raw = self._read_raw_segment(blob_seg["offset"], blob_seg["length"])
            parts = entry["segments_v2"]["parts"]

            def _part(pname: str) -> np.ndarray:
                m = parts[pname]
                dt = np.dtype(m["np_dtype"])
                count = m["length"] // dt.itemsize
                arr = np.frombuffer(raw, dtype=dt, count=count, offset=m["offset"]).copy()
                return arr.reshape(m["shape"])

            packed, scale, gmin = _part("packed"), _part("scale"), _part("gmin")
            codebook = _part("codebook") if entry.get("companded") else None
            if entry.get("outlier_encoding") == "delta_row_v1":
                # Roadmap #6: reconstruct the flat outlier_idx QuantizedTensor
                # expects from the (row_delta, col) encoding written by
                # _encode_outliers -- see that function's docstring.
                row_delta, col = _part("outlier_row_delta"), _part("outlier_col")
                outlier_idx = _decode_outliers(row_delta, col, entry["cols"])
                outlier_val = _part("outlier_val")
            else:
                # Older container: plain flat outlier_idx part, still fully
                # readable (this encoding upgrade is purely additive, same
                # policy as CONTAINER_FORMAT_VERSION's v1->v2 upgrade above).
                outlier_idx = _part("outlier_idx")
                outlier_val = _part("outlier_val")
        else:
            segs = entry["segments"]
            packed = self._read_segment(segs["packed"])
            scale = self._read_segment(segs["scale"])
            gmin = self._read_segment(segs["gmin"])
            outlier_idx = self._read_segment(segs["outlier_idx"])
            outlier_val = self._read_segment(segs["outlier_val"])
            codebook = None
        return QuantizedTensor(
            orig_shape=tuple(entry["orig_shape"]),
            orig_dtype=entry["orig_dtype"],
            bits=entry["bits"],
            group_size=entry["group_size"],
            rows=entry["rows"],
            cols=entry["cols"],
            padded_cols=entry["padded_cols"],
            n_groups=entry["n_groups"],
            packed=packed,
            scale=scale,
            gmin=gmin,
            outlier_idx=outlier_idx,
            outlier_val=outlier_val,
            codebook=codebook,
        )

    def get_quantized(self, name: str) -> QuantizedTensor:
        """Return the QuantizedTensor *without* dequantizing — this is what
        QuantLinear uses so weights stay packed (small) in host RAM / can be
        moved to GPU still packed, and are only expanded to full precision
        transiently inside the forward pass.
        """
        entry = self.manifest["tensors"][name]
        if entry["kind"] != "quant":
            raise ValueError(f"tensor {name!r} is stored raw, not quantized")
        return self._quantized_from_entry(entry)

    def get_quantized_many(self, names: Iterable[str], max_workers: Optional[int] = None) -> Dict[str, QuantizedTensor]:
        """Load several quantized tensors concurrently. zstd's decompressor
        (and the mmap slicing that feeds it) release the GIL, so a thread
        pool gives a real wall-clock win here despite Python's GIL — this is
        the loading-speed lever for whole-model loads (`load_quantized_module`
        uses this instead of calling `get_quantized` once per layer in a
        loop). Falls back to sequential reads with the file-handle path
        (`self._mm is None`) since that path isn't safe to share across
        threads the same way.
        """
        names = list(names)
        if not names:
            return {}
        if self._mm is None or len(names) == 1:
            return {n: self.get_quantized(n) for n in names}
        workers = max_workers or min(32, (os.cpu_count() or 4) * 2)
        out: Dict[str, QuantizedTensor] = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(self.get_quantized, n): n for n in names}
            for fut in futs:
                n = futs[fut]
                out[n] = fut.result()
        return out

    def get(self, name: str, device=None, dtype=None) -> torch.Tensor:
        """Return the fully materialized (dequantized, for quant entries)
        tensor — convenient for raw tensors (biases, norms) or debugging."""
        entry = self.manifest["tensors"][name]
        if entry["kind"] == "raw":
            arr = self._read_segment(entry["segments"]["data"])
            t = torch.from_numpy(arr.copy())
            orig_torch_dtype = entry.get("orig_torch_dtype")
            if orig_torch_dtype is not None:
                t = t.to(getattr(torch, orig_torch_dtype))
        else:
            qt = self.get_quantized(name)
            t = dequantize_tensor(qt)
        if device is not None:
            t = t.to(device)
        if dtype is not None:
            t = t.to(dtype)
        return t

    def stats(self) -> dict:
        """Summary of the container: how many tensors, bit-width mix, sizes."""
        total_orig = 0
        total_packed = 0
        bit_counts: Dict[str, int] = {}
        for name, entry in self.manifest["tensors"].items():
            shape = entry["orig_shape"]
            n = 1
            for d in shape:
                n *= d
            itemsize = {"float16": 2, "bfloat16": 2, "float32": 4}.get(entry.get("orig_dtype", "float16"), 2)
            total_orig += n * itemsize
            if entry["kind"] == "quant":
                bits = entry["bits"]
                bit_counts[str(bits)] = bit_counts.get(str(bits), 0) + 1
                if "segments_v2" in entry:
                    total_packed += entry["segments_v2"]["blob"]["length"]
                else:
                    for seg in entry["segments"].values():
                        total_packed += seg["length"]
            else:
                bit_counts["raw"] = bit_counts.get("raw", 0) + 1
                total_packed += entry["segments"]["data"]["length"]
        return {
            "n_tensors": len(self.manifest["tensors"]),
            "orig_bytes": total_orig,
            "container_bytes_for_tensor_data": total_packed,
            "file_bytes": os.path.getsize(self.path),
            "bit_width_counts": bit_counts,
        }

    def close(self):
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def save_container(state_dict: Dict[str, torch.Tensor], out_path, **quantize_kwargs) -> dict:
    """Convenience one-shot API: quantize every eligible tensor in an
    in-memory state_dict and write it to `out_path`. For large models
    prefer `stream_quantize_safetensors_dir`, which never materializes the
    whole state_dict in RAM at once.
    """
    min_numel = quantize_kwargs.pop("min_numel", 16384)
    skip_patterns = quantize_kwargs.pop(
        "skip_name_patterns", ("norm", "bias", "position", "shared", "relative_attention_bias")
    )
    # QuantLinear's forward path only implements 2/4/8-bit. 6-bit (RVQ)
    # round-trips fine through raw storage but needs a second buffer set that
    # QuantLinear doesn't build, so it's excluded by default here; pass
    # `allowed_bits=(2, 4, 6, 8)` explicitly for non-QuantLinear use cases.
    quantize_kwargs.setdefault("allowed_bits", (2, 4, 8))
    writer = ContainerWriter(out_path)
    stats = {"tensors": [], "orig_bytes": 0}
    for name, t in state_dict.items():
        orig_bytes = t.numel() * t.element_size()
        eligible = t.dim() >= 2 and t.numel() >= min_numel and not any(p in name.lower() for p in skip_patterns)
        if eligible:
            qt, err, escalated = quantize_tensor_adaptive(t, **quantize_kwargs)
            writer.add_quantized(name, qt, rel_error=err, escalated=escalated)
            stats["tensors"].append({"name": name, "bits": qt.bits, "rel_error": err, "orig_bytes": orig_bytes})
        else:
            writer.add_raw(name, t)
            stats["tensors"].append({"name": name, "bits": None, "rel_error": 0.0, "orig_bytes": orig_bytes})
        stats["orig_bytes"] += orig_bytes
    writer.close()
    stats["file_bytes"] = os.path.getsize(str(out_path))
    return stats


# ---------------------------------------------------------------------------
# streaming quantization straight from a HF-style safetensors checkpoint dir
# ---------------------------------------------------------------------------

def _discover_shards(model_dir: Path) -> Dict[str, List[str]]:
    """Return {shard_filename: [tensor_names...]} for a HF checkpoint dir,
    handling both the single-file and sharded-with-index.json layouts."""
    from safetensors import safe_open

    index_files = sorted(model_dir.glob("*.safetensors.index.json"))
    if index_files:
        idx = json.loads(index_files[0].read_text(encoding="utf-8"))
        weight_map = idx["weight_map"]
        shard_to_tensors: Dict[str, List[str]] = {}
        for tname, fname in weight_map.items():
            shard_to_tensors.setdefault(fname, []).append(tname)
        return shard_to_tensors

    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files found in {model_dir}")
    shard_to_tensors = {}
    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            shard_to_tensors[f.name] = list(sf.keys())
    return shard_to_tensors


def _quantize_one_tensor_worker(args: tuple) -> Tuple[str, dict, bytes, dict]:
    """Multiprocessing worker: quantize one tensor and zstd-compress it,
    returning only the finished blob + manifest entry. Appending bytes to the
    container file stays single-threaded in the main process
    (`ContainerWriter.add_precompressed`).

    Must be a module-level function, not a closure/lambda — Windows'
    `multiprocessing` uses `spawn`, which pickles by import path.
    """
    (shard_path, tname, group_size, outlier_frac, error_threshold, min_numel, skip_name_patterns, zstd_level,
     forced_bits) = args
    from safetensors import safe_open

    # Cap each worker to 1 thread: torch otherwise sizes its intra-op pool to
    # the whole machine, and N worker processes doing that fight over N cores
    # (and can crash with bad_alloc on Windows). Parallelism here is across
    # processes/tensors, not within one tensor's ops.
    torch.set_num_threads(1)

    with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
        t = sf.get_tensor(tname)
    orig_bytes = t.numel() * t.element_size()
    eligible = (
        t.dim() >= 2 and t.numel() >= min_numel and not any(p in tname.lower() for p in skip_name_patterns)
    )
    if eligible:
        if forced_bits is not None:
            # Budget-aware plan already decided this tensor's bit-width in an
            # earlier stats-only pass -- quantize once at that bit-width
            # instead of adaptive's own threshold-driven escalation logic.
            qt = quantize_tensor(t, bits=forced_bits, group_size=group_size, outlier_frac=outlier_frac)
            err = relative_error(t, qt)
            escalated = forced_bits == 8
        else:
            # allowed_bits=(2, 4, 8): see save_container's matching comment --
            # this container is normally reloaded straight into QuantLinear
            # via load_quantized_module, which supports these three (6-bit
            # RVQ excluded -- see codec.quantize_tensor_adaptive's docstring).
            qt, err, escalated = quantize_tensor_adaptive(
                t, group_size=group_size, outlier_frac=outlier_frac, error_threshold=error_threshold,
                allowed_bits=(2, 4, 8),
            )
        entry, blob = _build_quantized_entry_and_blob(qt, rel_error=err, escalated=escalated)
        # threads=1: this already runs inside one of N worker *processes* --
        # each additionally spinning up zstd's own N-way internal thread
        # pool is oversubscription (N^2 threads across N cores) and, worse,
        # multiplies zstd's per-thread compression-context memory by N per
        # process. See ZSTD_THREADS' module-level default, which is correct
        # for the single-process sequential path but wrong here.
        comp = _compress(blob, level=zstd_level, threads=1)
        info = dict(
            name=tname, shape=list(t.shape), bits=qt.bits, rel_error=err,
            orig_bytes=orig_bytes, packed_bytes=qt.compressed_nbytes(),
        )
    else:
        entry, raw = _build_raw_entry_and_blob(t)
        comp = _compress(raw, level=zstd_level, threads=1)
        info = dict(
            name=tname, shape=list(t.shape), bits=None, rel_error=0.0,
            orig_bytes=orig_bytes, packed_bytes=orig_bytes,
        )
    return tname, entry, comp, info


def stream_quantize_safetensors_dir(
    model_dir,
    out_path,
    group_size: int = 128,
    outlier_frac: float = 0.003,
    error_threshold: float = 0.06,
    min_numel: int = 16384,
    skip_name_patterns: Iterable[str] = ("norm", "bias", "position", "shared", "relative_attention_bias"),
    progress: bool = True,
    parallel: bool = True,
    n_workers: Optional[int] = None,
    zstd_level: int = ZSTD_LEVEL,
    vram_budget_mb: Optional[float] = None,
) -> dict:
    """Quantize an entire HF checkpoint directory (single-file or sharded
    safetensors) straight from disk into one tensorshrink container, one
    tensor at a time — the model's state_dict is never fully materialized in
    RAM at once, only the tensor currently being quantized plus its packed
    output. Peak RAM tracks roughly one open shard's size, not the whole
    checkpoint, since safetensors mmaps each shard.

    `parallel` (default True): quantize+compress tensors across a
    `multiprocessing.Pool` of `n_workers` (default `os.cpu_count()`)
    processes instead of a single loop — the quantize math is pure CPU/NumPy
    and scales close to linearly with cores. Only the file write itself
    stays single-threaded in the main process. Falls back to sequential when
    False or when there's only one eligible tensor. On Windows, callers
    using `parallel=True` from a script must guard the call with
    `if __name__ == "__main__":`.

    `zstd_level`: passed through to every tensor's compression step.

    `vram_budget_mb` (MiB): alternative to `error_threshold`-driven
    per-tensor escalation — see `quantize_module`'s parameter of the same
    name. A fixed per-tensor threshold has no notion of a model-wide byte
    budget. When set, this runs a cheap extra pass over every eligible
    tensor first (quantize at 4-bit and 8-bit, keep only the byte-cost/error
    numbers, discard the packed data) to build a `codec.budget_allocate_bits`
    plan, then quantizes each tensor once more at its assigned bit-width.
    This costs reading each tensor twice but keeps the "never materialize
    the whole model in RAM" property.
    """
    from safetensors import safe_open

    model_dir = Path(model_dir)
    shard_to_tensors = _discover_shards(model_dir)
    skip_name_patterns = tuple(skip_name_patterns)

    bit_plan: Dict[str, int] = {}
    if vram_budget_mb is not None:
        layer_stats: List[LayerBitStats] = []
        for fname, tensor_names in shard_to_tensors.items():
            shard_path = model_dir / fname
            with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
                for tname in tensor_names:
                    t = sf.get_tensor(tname)
                    eligible = (
                        t.dim() >= 2 and t.numel() >= min_numel
                        and not any(p in tname.lower() for p in skip_name_patterns)
                    )
                    if not eligible:
                        continue
                    qt4 = quantize_tensor(t, bits=4, group_size=group_size, outlier_frac=outlier_frac)
                    err4 = relative_error(t, qt4)
                    qt8 = quantize_tensor(t, bits=8, group_size=group_size, outlier_frac=outlier_frac)
                    err8 = relative_error(t, qt8)
                    layer_stats.append(
                        LayerBitStats(
                            tname, estimate_resident_nbytes(qt4), estimate_resident_nbytes(qt8), err4, err8
                        )
                    )
                    del t, qt4, qt8
        bit_plan = budget_allocate_bits(layer_stats, vram_budget_mb * 1024 * 1024)
        if progress:
            n8 = sum(1 for b in bit_plan.values() if b == 8)
            print(f"  budget-aware plan: {len(bit_plan) - n8} tensors at 4-bit, {n8} at 8-bit "
                  f"(budget {vram_budget_mb:.1f}MiB, {len(bit_plan)} eligible tensors)")

    # Flat (shard, tensor_name) task list, built up front from each shard's
    # tensor names only (no tensor data loaded yet) -- workers each open
    # their own safetensors handle and read just the one tensor they were
    # assigned, so this list itself stays tiny regardless of checkpoint size.
    tasks = [
        (str(model_dir / fname), tname, group_size, outlier_frac, error_threshold, min_numel,
         skip_name_patterns, zstd_level, bit_plan.get(tname))
        for fname, tensor_names in shard_to_tensors.items()
        for tname in tensor_names
    ]
    tensor_to_shard = {tname: fname for fname, names in shard_to_tensors.items() for tname in names}

    writer = ContainerWriter(out_path)
    per_tensor: List[dict] = []
    orig_bytes_total = 0
    packed_bytes_total = 0

    try:
        if parallel and len(tasks) > 1:
            import multiprocessing

            # Capped at 6, not the full core count: each worker is a full
            # torch-importing process (Windows uses `spawn`, no copy-on-write),
            # and returns diminish past ~6 from spawn overhead and disk I/O.
            # `n_workers` overrides this for machines with more headroom.
            workers = n_workers or min(os.cpu_count() or 1, 6)
            try:
                with multiprocessing.Pool(processes=workers) as pool:
                    for tname, entry, comp, info in pool.imap_unordered(_quantize_one_tensor_worker, tasks):
                        writer.add_precompressed(tname, entry, comp)
                        info["shard"] = tensor_to_shard[tname]
                        per_tensor.append(info)
                        orig_bytes_total += info["orig_bytes"]
                        packed_bytes_total += info["packed_bytes"]
                        if progress:
                            tag = f"{info['bits']}b" if info["bits"] else "raw"
                            print(
                                f"  [{tag:>4s}] {tname:65s} "
                                f"{info['orig_bytes']/1e6:9.2f}MB -> {info['packed_bytes']/1e6:9.2f}MB"
                                + (f"  err={info['rel_error']:.4f}" if info["bits"] else "")
                            )
            except Exception as exc:
                # Worker failures are usually environment issues (low RAM,
                # pagefile) rather than a bug tied to a specific tensor.
                # Parallelism is a speed optimization, not a correctness
                # dependency, so fall back to sequential rather than crash.
                if progress:
                    print(f"  parallel quantize failed ({exc!r}); falling back to sequential")
                writer.close()
                writer = ContainerWriter(out_path)
                per_tensor, orig_bytes_total, packed_bytes_total = [], 0, 0
                parallel = False

        if not parallel or len(tasks) <= 1:
            for fname, tensor_names in shard_to_tensors.items():
                shard_path = model_dir / fname
                with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
                    for tname in tensor_names:
                        t = sf.get_tensor(tname)
                        orig_bytes = t.numel() * t.element_size()
                        eligible = (
                            t.dim() >= 2
                            and t.numel() >= min_numel
                            and not any(p in tname.lower() for p in skip_name_patterns)
                        )
                        if eligible:
                            forced_bits = bit_plan.get(tname)
                            if forced_bits is not None:
                                qt = quantize_tensor(
                                    t, bits=forced_bits, group_size=group_size, outlier_frac=outlier_frac
                                )
                                err = relative_error(t, qt)
                                escalated = forced_bits == 8
                            else:
                                # allowed_bits=(2, 4, 8): see save_container's comment.
                                qt, err, escalated = quantize_tensor_adaptive(
                                    t,
                                    group_size=group_size,
                                    outlier_frac=outlier_frac,
                                    error_threshold=error_threshold,
                                    allowed_bits=(2, 4, 8),
                                )
                            writer.add_quantized(tname, qt, rel_error=err, escalated=escalated, level=zstd_level)
                            packed_bytes = qt.compressed_nbytes()
                            bits_used = qt.bits
                        else:
                            writer.add_raw(tname, t, level=zstd_level)
                            packed_bytes = orig_bytes
                            err = 0.0
                            bits_used = None

                        per_tensor.append(
                            {
                                "name": tname,
                                "shard": fname,
                                "shape": list(t.shape),
                                "bits": bits_used,
                                "rel_error": err,
                                "orig_bytes": orig_bytes,
                                "packed_bytes": packed_bytes,
                            }
                        )
                        orig_bytes_total += orig_bytes
                        packed_bytes_total += packed_bytes
                        if progress:
                            tag = f"{bits_used}b" if bits_used else "raw"
                            print(
                                f"  [{tag:>4s}] {tname:65s} "
                                f"{orig_bytes/1e6:9.2f}MB -> {packed_bytes/1e6:9.2f}MB"
                                + (f"  err={err:.4f}" if bits_used else "")
                            )
                        del t
    finally:
        writer.close()

    return {
        "n_tensors": len(per_tensor),
        "orig_bytes": orig_bytes_total,
        "packed_bytes_pre_entropy_coding": packed_bytes_total,
        "container_file_bytes": os.path.getsize(str(out_path)),
        "tensors": per_tensor,
    }
