"""
tensorshrink.vq
================

Additive Vector Quantization (AVQ) — a second, independent compression
algorithm alongside the scalar GOAP codec in `codec.py`.

Where GOAP (and bitsandbytes' NF4/int8) quantize every weight *independently*
onto a fixed 1-D grid, AVQ quantizes small *vectors* of weights (default 8
elements) against a codebook learned directly from the tensor's own values
(k-means, no calibration data, no activations). This exploits the fact that
neighboring weights in a row are correlated, so a shared library of typical
"weight-vector shapes" reconstructs the tensor far more accurately per bit
than any scalar grid can — the same principle behind published multi-bit
vector-quantization LLM compressors (e.g. AQLM, QuIP#), applied here with a
from-scratch, dependency-free implementation.

Algorithm, per weight matrix (rows, cols):

  1. Outlier side-channel (same scheme as codec.py): the top `outlier_frac`
     elements by magnitude are pulled out losslessly into a small sparse
     fp16 table and clipped in-place before anything else touches them.

  2. Per-row RMS normalization. Each output row is divided by its own mean
     absolute value, so a single shared codebook can represent rows that
     live at very different magnitude scales (attention vs MLP weights,
     etc.) without needing a separate codebook per row.

  3. Stage 1: reshape each row into vectors of `dim` consecutive columns
     and fit a K1-entry codebook (k-means, Lloyd's algorithm, GPU-batched)
     over all vectors in the tensor. Each vector is replaced by the index
     of its nearest codebook entry (log2(K1) bits / dim elements).

  4. Stage 2 (additive residual): the stage-1 reconstruction error
     (vector - codebook1[code]) is itself vector-quantized against a
     second, independently-trained K2-entry codebook, and added back in.
     This is what lets a tiny K1 (e.g. 256 entries, 8 bits/vector = 1
     bit/weight at dim=8) reach usable quality: the residual stage mops up
     what the coarse first codebook missed, similar in spirit to the
     residual-VQ literature (RVQ/AQLM) but both stages here are true
     multi-dimensional codebooks, not scalar grids.

  5. Adaptive tier escalation: reconstruction error is measured after each
     tier; if it exceeds `error_threshold`, quantization is retried at the
     next (larger, more accurate) tier in `TIER_CHAIN`.

Effective bit rate for the default tier (dim=8, K1=256, K2=16) is
8/8 + 4/8 = 1.5 bits/weight, before the (tiny, fixed) codebook and per-row
scale overhead — roughly a third of bitsandbytes' practical floor of ~4.5
bits/weight for NF4 (4-bit codes + double-quant metadata).

Works on CPU or CUDA tensors; k-means fitting runs in float32, on whatever
device the input tensor is on.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

import numpy as np
import torch

from .codec import _pack_nibbles, _unpack_nibbles, _EPS

DEFAULT_DIM = 8
DEFAULT_OUTLIER_FRAC = 0.003
DEFAULT_ERROR_THRESHOLD = 0.09
DEFAULT_KMEANS_ITERS = 12
DEFAULT_KMEANS_SAMPLE_CAP = 262_144   # max vectors used to *fit* centroids

# (dim, K1, K2) tiers, cheapest first. K2=0 disables the residual stage.
# Effective bits/weight ~= (log2(K1) + log2(K2 or 1) * (K2>0)) / dim.
TIER_CHAIN: List[Tuple[int, int, int]] = [
    (8, 256, 16),   # 1.5 bit/weight  (default, ~3x smaller than BNB NF4)
    (8, 256, 256),  # 2.0 bit/weight
    (4, 256, 16),   # 3.0 bit/weight
    (4, 256, 256),  # 4.0 bit/weight
]


@dataclasses.dataclass
class VQTensor:
    """In-memory representation of one AVQ-compressed weight tensor."""

    orig_shape: Tuple[int, ...]
    orig_dtype: str
    dim: int
    k1: int
    k2: int                     # 0 => stage 2 disabled
    rows: int
    cols: int
    padded_cols: int
    n_vecs: int                 # vectors per row = padded_cols // dim

    codebook1: np.ndarray       # float16 [k1, dim]
    codes1: np.ndarray          # uint8   [rows, n_vecs]
    codebook2: Optional[np.ndarray]   # float16 [k2, dim] or None
    codes2: Optional[np.ndarray]      # uint8 [rows, n_vecs] (k2<=16 packed as nibbles) or None
    codes2_packed: bool         # True if codes2 is nibble-packed (k2 <= 16)

    row_scale: np.ndarray       # float16 [rows]

    outlier_idx: np.ndarray     # int32
    outlier_val: np.ndarray     # float16

    def n_outliers(self) -> int:
        return int(self.outlier_idx.shape[0])

    def bits_per_weight(self) -> float:
        import math
        b = math.log2(self.k1)
        if self.k2:
            b += math.log2(self.k2)
        return b / self.dim

    def compressed_nbytes(self) -> int:
        total = (
            self.codebook1.nbytes
            + self.codes1.nbytes
            + self.row_scale.nbytes
            + self.outlier_idx.nbytes
            + self.outlier_val.nbytes
        )
        if self.codebook2 is not None:
            total += self.codebook2.nbytes + self.codes2.nbytes
        return total

    def orig_nbytes(self) -> int:
        itemsize = {"float16": 2, "bfloat16": 2, "float32": 4}.get(self.orig_dtype, 2)
        n = 1
        for d in self.orig_shape:
            n *= d
        return n * itemsize


# ---------------------------------------------------------------------------
# k-means (Lloyd's algorithm, GPU-batched, chunked to bound memory)
# ---------------------------------------------------------------------------

def _kmeans_fit(vectors: torch.Tensor, k: int, iters: int = DEFAULT_KMEANS_ITERS,
                 sample_cap: int = DEFAULT_KMEANS_SAMPLE_CAP,
                 chunk: int = 65_536) -> torch.Tensor:
    """Fit a k-entry codebook to `vectors` [N, dim] (float32). Returns
    centroids [k, dim] (float32). Deterministic-ish (seeded) but not
    order-sensitive in any way that matters for weight compression."""
    n, dim = vectors.shape
    device = vectors.device
    k = min(k, max(1, n))

    if n > sample_cap:
        idx = torch.randperm(n, device=device)[:sample_cap]
        fit_vecs = vectors[idx]
    else:
        fit_vecs = vectors

    gen = torch.Generator(device="cpu").manual_seed(0)
    init_idx = torch.randperm(fit_vecs.shape[0], generator=gen)[:k].to(device)
    centroids = fit_vecs[init_idx].clone()
    if centroids.shape[0] < k:
        # degenerate (fewer vectors than k): pad by repeating.
        pad = centroids[torch.randint(0, centroids.shape[0], (k - centroids.shape[0],))]
        centroids = torch.cat([centroids, pad], dim=0)

    m = fit_vecs.shape[0]
    for _ in range(iters):
        assign = torch.empty(m, dtype=torch.long, device=device)
        for start in range(0, m, chunk):
            end = min(start + chunk, m)
            block = fit_vecs[start:end]                      # [b, dim]
            d2 = torch.cdist(block, centroids)                # [b, k]
            assign[start:end] = d2.argmin(dim=1)

        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(k, device=device, dtype=torch.float32)
        new_centroids.index_add_(0, assign, fit_vecs)
        counts.index_add_(0, assign, torch.ones(m, device=device))
        empty = counts == 0
        counts = counts.clamp(min=1.0)
        new_centroids = new_centroids / counts.unsqueeze(1)
        if empty.any():
            # Re-seed dead clusters from random data vectors so we don't
            # waste codebook capacity on unused entries.
            n_empty = int(empty.sum().item())
            reseed = fit_vecs[torch.randint(0, m, (n_empty,), device=device)]
            new_centroids[empty] = reseed
        centroids = new_centroids

    return centroids


def _kmeans_assign(vectors: torch.Tensor, centroids: torch.Tensor,
                    chunk: int = 65_536) -> torch.Tensor:
    """Nearest-centroid assignment for every vector. Returns int64 codes [N]."""
    n = vectors.shape[0]
    device = vectors.device
    codes = torch.empty(n, dtype=torch.long, device=device)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        d2 = torch.cdist(vectors[start:end], centroids)
        codes[start:end] = d2.argmin(dim=1)
    return codes


# ---------------------------------------------------------------------------
# core encode / decode
# ---------------------------------------------------------------------------

def _to_2d(w: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    orig_shape = tuple(w.shape)
    if w.dim() == 1:
        raise ValueError("vq_quantize_tensor expects a tensor with ndim >= 2")
    rows = orig_shape[0]
    cols = 1
    for d in orig_shape[1:]:
        cols *= d
    return w.reshape(rows, cols), orig_shape


def _vq_encode_tier(w2d: torch.Tensor, rows: int, cols: int, orig_shape,
                     orig_dtype: str, dim: int, k1: int, k2: int,
                     outlier_frac: float) -> VQTensor:
    device = w2d.device
    work = w2d.detach().to(torch.float32, copy=True)

    # ---- outlier extraction (identical scheme to codec.py) --------------
    flat = work.reshape(-1)
    n_elem = flat.numel()
    k_out = max(1, int(n_elem * outlier_frac)) if outlier_frac > 0 else 0
    if k_out > 0 and k_out < n_elem:
        abs_flat = flat.abs()
        topk_vals, topk_idx = torch.topk(abs_flat, k_out)
        threshold = topk_vals.min()
        outlier_idx_t = topk_idx
        outlier_val_t = flat[topk_idx].clone()
        flat.clamp_(min=-threshold, max=threshold)
        del abs_flat, topk_vals
    else:
        outlier_idx_t = torch.empty(0, dtype=torch.int64, device=device)
        outlier_val_t = torch.empty(0, dtype=torch.float32, device=device)

    clipped = flat.reshape(rows, cols)

    # ---- per-row RMS normalization ---------------------------------------
    row_scale = clipped.abs().mean(dim=1).clamp(min=_EPS)     # [rows]
    normed = clipped / row_scale.unsqueeze(1)

    # ---- pad to a multiple of dim -----------------------------------------
    pad = (-cols) % dim
    if pad > 0:
        pad_block = normed[:, -1:].expand(rows, pad)
        padded = torch.cat([normed, pad_block], dim=1)
    else:
        padded = normed
    padded_cols = cols + pad
    n_vecs = padded_cols // dim

    vectors = padded.reshape(rows * n_vecs, dim)

    # ---- stage 1 codebook --------------------------------------------------
    cb1 = _kmeans_fit(vectors, k1)
    codes1 = _kmeans_assign(vectors, cb1)
    recon = cb1[codes1]

    cb2_np = None
    codes2_np = None
    if k2 and k2 > 1:
        residual = vectors - recon
        cb2 = _kmeans_fit(residual, k2)
        codes2 = _kmeans_assign(residual, cb2)
        recon = recon + cb2[codes2]
        cb2_np = cb2.to(torch.float16).cpu().numpy()
        codes2_2d = codes2.reshape(rows, n_vecs).to(torch.uint8)
        if k2 <= 16:
            pack_in = codes2_2d if n_vecs % 2 == 0 else torch.cat(
                [codes2_2d, codes2_2d[:, -1:]], dim=1)
            codes2_np = _pack_nibbles(pack_in).cpu().numpy()
        else:
            codes2_np = codes2_2d.cpu().numpy()

    return VQTensor(
        orig_shape=orig_shape,
        orig_dtype=orig_dtype,
        dim=dim,
        k1=k1,
        k2=k2 if (k2 and k2 > 1) else 0,
        rows=rows,
        cols=cols,
        padded_cols=padded_cols,
        n_vecs=n_vecs,
        codebook1=cb1.to(torch.float16).cpu().numpy(),
        codes1=codes1.reshape(rows, n_vecs).to(torch.uint8).cpu().numpy(),
        codebook2=cb2_np,
        codes2=codes2_np,
        codes2_packed=bool(k2 and k2 > 1 and k2 <= 16),
        row_scale=row_scale.to(torch.float16).cpu().numpy(),
        outlier_idx=outlier_idx_t.to(torch.int32).cpu().numpy(),
        outlier_val=outlier_val_t.to(torch.float16).cpu().numpy(),
    ), recon.reshape(rows, padded_cols), row_scale


def vq_quantize_tensor(
    w: torch.Tensor,
    dim: int = DEFAULT_DIM,
    k1: int = 256,
    k2: int = 16,
    outlier_frac: float = DEFAULT_OUTLIER_FRAC,
    adaptive: bool = True,
    error_threshold: float = DEFAULT_ERROR_THRESHOLD,
) -> VQTensor:
    """Additive Vector Quantization of a weight tensor (ndim >= 2).

    If `adaptive`, starts at the (dim, k1, k2) tier requested (or the first
    tier in TIER_CHAIN if it doesn't match one exactly) and escalates
    through TIER_CHAIN until relative reconstruction error is under
    `error_threshold` or the chain is exhausted.
    """
    orig_dtype = str(w.dtype).replace("torch.", "")
    w2d, orig_shape = _to_2d(w)
    rows, cols = w2d.shape

    tiers = TIER_CHAIN
    start = 0
    for i, (d, a, b) in enumerate(tiers):
        if (d, a, b) == (dim, k1, k2):
            start = i
            break
    else:
        tiers = [(dim, k1, k2)] + TIER_CHAIN
        start = 0

    w32 = w2d.detach().to(torch.float32)
    orig_norm = w32.norm().clamp(min=_EPS)

    qt = None
    for (d, a, b) in tiers[start:]:
        qt, recon, row_scale = _vq_encode_tier(w2d, rows, cols, orig_shape,
                                                orig_dtype, d, a, b, outlier_frac)
        recon_full = recon[:, :cols] * row_scale.unsqueeze(1)
        err = (w32 - recon_full).norm() / orig_norm
        if not adaptive or err.item() <= error_threshold:
            break
    return qt


def vq_dequantize_tensor(qt: VQTensor, device=None, dtype=None) -> torch.Tensor:
    device = device or "cpu"
    dtype = dtype or getattr(torch, qt.orig_dtype, torch.float16)

    cb1 = torch.from_numpy(qt.codebook1).to(device=device, dtype=torch.float32)
    codes1 = torch.from_numpy(qt.codes1).to(device=device).to(torch.long)
    recon = cb1[codes1.reshape(-1)]                         # [rows*n_vecs, dim]

    if qt.codebook2 is not None:
        cb2 = torch.from_numpy(qt.codebook2).to(device=device, dtype=torch.float32)
        if qt.codes2_packed:
            packed = torch.from_numpy(qt.codes2).to(device=device)
            codes2 = _unpack_nibbles(packed, qt.n_vecs).to(torch.long)
        else:
            codes2 = torch.from_numpy(qt.codes2).to(device=device).to(torch.long)
        recon = recon + cb2[codes2.reshape(-1)]

    recon = recon.reshape(qt.rows, qt.padded_cols)[:, :qt.cols]
    row_scale = torch.from_numpy(qt.row_scale).to(device=device, dtype=torch.float32)
    recon = recon * row_scale.unsqueeze(1)

    flat = recon.reshape(-1)
    if qt.n_outliers() > 0:
        outlier_idx = torch.from_numpy(qt.outlier_idx).to(device=device).to(torch.long)
        outlier_val = torch.from_numpy(qt.outlier_val).to(device=device, dtype=torch.float32)
        flat = flat.clone()
        flat[outlier_idx] = outlier_val

    out = flat.reshape(qt.rows, qt.cols).reshape(qt.orig_shape)
    return out.to(dtype)


def relative_error(w: torch.Tensor, qt: VQTensor) -> float:
    w2d, _ = _to_2d(w)
    recon = vq_dequantize_tensor(qt, device=w.device, dtype=torch.float32)
    recon2d, _ = _to_2d(recon)
    w32 = w2d.detach().to(torch.float32)
    return float((w32 - recon2d).norm() / w32.norm().clamp(min=_EPS))
