"""
tensorshrink.codec
==================

The core, original compression algorithm used by this project. It is a
data-free (no calibration activations required), weight-only codec designed
specifically for the case of "I have a fixed VRAM/RAM budget and want a real
transformers/diffusers model to fit in it with minimal quality loss".

Algorithm ("GOAP" — Grouped Outlier-Aware Packing), applied independently to
every weight matrix:

  1. Row-group affine quantization.
     Each output row (a matrix is treated as [rows, cols], e.g. an
     nn.Linear.weight of shape [out_features, in_features]) is split into
     contiguous groups of `group_size` columns (default 128). Each group
     gets its own (min, scale) pair, so the quantizer adapts to local
     statistics instead of one global range for the whole tensor — this is
     what keeps 4-bit weights usable instead of collapsing everything to a
     handful of distinguishable values.

  2. Outlier clipping + sparse side-channel.
     Transformer/diffusion weights reliably contain a small number of
     high-magnitude outlier elements that would otherwise blow out a
     group's dynamic range and waste most of the 16 quantization levels on
     values nothing else is close to. Before quantizing, the top
     `outlier_frac` fraction of elements by magnitude (per tensor) are
     clipped down to the clipping threshold for the purpose of computing
     group ranges, and their *original* values are stored losslessly in a
     tiny sparse side table (index + fp16 value) that is scattered back in
     after dequantization. This is conceptually related to published
     outlier-aware schemes (e.g. SpQR, AWQ) but implemented independently
     here, fully vectorized, with no calibration data.

  3. Sub-byte packing.
     2-bit codes (4 levels) are packed four-per-byte. 4-bit codes (16
     levels) are packed two-per-byte. 8-bit codes are stored as-is. This
     alone is the direct RAM/VRAM win: a linear layer that cost 2
     bytes/element (fp16) now costs ~0.25 bytes/element at 2-bit or ~0.5
     bytes/element at 4-bit, before any entropy coding.

  4. Adaptive bit-width escalation.
     After a tensor is quantized at the lowest candidate bit-width, its
     reconstruction error (relative Frobenius norm) is measured. If it
     exceeds `error_threshold`, the tensor is automatically re-quantized
     at the next higher bit-width in the chain (2 -> 4 -> 6 -> 8). This
     makes the codec safe-by-default on sensitive tensors (embeddings,
     first/last projection layers, layernorm-adjacent weights) without the
     caller having to hand-tune a per-layer bit plan.

  5. Entropy coding (done in container.py, one layer up).
     The packed byte buffer is then zstd-compressed. Quantized code
     distributions cluster near the middle of the range for typical
     weight matrices, so this recovers a further ~5-20% on top of the
     packing step for free.

Extended features:

  - Double quantization (`double_quant=True`): per-group scale/gmin
    metadata (fp16) is itself quantized to uint8 with a single fp16
    meta-scale and meta-min per super-group (default 256 groups),
    saving ~50% on metadata overhead.

  - Smooth scaling (`smooth=True`): SmoothQuant-lite, weight-only.
    Per-column magnitude statistics are used to rescale the weight matrix
    before quantization, redistributing dynamic range for lower
    quantization error. The scaling factors are stored and applied in
    reverse during dequantization.

  - Residual Vector Quantization (bits=6): two-stage quantization where
    stage 1 is 4-bit and stage 2 quantizes the residual at 2-bit,
    yielding ~6 effective bits with quality much better than straight
    4-bit.

Everything here operates on plain torch tensors and only uses torch/numpy —
no dependency on bitsandbytes/GPTQ/AWQ/etc. It works identically on CPU or
CUDA tensors, which is what lets the same function be used both for
"quantize a checkpoint on disk" and "dequantize a layer's weights on the GPU
right before a matmul".
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

DEFAULT_GROUP_SIZE = 128
DEFAULT_OUTLIER_FRAC = 0.001     # 0.1% of elements kept as lossless fp16 side channel
DEFAULT_ERROR_THRESHOLD = 0.10   # relative Frobenius error that triggers escalation
DEFAULT_DQ_GROUP_SIZE = 256      # super-group size for double quantization of scale/gmin
DEFAULT_SMOOTH_ALPHA = 0.5       # exponent for per-column smooth scaling
_EPS = 1e-8


@dataclasses.dataclass
class QuantizedTensor:
    """In-memory representation of one compressed weight tensor.

    All heavy fields are raw numpy arrays (not yet byte-packed/compressed —
    that happens in container.py so this class stays reusable for the
    "keep it quantized in VRAM and dequantize on the fly" path too, where we
    never want to touch zstd on the hot path).
    """

    orig_shape: Tuple[int, ...]
    orig_dtype: str            # e.g. "float16"
    bits: int                  # 2, 4, 6 (RVQ), or 8
    group_size: int
    rows: int
    cols: int                  # original (unpadded) column count
    padded_cols: int
    n_groups: int

    packed: np.ndarray         # uint8, shape [rows, packed_cols]
    scale: np.ndarray          # float16, shape [rows, n_groups]
    gmin: np.ndarray           # float16, shape [rows, n_groups]

    outlier_idx: np.ndarray    # int32, flat indices into rows*cols
    outlier_val: np.ndarray    # float16

    # -- double quantization fields (None when not used) ------------------
    dq_scale_codes: Optional[np.ndarray] = None   # uint8, flattened codes for scale
    dq_scale_meta: Optional[np.ndarray] = None    # float16, [n_super_groups, 2] (meta_scale, meta_min)
    dq_gmin_codes: Optional[np.ndarray] = None    # uint8, flattened codes for gmin
    dq_gmin_meta: Optional[np.ndarray] = None     # float16, [n_super_groups, 2]

    # -- smooth scaling (None when not used) ------------------------------
    smooth_scale: Optional[np.ndarray] = None     # float16, [cols]

    # -- RVQ stage 2 fields (None when not bits=6) ------------------------
    stage2_packed: Optional[np.ndarray] = None    # uint8, packed 2-bit residual codes
    stage2_scale: Optional[np.ndarray] = None     # float16, [rows, n_groups]
    stage2_gmin: Optional[np.ndarray] = None      # float16, [rows, n_groups]

    # -- companded (Lloyd-Max) codebook (None => plain uniform grid) ------
    # float16, shape [1 << bits]. Reconstruction levels in the same
    # normalized 0..(1<<bits)-1 space that a uniform grid's integer codes
    # would occupy, but non-uniformly spaced to match this tensor's actual
    # value distribution (data-adaptive optimal MSE quantizer for that
    # distribution, a la Lloyd-Max / generalized Lloyd algorithm — this is
    # what NF4 approximates with a *fixed*, non-adaptive grid; here it's
    # refit per tensor). `codes` still index into this array the same way
    # they'd otherwise be used directly as the reconstruction value.
    codebook: Optional[np.ndarray] = None

    def n_outliers(self) -> int:
        return int(self.outlier_idx.shape[0])

    def compressed_nbytes(self) -> int:
        """Raw (pre-entropy-coding) byte footprint of the quantized form."""
        total = (
            self.packed.nbytes
            + self.outlier_idx.nbytes
            + self.outlier_val.nbytes
        )
        # Use double-quant metadata instead of full scale/gmin when present,
        # since the double-quant form is what would actually be serialized.
        if self.dq_scale_codes is not None:
            total += self.dq_scale_codes.nbytes + self.dq_scale_meta.nbytes
            total += self.dq_gmin_codes.nbytes + self.dq_gmin_meta.nbytes
        else:
            total += self.scale.nbytes + self.gmin.nbytes
        # Smooth scaling metadata
        if self.smooth_scale is not None:
            total += self.smooth_scale.nbytes
        # RVQ stage 2
        if self.stage2_packed is not None:
            total += self.stage2_packed.nbytes
            total += self.stage2_scale.nbytes
            total += self.stage2_gmin.nbytes
        # Companding codebook: tiny (<= 256 fp16 entries), but count it.
        if self.codebook is not None:
            total += self.codebook.nbytes
        return total

    def orig_nbytes(self) -> int:
        itemsize = {"float16": 2, "bfloat16": 2, "float32": 4}.get(self.orig_dtype, 2)
        n = 1
        for d in self.orig_shape:
            n *= d
        return n * itemsize


# ---------------------------------------------------------------------------
# bit packing helpers
# ---------------------------------------------------------------------------

def _pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """codes: uint8 tensor [..., N] with values in [0,15], N must be even.
    Returns uint8 tensor [..., N//2] with two 4-bit codes packed per byte.
    """
    assert codes.shape[-1] % 2 == 0
    low = codes[..., 0::2]
    high = codes[..., 1::2]
    return (low | (high << 4)).to(torch.uint8)


def _unpack_nibbles(packed: torch.Tensor, n: int) -> torch.Tensor:
    """Inverse of _pack_nibbles. Returns uint8 tensor [..., n]."""
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    out = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.uint8, device=packed.device)
    out[..., 0::2] = low
    out[..., 1::2] = high
    return out[..., :n]


def _pack_crumbs(codes: torch.Tensor) -> torch.Tensor:
    """codes: uint8 tensor [..., N] with values in [0,3], N must be divisible
    by 4. Returns uint8 tensor [..., N//4] with four 2-bit codes packed per
    byte (bits 0-1 = code 0, bits 2-3 = code 1, bits 4-5 = code 2, bits
    6-7 = code 3).
    """
    assert codes.shape[-1] % 4 == 0
    c0 = codes[..., 0::4]
    c1 = codes[..., 1::4]
    c2 = codes[..., 2::4]
    c3 = codes[..., 3::4]
    return (c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)).to(torch.uint8)


def _unpack_crumbs(packed: torch.Tensor, n: int) -> torch.Tensor:
    """Inverse of _pack_crumbs. Returns uint8 tensor [..., n]."""
    c0 = packed & 0x03
    c1 = (packed >> 2) & 0x03
    c2 = (packed >> 4) & 0x03
    c3 = (packed >> 6) & 0x03
    out = torch.empty(*packed.shape[:-1], packed.shape[-1] * 4, dtype=torch.uint8, device=packed.device)
    out[..., 0::4] = c0
    out[..., 1::4] = c1
    out[..., 2::4] = c2
    out[..., 3::4] = c3
    return out[..., :n]


# ---------------------------------------------------------------------------
# Lloyd-Max companded (non-uniform) codebook helpers
# ---------------------------------------------------------------------------
#
# A plain group-affine quantizer rounds each normalized value to the nearest
# *integer* in [0, levels] — a uniform grid. That's optimal only if values
# within a group are uniformly distributed, which real weight groups aren't
# (they cluster toward the group's center, roughly bell-shaped). NF4's fixed
# quantile grid exploits this by assuming a *standard* Gaussian; the Lloyd-Max
# algorithm (a 1-D k-means / generalized Lloyd algorithm) instead fits the
# provably MSE-optimal non-uniform grid directly to this tensor's own
# post-normalization value distribution, so it's never worse than NF4's
# fixed assumption and adapts automatically to whatever shape a given
# layer's weights actually have.

DEFAULT_COMPAND_ITERS = 12
DEFAULT_COMPAND_SAMPLE_CAP = 300_000

# ---------------------------------------------------------------------------
# Pre-computed normal-float codebooks (NF-style)
# ---------------------------------------------------------------------------
# BNB's NF4 advantage: a fixed codebook designed for normally-distributed
# values (which weight tensors approximately are). Instead of per-tensor
# Lloyd-Max fitting (our `companded` mode), these are pre-computed once
# from a standard Gaussian, stored as constants, and produce near-identical
# quality at zero fitting cost. The codebook values are the quantile midpoints
# of a standard normal distribution divided into `n_levels` equal-probability
# bins, then rescaled to [0, n_levels-1] range for compatibility with our
# existing group-affine dequant (code * scale + gmin).

def _compute_nf_codebook(n_levels: int) -> 'np.ndarray':
    """Pre-compute a normal-float codebook with `n_levels` entries.
    Each entry is the expected value (centroid) of the corresponding
    equal-probability bin of a standard normal."""
    from scipy.stats import norm as _norm
    import numpy as _np
    boundaries = _np.linspace(0, 1, n_levels + 1)
    # Centroid of each bin: E[X | X in (a, b)] for X ~ N(0,1)
    codebook = _np.zeros(n_levels, dtype=_np.float32)
    for i in range(n_levels):
        a, b = boundaries[i], boundaries[i + 1]
        # E[X | a < Phi(X) < b] = (phi(Phi^-1(a)) - phi(Phi^-1(b))) / (b - a)
        qa = _norm.ppf(max(a, 1e-10))
        qb = _norm.ppf(min(b, 1 - 1e-10))
        codebook[i] = (_norm.pdf(qa) - _norm.pdf(qb)) / max(b - a, 1e-10)
    # Rescale to [0, n_levels-1] range
    cmin, cmax = codebook.min(), codebook.max()
    if cmax > cmin:
        codebook = (codebook - cmin) / (cmax - cmin) * (n_levels - 1)
    return codebook.astype(_np.float32)

# Pre-computed codebooks for common bit-widths, lazily initialized
_NF_CODEBOOKS: Dict[int, 'np.ndarray'] = {}

def _get_nf_codebook(bits: int) -> 'np.ndarray':
    """Get or compute the NF codebook for `bits` bit-width."""
    if bits not in _NF_CODEBOOKS:
        n_levels = 1 << bits
        try:
            _NF_CODEBOOKS[bits] = _compute_nf_codebook(n_levels)
        except ImportError:
            # scipy not available, fall back to a simpler approximation
            # using torch's built-in erfinv
            import math
            codebook = np.zeros(n_levels, dtype=np.float32)
            for i in range(n_levels):
                # Midpoint of each equal-probability bin
                p = (i + 0.5) / n_levels
                # Approximate quantile via erfinv
                codebook[i] = math.sqrt(2) * torch.erfinv(torch.tensor(2 * p - 1)).item()
            cmin, cmax = codebook.min(), codebook.max()
            if cmax > cmin:
                codebook = (codebook - cmin) / (cmax - cmin) * (n_levels - 1)
            _NF_CODEBOOKS[bits] = codebook
    return _NF_CODEBOOKS[bits]


def _fit_lloyd_max_1d(values: torch.Tensor, n_levels: int,
                       iters: int = DEFAULT_COMPAND_ITERS,
                       sample_cap: int = DEFAULT_COMPAND_SAMPLE_CAP) -> torch.Tensor:
    """Fit `n_levels` 1-D reconstruction points to `values` (float32, any
    shape) via Lloyd's algorithm. Returns a sorted float32 tensor [n_levels].
    """
    device = values.device
    flat = values.reshape(-1)
    n = flat.numel()
    if n > sample_cap:
        idx = torch.randperm(n, device=device)[:sample_cap]
        fit_vals = flat[idx]
    else:
        fit_vals = flat
    m = fit_vals.shape[0]
    n_levels = max(1, min(n_levels, m))

    lo, hi = fit_vals.min(), fit_vals.max()
    cent = torch.linspace(lo.item(), hi.item(), n_levels, device=device)

    for _ in range(iters):
        cent_sorted, order = torch.sort(cent)
        if n_levels > 1:
            mids = (cent_sorted[:-1] + cent_sorted[1:]) / 2
            assign_sorted = torch.searchsorted(mids, fit_vals)
        else:
            assign_sorted = torch.zeros(m, dtype=torch.long, device=device)
        assign = order[assign_sorted]

        newc = torch.zeros(n_levels, device=device)
        cnt = torch.zeros(n_levels, device=device)
        newc.index_add_(0, assign, fit_vals)
        cnt.index_add_(0, assign, torch.ones(m, device=device))
        empty = cnt == 0
        cnt = cnt.clamp(min=1.0)
        newc = newc / cnt
        if empty.any():
            newc[empty] = cent[empty]
        cent = newc

    cent, _ = torch.sort(cent)
    return cent


def _companded_assign(values: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Nearest-codebook-entry assignment. `codebook` must be sorted
    ascending. Returns int64 codes with the same shape as `values`."""
    k = codebook.numel()
    if k <= 1:
        return torch.zeros_like(values, dtype=torch.long)
    mids = (codebook[:-1] + codebook[1:]) / 2
    flat = values.reshape(-1)
    codes = torch.searchsorted(mids, flat)
    return codes.clamp(0, k - 1).reshape(values.shape)


# ---------------------------------------------------------------------------
# double quantization helpers
# ---------------------------------------------------------------------------

def _double_quantize_np(values_np: np.ndarray, dq_group_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize a float16 numpy array to uint8 in super-groups of
    `dq_group_size`, with one (meta_scale, meta_min) pair per super-group.

    Returns (codes_uint8_1d, meta_fp16) where meta has shape
    [n_super_groups, 2].
    """
    flat = values_np.astype(np.float32).ravel()
    n = flat.shape[0]
    pad = (-n) % dq_group_size
    if pad > 0:
        flat = np.concatenate([flat, np.full(pad, flat[-1], dtype=np.float32)])
    n_super = flat.shape[0] // dq_group_size
    grouped = flat.reshape(n_super, dq_group_size)
    gmin_arr = grouped.min(axis=1)
    gmax_arr = grouped.max(axis=1)
    meta_scale = (gmax_arr - gmin_arr) / 255.0
    meta_scale = np.maximum(meta_scale, 1e-8)
    codes = np.round((grouped - gmin_arr[:, None]) / meta_scale[:, None])
    codes = codes.clip(0, 255).astype(np.uint8)
    meta = np.stack([meta_scale, gmin_arr], axis=1).astype(np.float16)
    return codes.ravel(), meta


def _double_dequantize_np(codes_np: np.ndarray, meta_np: np.ndarray,
                          orig_shape: Tuple[int, ...]) -> np.ndarray:
    """Reconstruct float16 values from double-quantized codes + meta
    (numpy path, used during quantization to store reconstructed values)."""
    n_super = meta_np.shape[0]
    dq_gs = codes_np.shape[0] // n_super
    codes = codes_np.astype(np.float32).reshape(n_super, dq_gs)
    meta_scale = meta_np[:, 0].astype(np.float32)
    meta_min = meta_np[:, 1].astype(np.float32)
    recon = codes * meta_scale[:, None] + meta_min[:, None]
    n_orig = 1
    for d in orig_shape:
        n_orig *= d
    return recon.ravel()[:n_orig].reshape(orig_shape).astype(np.float16)


def _double_dequantize_torch(codes_np: np.ndarray, meta_np: np.ndarray,
                             orig_shape: Tuple[int, ...],
                             device) -> torch.Tensor:
    """Reconstruct float32 torch tensor from double-quantized codes + meta
    (torch path, used during dequantization on the target device)."""
    n_super = meta_np.shape[0]
    dq_gs = codes_np.shape[0] // n_super
    codes = torch.from_numpy(codes_np).to(device).to(torch.float32).reshape(n_super, dq_gs)
    meta = torch.from_numpy(meta_np).to(device).to(torch.float32)
    meta_scale = meta[:, 0]
    meta_min = meta[:, 1]
    recon = codes * meta_scale.unsqueeze(1) + meta_min.unsqueeze(1)
    n_orig = 1
    for d in orig_shape:
        n_orig *= d
    return recon.reshape(-1)[:n_orig].reshape(orig_shape)


# ---------------------------------------------------------------------------
# core quantize / dequantize
# ---------------------------------------------------------------------------

def _to_2d(w: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    orig_shape = tuple(w.shape)
    if w.dim() == 1:
        raise ValueError("quantize_tensor expects a tensor with ndim >= 2")
    rows = orig_shape[0]
    cols = 1
    for d in orig_shape[1:]:
        cols *= d
    return w.reshape(rows, cols), orig_shape


def quantize_tensor(
    w: torch.Tensor,
    bits: int = 4,
    group_size: int = DEFAULT_GROUP_SIZE,
    outlier_frac: float = DEFAULT_OUTLIER_FRAC,
    double_quant: bool = False,
    dq_group_size: int = DEFAULT_DQ_GROUP_SIZE,
    smooth: bool = False,
    smooth_alpha: float = DEFAULT_SMOOTH_ALPHA,
    companded: bool = False,
    nf_quant: bool = False,
) -> QuantizedTensor:
    """Quantize a weight tensor (ndim >= 2) with the GOAP scheme.

    Works on CPU or CUDA input; internal compute happens in float32 for
    numerical stability of the min/max/round arithmetic, everything is cast
    back down before leaving the function.

    Supported bit widths:
      2  — 4 levels, ~8x compression vs fp16.
      4  — 16 levels, ~4x compression (the default).
      6  — Residual Vector Quantization: 4-bit first stage + 2-bit residual,
           ~6 effective bits with quality much better than straight 4-bit.
      8  — 256 levels, ~2x compression.

    `double_quant`: when True, the per-group scale and gmin metadata (both
    fp16) are themselves quantized to uint8 with a single fp16 meta-scale
    and meta-min per super-group of `dq_group_size` groups, saving ~50%
    on metadata overhead.

    `smooth`: when True, per-column magnitude statistics are used to
    rescale the weight matrix before quantization (SmoothQuant-lite,
    weight-only). For each column j, s_j = max(|W[:, j]|)^smooth_alpha;
    columns are divided by s_j before quantization, and s_j is stored so
    dequantization can reverse the scaling.

    `companded`: when True, replaces the uniform round-to-nearest-integer
    grid with a per-tensor non-uniform grid fit by Lloyd's algorithm to
    this tensor's actual (post-normalization) value distribution — the
    provably MSE-optimal quantizer for that distribution, so reconstruction
    error is never worse than the uniform grid's and is typically several
    percent lower for the bell-shaped distributions real weight groups
    have. Not supported together with bits=6 (RVQ).
    """
    if bits not in (2, 4, 6, 8):
        raise ValueError("bits must be 2, 4, 6, or 8")

    # RVQ (bits=6) is a two-stage scheme handled by a dedicated helper.
    if bits == 6:
        if companded:
            raise ValueError("companded=True is not supported with bits=6 (RVQ)")
        return _quantize_rvq(w, group_size=group_size, outlier_frac=outlier_frac,
                             double_quant=double_quant, dq_group_size=dq_group_size,
                             smooth=smooth, smooth_alpha=smooth_alpha)

    orig_dtype = str(w.dtype).replace("torch.", "")
    w2d, orig_shape = _to_2d(w)
    rows, cols = w2d.shape
    # `copy=True` is required, not cosmetic: `.to(torch.float32)` is a no-op
    # (returns the same tensor, no copy) when `w2d` is already float32, and
    # the in-place ops below (`flat.clamp_`, `grouped.sub_`/`div_`/etc.)
    # would then silently mutate the *caller's* original tensor in place --
    # a real, hard-to-spot corruption bug for any fp32 caller (e.g.
    # `_quantize_rvq`'s residual stage), not just a performance concern.
    work = w2d.detach().to(torch.float32, copy=True)

    # ---- 0. smooth scaling (optional) -----------------------------------
    smooth_scale_np = None
    if smooth:
        col_max = work.abs().amax(dim=0).clamp(min=_EPS)       # [cols]
        s_j = col_max.pow(smooth_alpha)
        work = work / s_j.unsqueeze(0)
        smooth_scale_np = s_j.to(torch.float16).cpu().numpy()

    # ---- 1. outlier extraction (global, per-tensor) -----------------
    flat = work.reshape(-1)
    n_elem = flat.numel()
    k = max(1, int(n_elem * outlier_frac)) if outlier_frac > 0 else 0

    if k > 0 and k < n_elem:
        abs_flat = flat.abs()
        topk_vals, topk_idx = torch.topk(abs_flat, k)
        threshold = topk_vals.min()
        outlier_idx_t = topk_idx
        outlier_val_t = flat[topk_idx].clone()
        # In-place: nothing downstream needs `flat`'s unclamped values
        # (`outlier_val_t` above already captured them), so clamping into
        # `flat`'s own storage instead of `flat.clamp(...)` (a fresh,
        # separate full-tensor-sized fp32 buffer) avoids materializing a
        # second copy of the whole tensor just to hold the clipped version.
        flat.clamp_(min=-threshold, max=threshold)
        clipped_flat = flat
        del abs_flat, topk_vals
    else:
        outlier_idx_t = torch.empty(0, dtype=torch.int64, device=w.device)
        outlier_val_t = torch.empty(0, dtype=torch.float32, device=w.device)
        clipped_flat = flat
    del flat, work  # both now aliases of (or replaced by) `clipped_flat`

    clipped = clipped_flat.reshape(rows, cols)

    # ---- 2. pad to a multiple of group_size --------------------------
    pad = (-cols) % group_size
    if pad > 0:
        pad_block = clipped[:, -1:].expand(rows, pad)
        padded = torch.cat([clipped, pad_block], dim=1)
    else:
        padded = clipped
    padded_cols = cols + pad
    n_groups = padded_cols // group_size

    grouped = padded.reshape(rows, n_groups, group_size)

    # ---- 3. per-group affine quant -----------------------------------
    gmin = grouped.amin(dim=-1)                      # [rows, n_groups]
    gmax = grouped.amax(dim=-1)
    levels = (1 << bits) - 1
    scale = (gmax - gmin) / levels
    scale = scale.clamp(min=_EPS)

    # In-place directly on `grouped` (a view we're about to discard anyway --
    # see the `del` below) instead of the out-of-place `-`/`/`/`round`/`clamp`
    # chain, each step of which would otherwise materialize its own fresh
    # full-tensor-sized fp32 buffer. Reusing `grouped`'s existing storage for
    # the whole affine transform means the only new allocation left is the
    # final (4x smaller) uint8 cast.
    grouped.sub_(gmin.unsqueeze(-1))
    grouped.div_(scale.unsqueeze(-1))
    codebook_np = None
    if nf_quant and not companded:
        # Normal-float quantization: pre-computed codebook, zero fitting cost.
        # The NF codebook is already in [0, levels] range, so we use
        # nearest-codebook-entry assignment instead of round-to-nearest-integer.
        nf_cb_np = _get_nf_codebook(bits)
        codebook_t = torch.from_numpy(nf_cb_np).to(grouped.device)
        codes_long = _companded_assign(grouped, codebook_t)
        codes = codes_long.to(torch.uint8)
        codebook_np = nf_cb_np
    elif companded:
        n_levels = 1 << bits
        codebook_t = _fit_lloyd_max_1d(grouped, n_levels)
        codes_long = _companded_assign(grouped, codebook_t)
        codes = codes_long.to(torch.uint8)
        codebook_np = codebook_t.to(torch.float16).cpu().numpy()
    else:
        grouped.round_()
        grouped.clamp_(0, levels)
        codes = grouped.to(torch.uint8)
    codes_flat = codes.reshape(rows, padded_cols)
    # `clipped`/`clipped_flat`/`padded`/`grouped` are all full-tensor-sized
    # fp32 views or copies that nothing below this line reads again
    # (`codes_flat` is derived from `codes`, not from these), but as named
    # locals they'd otherwise stay referenced through packing, scale/gmin
    # extraction, and QuantizedTensor construction. Several of these names
    # may already be aliases of each other (e.g. no padding needed, or no
    # outliers) -- deleting a name that's an alias of another already-deleted
    # one is harmless.
    del clipped, clipped_flat, padded, grouped

    # ---- 4. sub-byte packing ----------------------------------------
    if bits == 2:
        # _pack_crumbs needs column count divisible by 4. padded_cols is
        # only guaranteed divisible by 4 when group_size is divisible by 4
        # (true for the default 128). Pad with a dummy column otherwise.
        if padded_cols % 4 != 0:
            extra = (-padded_cols) % 4
            pack_input = torch.cat([codes_flat, codes_flat[:, -1:].expand(rows, extra)], dim=1)
        else:
            pack_input = codes_flat
        packed = _pack_crumbs(pack_input)
    elif bits == 4:
        # _pack_nibbles needs an even column count. padded_cols is only
        # guaranteed even when group_size is even (true for the default 128,
        # not guaranteed for an arbitrary caller-supplied group_size) — pad
        # one more dummy column when needed. This is independent of the
        # group-quantization padding above: dequantize_tensor always slices
        # back down using `padded_cols`, so this extra column never leaks
        # into any real output regardless of whether it existed.
        pack_input = codes_flat if padded_cols % 2 == 0 else torch.cat([codes_flat, codes_flat[:, -1:]], dim=1)
        packed = _pack_nibbles(pack_input)
    else:
        packed = codes_flat.contiguous()

    # ---- 5. double quantization (optional) ---------------------------
    scale_np = scale.to("cpu").to(torch.float16).numpy()
    gmin_np = gmin.to("cpu").to(torch.float16).numpy()
    dq_scale_codes = None
    dq_scale_meta = None
    dq_gmin_codes = None
    dq_gmin_meta = None

    if double_quant:
        dq_scale_codes, dq_scale_meta = _double_quantize_np(scale_np, dq_group_size)
        dq_gmin_codes, dq_gmin_meta = _double_quantize_np(gmin_np, dq_group_size)
        # Replace scale/gmin with reconstructed (approximate) values so
        # existing code that reads qt.scale / qt.gmin directly (e.g.
        # QuantLinear._compute_outlier_delta in modules.py) still gets
        # consistent values without needing to know about double quant.
        scale_np = _double_dequantize_np(dq_scale_codes, dq_scale_meta, scale_np.shape)
        gmin_np = _double_dequantize_np(dq_gmin_codes, dq_gmin_meta, gmin_np.shape)

    qt = QuantizedTensor(
        orig_shape=orig_shape,
        orig_dtype=orig_dtype,
        bits=bits,
        group_size=group_size,
        rows=rows,
        cols=cols,
        padded_cols=padded_cols,
        n_groups=n_groups,
        packed=packed.to("cpu").numpy(),
        scale=scale_np,
        gmin=gmin_np,
        outlier_idx=outlier_idx_t.to("cpu").to(torch.int32).numpy(),
        outlier_val=outlier_val_t.to("cpu").to(torch.float16).numpy(),
        dq_scale_codes=dq_scale_codes,
        dq_scale_meta=dq_scale_meta,
        dq_gmin_codes=dq_gmin_codes,
        dq_gmin_meta=dq_gmin_meta,
        smooth_scale=smooth_scale_np,
        codebook=codebook_np,
    )
    return qt


def _quantize_rvq(
    w: torch.Tensor,
    group_size: int = DEFAULT_GROUP_SIZE,
    outlier_frac: float = DEFAULT_OUTLIER_FRAC,
    double_quant: bool = False,
    dq_group_size: int = DEFAULT_DQ_GROUP_SIZE,
    smooth: bool = False,
    smooth_alpha: float = DEFAULT_SMOOTH_ALPHA,
) -> QuantizedTensor:
    """Residual Vector Quantization: stage 1 at 4-bit, stage 2 quantizes
    the residual at 2-bit, yielding ~6 effective bits with quality much
    better than straight 4-bit.

    The residual is computed in original (unsmoothed) space, so smooth
    scaling is only applied to stage 1. Stage 2 has no outlier extraction
    (stage 1 already captures them).
    """
    # Stage 1: full 4-bit quantize (with outliers, smooth, double_quant).
    qt4 = quantize_tensor(w, bits=4, group_size=group_size,
                          outlier_frac=outlier_frac,
                          double_quant=double_quant, dq_group_size=dq_group_size,
                          smooth=smooth, smooth_alpha=smooth_alpha)

    # Dequantize stage 1 to get the 4-bit approximation (in original space,
    # including outlier restoration and smooth un-scaling).
    deq4 = dequantize_tensor(qt4, device=w.device, dtype=torch.float32)

    # Residual = original - 4-bit approximation (in original space).
    w32 = w.detach().to(torch.float32)
    residual = w32 - deq4

    # Stage 2: 2-bit quantize the residual. No outliers (stage 1 already
    # captured them) and no smooth (residual has different structure).
    qt_res = quantize_tensor(residual, bits=2, group_size=group_size,
                             outlier_frac=0.0, double_quant=False, smooth=False)

    # Combine into a single QuantizedTensor with bits=6 marker.
    return QuantizedTensor(
        orig_shape=qt4.orig_shape,
        orig_dtype=qt4.orig_dtype,
        bits=6,
        group_size=qt4.group_size,
        rows=qt4.rows,
        cols=qt4.cols,
        padded_cols=qt4.padded_cols,
        n_groups=qt4.n_groups,
        packed=qt4.packed,
        scale=qt4.scale,
        gmin=qt4.gmin,
        outlier_idx=qt4.outlier_idx,
        outlier_val=qt4.outlier_val,
        dq_scale_codes=qt4.dq_scale_codes,
        dq_scale_meta=qt4.dq_scale_meta,
        dq_gmin_codes=qt4.dq_gmin_codes,
        dq_gmin_meta=qt4.dq_gmin_meta,
        smooth_scale=qt4.smooth_scale,
        stage2_packed=qt_res.packed,
        stage2_scale=qt_res.scale,
        stage2_gmin=qt_res.gmin,
    )


def _resolve_scale_gmin(qt: QuantizedTensor, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (scale, gmin) as float32 tensors on `device`, reconstructing
    from double-quant metadata when present."""
    dq_sc = getattr(qt, 'dq_scale_codes', None)
    dq_sm = getattr(qt, 'dq_scale_meta', None)
    dq_gc = getattr(qt, 'dq_gmin_codes', None)
    dq_gm = getattr(qt, 'dq_gmin_meta', None)

    if dq_sc is not None and dq_sm is not None:
        scale = _double_dequantize_torch(dq_sc, dq_sm, qt.scale.shape, device)
    else:
        scale = torch.from_numpy(qt.scale).to(device).to(torch.float32)

    if dq_gc is not None and dq_gm is not None:
        gmin = _double_dequantize_torch(dq_gc, dq_gm, qt.gmin.shape, device)
    else:
        gmin = torch.from_numpy(qt.gmin).to(device).to(torch.float32)

    return scale, gmin


def dequantize_tensor(qt: QuantizedTensor, device=None, dtype=None) -> torch.Tensor:
    """Reconstruct the (approximate, except for outliers which are exact)
    original tensor from a QuantizedTensor.
    """
    device = device or "cpu"
    out_dtype = dtype or getattr(torch, qt.orig_dtype, torch.float16)

    # RVQ (bits=6) is a two-stage scheme handled by a dedicated path.
    if qt.bits == 6:
        return _dequantize_rvq(qt, device, out_dtype)

    packed = torch.from_numpy(qt.packed).to(device)
    scale, gmin = _resolve_scale_gmin(qt, device)

    if qt.bits == 2:
        codes_flat = _unpack_crumbs(packed, qt.padded_cols)
    elif qt.bits == 4:
        codes_flat = _unpack_nibbles(packed, qt.padded_cols)
    else:
        codes_flat = packed

    codebook = getattr(qt, 'codebook', None)
    if codebook is not None:
        cb = torch.from_numpy(codebook).to(device).to(torch.float32)
        codes = cb[codes_flat.reshape(-1).to(torch.long)].reshape(
            qt.rows, qt.n_groups, qt.group_size)
    else:
        codes = codes_flat.reshape(qt.rows, qt.n_groups, qt.group_size).to(torch.float32)
    deq = codes * scale.unsqueeze(-1) + gmin.unsqueeze(-1)
    deq = deq.reshape(qt.rows, qt.padded_cols)[:, : qt.cols]
    flat = deq.reshape(-1).clone()

    if qt.n_outliers() > 0:
        idx = torch.from_numpy(qt.outlier_idx.astype(np.int64)).to(device)
        val = torch.from_numpy(qt.outlier_val).to(device).to(torch.float32)
        flat[idx] = val

    result = flat.reshape(qt.rows, qt.cols)

    # Apply smooth scale: multiply each column by s_j to undo the pre-quant
    # division. smooth_scale is [cols].
    smooth_s = getattr(qt, 'smooth_scale', None)
    if smooth_s is not None:
        s = torch.from_numpy(smooth_s).to(device).to(torch.float32)
        result = result * s.unsqueeze(0)

    return result.reshape(qt.orig_shape).to(out_dtype)


def _dequantize_rvq(qt: QuantizedTensor, device, out_dtype: torch.dtype) -> torch.Tensor:
    """Dequantize a bits=6 (RVQ) QuantizedTensor: reconstruct stage 1
    (4-bit + outliers + smooth) and stage 2 (2-bit residual), then sum."""
    # --- stage 1: 4-bit ---
    packed1 = torch.from_numpy(qt.packed).to(device)
    scale1, gmin1 = _resolve_scale_gmin(qt, device)

    codes1_flat = _unpack_nibbles(packed1, qt.padded_cols)
    codes1 = codes1_flat.reshape(qt.rows, qt.n_groups, qt.group_size).to(torch.float32)
    deq1 = codes1 * scale1.unsqueeze(-1) + gmin1.unsqueeze(-1)
    deq1 = deq1.reshape(qt.rows, qt.padded_cols)[:, : qt.cols]
    flat1 = deq1.reshape(-1).clone()

    if qt.n_outliers() > 0:
        idx = torch.from_numpy(qt.outlier_idx.astype(np.int64)).to(device)
        val = torch.from_numpy(qt.outlier_val).to(device).to(torch.float32)
        flat1[idx] = val

    result1 = flat1.reshape(qt.rows, qt.cols)

    # Un-smooth stage 1 (if smooth was used).
    smooth_s = getattr(qt, 'smooth_scale', None)
    if smooth_s is not None:
        s = torch.from_numpy(smooth_s).to(device).to(torch.float32)
        result1 = result1 * s.unsqueeze(0)

    # --- stage 2: 2-bit residual ---
    s2_packed = torch.from_numpy(qt.stage2_packed).to(device)
    s2_scale = torch.from_numpy(qt.stage2_scale).to(device).to(torch.float32)
    s2_gmin = torch.from_numpy(qt.stage2_gmin).to(device).to(torch.float32)

    codes2_flat = _unpack_crumbs(s2_packed, qt.padded_cols)
    codes2 = codes2_flat.reshape(qt.rows, qt.n_groups, qt.group_size).to(torch.float32)
    deq2 = codes2 * s2_scale.unsqueeze(-1) + s2_gmin.unsqueeze(-1)
    deq2 = deq2.reshape(qt.rows, qt.padded_cols)[:, : qt.cols]

    result = (result1 + deq2).reshape(qt.orig_shape).to(out_dtype)
    return result


def relative_error(w: torch.Tensor, qt: QuantizedTensor) -> float:
    """Relative Frobenius reconstruction error, used for adaptive bit-width
    selection and for benchmark reporting."""
    deq = dequantize_tensor(qt, device=w.device, dtype=torch.float32)
    w32 = w.detach().to(torch.float32)
    num = torch.linalg.norm((w32 - deq).reshape(-1))
    den = torch.linalg.norm(w32.reshape(-1)).clamp(min=_EPS)
    return float((num / den).item())


DEFAULT_ADAPTIVE_BIT_CHAIN: Tuple[int, ...] = (2, 4, 6, 8)


def quantize_tensor_adaptive(
    w: torch.Tensor,
    group_size: int = DEFAULT_GROUP_SIZE,
    outlier_frac: float = DEFAULT_OUTLIER_FRAC,
    error_threshold: float = DEFAULT_ERROR_THRESHOLD,
    force_bits: Optional[int] = None,
    double_quant: bool = False,
    dq_group_size: int = DEFAULT_DQ_GROUP_SIZE,
    smooth: bool = False,
    smooth_alpha: float = DEFAULT_SMOOTH_ALPHA,
    allowed_bits: Sequence[int] = DEFAULT_ADAPTIVE_BIT_CHAIN,
    companded: bool = False,
    nf_quant: bool = False,
) -> Tuple[QuantizedTensor, float, bool]:
    """Quantize with adaptive escalation through `allowed_bits` (ascending
    order; default `(2, 4, 6, 8)`): try the lowest bit-width first, escalate
    to the next one if the reconstruction error exceeds `error_threshold`,
    until the highest bit-width in the chain (always accepted regardless of
    its error). Returns (quantized, rel_error, escalated).

    `allowed_bits`: restrict the escalation chain to a subset of
    `(2, 4, 6, 8)` (any subset, any order it's given in is resorted
    ascending). `tensorshrink.modules.QuantLinear` only implements the
    live-forward-pass dequant path for bits 4 and 8 today — 2-bit and 6-bit
    (RVQ) are fully supported by `quantize_tensor`/`dequantize_tensor` here
    in `codec.py` (container-level / storage-only use, or direct
    `dequantize_tensor` calls) but callers that build a `QuantLinear` from
    the result (i.e. everything that goes through
    `tensorshrink.modules.quantize_module`/`load_quantized_module`/the HF
    `transformers` integration/`gpu_quantize`) pass `allowed_bits=(4, 8)`
    to stay within what `QuantLinear` can actually serve.

    `force_bits`, if given, skips the adaptive check and always uses that
    bit-width (used by callers that already know a tensor is sensitive,
    e.g. by name pattern).

    `double_quant`, `dq_group_size`, `smooth`, `smooth_alpha` are
    forwarded to every `quantize_tensor` call (see its docstring).
    """
    if companded and (force_bits == 6 or 6 in allowed_bits):
        raise ValueError("companded=True is not supported with bits=6 (RVQ)")

    _qkw = dict(group_size=group_size, outlier_frac=outlier_frac,
                double_quant=double_quant, dq_group_size=dq_group_size,
                smooth=smooth, smooth_alpha=smooth_alpha, companded=companded,
                nf_quant=nf_quant)

    if force_bits is not None:
        qt = quantize_tensor(w, bits=force_bits, **_qkw)
        err = relative_error(w, qt)
        return qt, err, False

    chain = sorted(set(allowed_bits))
    if not chain or any(b not in (2, 4, 6, 8) for b in chain):
        raise ValueError(f"allowed_bits must be a non-empty subset of (2, 4, 6, 8), got {allowed_bits!r}")

    last_qt, last_err = None, None
    for i, bits in enumerate(chain):
        qt = quantize_tensor(w, bits=bits, **_qkw)
        err = relative_error(w, qt)
        is_last = i == len(chain) - 1
        if err <= error_threshold or is_last:
            escalated = i > 0
            return qt, err, escalated
        last_qt, last_err = qt, err
    # Unreachable (the loop always returns on its last iteration), kept only
    # so a future refactor that changes the loop shape fails loudly instead
    # of silently returning None.
    return last_qt, last_err, True


def estimate_resident_nbytes(qt: QuantizedTensor) -> int:
    """Byte cost of `qt` once it's actually resident as a live `QuantLinear`
    (`tensorshrink/modules.py`), NOT the same number as
    `qt.compressed_nbytes()`. The two differ because `QuantLinear` stores
    outliers as `outlier_row`/`outlier_col` (two int32 buffers — split out
    for the sync-free chunk-bounds lookup in `QuantLinear.__init__`) plus a
    precomputed `outlier_delta` (one more float16 per outlier, for the
    fused Triton kernel path in `triton_kernels.py`) — NOT `outlier_val`,
    which isn't kept resident at all (every consumer reconstructs it from
    `outlier_delta` + the plain group-affine base value on demand — see
    `QuantLinear._outlier_base_val`) — instead of `compressed_nbytes()`'s
    compact single int32 `outlier_idx` + `outlier_val`. Used by
    `budget_allocate_bits` callers (`modules.quantize_module` and
    `container.stream_quantize_safetensors_dir`, both `vram_budget_mb=`)
    so a budget plan is sized against what actually ends up resident on
    GPU, not the smaller on-disk/transport form — using
    `compressed_nbytes()` here would silently under-budget.

    Free function (not a `QuantLinear` method) so `container.py` can use it
    without importing `modules.py` — `QuantLinear.estimate_resident_nbytes`
    is kept as a thin alias to this for backward-compatible call sites.
    """
    n_outliers = qt.n_outliers()
    total = (
        qt.packed.nbytes
        + n_outliers * 4  # outlier_row: int32
        + n_outliers * 4  # outlier_col: int32
        + n_outliers * 2  # outlier_delta: float16
    )
    # Metadata: use double-quant form when present.
    dq_sc = getattr(qt, 'dq_scale_codes', None)
    dq_gc = getattr(qt, 'dq_gmin_codes', None)
    if dq_sc is not None:
        total += qt.dq_scale_codes.nbytes + qt.dq_scale_meta.nbytes
        total += qt.dq_gmin_codes.nbytes + qt.dq_gmin_meta.nbytes
    else:
        total += qt.scale.nbytes + qt.gmin.nbytes
    # Smooth scale.
    smooth_s = getattr(qt, 'smooth_scale', None)
    if smooth_s is not None:
        total += smooth_s.nbytes
    # RVQ stage 2.
    s2p = getattr(qt, 'stage2_packed', None)
    if s2p is not None:
        total += s2p.nbytes + qt.stage2_scale.nbytes + qt.stage2_gmin.nbytes
    return total


@dataclasses.dataclass
class LayerBitStats:
    """Per-layer inputs to `budget_allocate_bits`: the byte cost and
    reconstruction error a layer would have at 4-bit vs 8-bit, computed once
    up front so the allocator can compare layers against each other instead
    of each layer deciding its own bit-width in isolation against a fixed
    error threshold, which is blind to the caller's actual VRAM/RAM budget."""

    name: str
    bytes4: int
    bytes8: int
    err4: float
    err8: float


def budget_allocate_bits(stats: Sequence[LayerBitStats], budget_bytes: float) -> Dict[str, int]:
    """Greedy knapsack bit-width plan: every layer starts at 4-bit (the
    minimum footprint), then layers are upgraded to 8-bit one at a time, in
    order of the best reconstruction-error reduction bought per extra byte
    spent, until `budget_bytes` is exhausted or every layer is at 8-bit.

    This is deliberately a greedy-by-ratio pass, not full 0/1-knapsack
    optimization (the roadmap's own framing) — for hundreds of layers with
    two choices each, greedy-by-ratio is a well-known good approximation
    and, more importantly, is a single sort instead of a DP table sized by
    byte-budget (which would be enormous at real VRAM-budget granularity).

    Returns `{layer_name: 4 or 8}` for every layer in `stats`. If even every
    layer at 4-bit already exceeds `budget_bytes`, no upgrades happen (every
    layer stays at 4-bit — the minimum this codec can offer, not a resolved
    "fits" guarantee for a budget that's simply too small for the model
    even at the smallest bit-width).
    """
    bits = {s.name: 4 for s in stats}
    total_bytes = sum(s.bytes4 for s in stats)
    remaining = budget_bytes - total_bytes

    candidates: List[Tuple[float, str, int]] = []
    for s in stats:
        extra_bytes = s.bytes8 - s.bytes4
        err_reduction = s.err4 - s.err8
        if extra_bytes <= 0 or err_reduction <= 0:
            continue  # 8-bit isn't strictly better here (shouldn't normally happen) -- never worth "spending" on
        candidates.append((err_reduction / extra_bytes, s.name, extra_bytes))
    candidates.sort(key=lambda c: c[0], reverse=True)  # best error-reduction-per-byte first

    for _, name, extra_bytes in candidates:
        if extra_bytes <= remaining:
            bits[name] = 8
            remaining -= extra_bytes

    return bits
