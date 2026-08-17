"""
tensorshrink.modules
=====================

The live, in-VRAM half of the codec: `QuantLinear` is a drop-in replacement
for `torch.nn.Linear` that stores its weight matrix packed (4-bit or 8-bit,
grouped, with a sparse fp16 outlier side-channel — see codec.py) instead of
as a dense fp16/fp32 tensor, and dequantizes on the fly inside `forward()`.

Because the packed representation is what's registered as the module's
buffers, calling `model.to("cuda")` moves only the small packed buffers to
the GPU — the model's *steady-state* VRAM footprint for every quantized
layer drops by ~4x (4-bit) or ~2x (8-bit) versus fp16, before even counting
the entropy-coding savings that only apply to the on-disk container.

`quantize_module()` walks a real `transformers`/`diffusers` model
(any `torch.nn.Module`) and swaps its `nn.Linear` layers for `QuantLinear`
in place, so the exact same model object you got from
`AutoModel.from_pretrained(...)` can be quantized and used for inference
with no other code changes.
"""

from __future__ import annotations

import os
import warnings
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .codec import (
    QuantizedTensor,
    quantize_tensor,
    quantize_tensor_adaptive,
    relative_error,
    _unpack_nibbles,
    _unpack_crumbs,
    LayerBitStats,
    budget_allocate_bits,
    estimate_resident_nbytes as _estimate_resident_nbytes,
    DEFAULT_DQ_GROUP_SIZE,
)
from .triton_kernels import (
    fused_quantized_linear,
    apply_outlier_correction as _triton_apply_outlier_correction,
    dequantize_weight as _triton_dequantize_weight,
    TRITON_AVAILABLE,
    SUPPORTED_GROUP_SIZE,
    _SUPPORTED_GROUP_SIZES,
    SUPPORTED_BITS as _TRITON_SUPPORTED_BITS,
)

# NOTE: these match against *module* dotted names (e.g. "transformer.h.0.attn"),
# not parameter names — biases never need to be listed here because QuantLinear
# only ever touches `.weight`; a Linear's bias is always kept at full precision
# automatically (see QuantLinear.__init__).
DEFAULT_SKIP_PATTERNS = ("norm", "ln_", "embed", "lm_head", "shared", "position")
DEFAULT_MIN_NUMEL = 16384
DEFAULT_MAX_DEQUANT_ROWS = 4096  # caps the transient fp16 buffer size during forward()

# T5 (and derivatives, e.g. FLUX's text_encoder_2) is sensitive to quantizing
# the FFN down-projection (`DenseReluDense.wo`): the error compounds through
# the residual stream across layers. HF special-cases this too (`wo` is
# forced to fp32 in modeling_t5.py). Not in DEFAULT_SKIP_PATTERNS since it's
# architecture-specific; opt in explicitly for T5 models:
#     quantize_module(model, skip_name_patterns=DEFAULT_SKIP_PATTERNS + T5_EXTRA_SKIP_PATTERNS)
T5_EXTRA_SKIP_PATTERNS = (".wo",)

# force_fp32_compute does NOT fix T5's `wo` sensitivity (the error comes from
# quantizing the weight itself, not the compute dtype) — use
# T5_EXTRA_SKIP_PATTERNS instead. Kept only as a documented non-fix.
T5_FORCE_FP32_PATTERNS_DOES_NOT_FIX_T5 = (".wo",)


def suggest_max_dequant_rows(
    out_features: int,
    in_features: int,
    itemsize: int = 4,
    device: Optional[str] = None,
    vram_frac: float = 0.02,
    min_rows: int = 128,
    max_rows: int = 65536,
) -> int:
    """Pick a `max_dequant_rows` chunk size from the *actual* free VRAM
    instead of the fixed `DEFAULT_MAX_DEQUANT_ROWS=4096` guess. That default
    is a reasonable one-size-fits-all value, but it's blind to both the
    layer's own shape and how much headroom is currently free on the card:
    a narrow layer (small `in_features`) can safely dequantize far more than
    4096 rows at once inside the same transient-memory budget (fewer,
    larger matmul calls -> less Python/kernel-launch overhead per token
    during generation), while a very wide layer on a nearly-full card needs
    a smaller chunk than the default to avoid OOM. Caps the transient
    dequant buffer to `vram_frac` (default 2%) of currently-free device
    memory. Falls back to `DEFAULT_MAX_DEQUANT_ROWS` when there's no CUDA
    device to query (CPU inference, or `device` not given and no GPU
    present) since there's no free-memory signal to auto-tune from there.
    """
    free_bytes = None
    dev = device or ("cuda" if torch.cuda.is_available() else None)
    if dev is not None and str(dev).startswith("cuda") and torch.cuda.is_available():
        try:
            free_bytes, _ = torch.cuda.mem_get_info(dev)
        except Exception:
            free_bytes = None
    if free_bytes is None:
        return DEFAULT_MAX_DEQUANT_ROWS
    budget = free_bytes * vram_frac
    per_row_bytes = max(1, in_features * itemsize)
    rows = int(budget // per_row_bytes)
    rows = max(min_rows, min(rows, max_rows, out_features))
    return rows


class WeightCache:
    """Byte-budgeted, LRU-evicted cache of fully-dequantized weights, shared
    across every `QuantLinear` in a model.

    `QuantLinear.forward()` normally re-dequantizes its weight every call to
    protect the VRAM budget, which is pure overhead in a generation loop
    where the same weight is reused every token. Since packed weights are
    already ~2-4x smaller than fp16, there's usually VRAM headroom left to
    cache some dequantized weights and skip that work on a hit. Opt-in and
    zero-cost when unused.

    Also supports an optional host-RAM tier (`host_budget_bytes`) for models
    with ~zero VRAM headroom (e.g. FLUX-scale) that still reuse the same
    weights across many steps: a host-tier hit is a host->device copy
    instead of a full unpack+reconstruct, cheaper than a full dequant though
    not as cheap as a VRAM hit. `get()`/`put()` handle both tiers
    transparently. Pairs well with `streaming.StreamingQuantModel`.

    **Thrashing failure mode.** A whole-model forward pass touches every
    layer once per call in a fixed order — a cyclic full-scan. Plain LRU
    handles this badly once the working set exceeds the cache budget: by the
    time the scan wraps back around to layer 1, later layers have already
    evicted it, so every layer misses on every call (0% hit rate, worse than
    no cache — measured on a real LLM: generation dropped to ~1 token/s).

    `admission_policy="static_fill"` fixes this for that access pattern:
    once a tier is full, new keys are simply not inserted (no eviction).
    Whichever layers fill the cache first stay cached permanently — a
    stable, high hit rate instead of 0%. Default remains `"lru"` for
    backward compatibility with direct `WeightCache(...)` callers;
    `quantize_module`, `load_quantized_module`, the HF integration, and
    `gpu_quantize.quantize_module_gpu` all use `"static_fill"` for the
    caches they build automatically.
    """

    def __init__(self, budget_bytes: int, host_budget_bytes: int = 0, pin_host_memory: bool = True,
                 admission_policy: str = "lru"):
        if admission_policy not in ("lru", "static_fill"):
            raise ValueError(f"admission_policy must be 'lru' or 'static_fill', got {admission_policy!r}")
        self.budget_bytes = max(0, int(budget_bytes))
        self.used_bytes = 0
        self._store: "OrderedDict[int, torch.Tensor]" = OrderedDict()  # VRAM tier

        self.host_budget_bytes = max(0, int(host_budget_bytes))
        self.host_used_bytes = 0
        self._host_store: "OrderedDict[int, torch.Tensor]" = OrderedDict()  # host RAM tier
        self._pin_host_memory = pin_host_memory
        self.admission_policy = admission_policy

    def get(self, key: int, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        """`device`: needed to promote a host-tier hit to GPU. Omit it (or
        pass None) to only ever consult the VRAM tier — matches this
        method's pre-host-tier signature/behavior exactly, so existing
        callers that never pass `device` see no change."""
        t = self._store.get(key)
        if t is not None:
            self._store.move_to_end(key)
            return t
        if device is None:
            return None
        h = self._host_store.get(key)
        if h is None:
            return None
        self._host_store.move_to_end(key)
        promoted = h.to(device, non_blocking=self._pin_host_memory)
        self._vram_put(key, promoted)  # opportunistic: also worth keeping on GPU if there's room
        return promoted

    def _vram_put(self, key: int, tensor: torch.Tensor) -> None:
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes > self.budget_bytes:
            return  # this single weight alone would never fit; don't cache it
        if self.used_bytes + nbytes > self.budget_bytes:
            if self.admission_policy == "static_fill" and key not in self._store:
                return  # tier is full under a scan-resistant policy -- don't evict, just skip
            while self.used_bytes + nbytes > self.budget_bytes and self._store:
                _, evicted = self._store.popitem(last=False)  # evict least-recently-used
                self.used_bytes -= evicted.numel() * evicted.element_size()
        self._store[key] = tensor
        self.used_bytes += nbytes

    def _host_put(self, key: int, tensor: torch.Tensor) -> None:
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes > self.host_budget_bytes:
            return
        if self.host_used_bytes + nbytes > self.host_budget_bytes:
            if self.admission_policy == "static_fill" and key not in self._host_store:
                return
            while self.host_used_bytes + nbytes > self.host_budget_bytes and self._host_store:
                _, evicted = self._host_store.popitem(last=False)
                self.host_used_bytes -= evicted.numel() * evicted.element_size()
        host_t = tensor.detach().to("cpu")
        if self._pin_host_memory and torch.cuda.is_available():
            host_t = host_t.pin_memory()
        self._host_store[key] = host_t
        self.host_used_bytes += nbytes

    def should_populate(self, key: int, nbytes: int) -> bool:
        """Whether it's worth paying a full-dequant cost right now to
        populate `key` into the cache. Under the default `"lru"` policy this
        is always True (matches the original, always-cache behavior — LRU
        eviction makes room if needed). Under `"static_fill"`, only True
        when `key` is already resident in one of the tiers (a cheap refresh)
        or when at least one tier has genuine free capacity for it without
        evicting — this is what lets `QuantLinear.forward()` skip the
        expensive `_dequantize_full` call entirely (using the fast Triton/
        chunked path instead, uncached) for a layer that would just be
        rejected by `_vram_put`/`_host_put` anyway, rather than paying the
        full dequant cost purely to throw the result away. See the class
        docstring's "Thrashing failure mode" section."""
        if self.admission_policy != "static_fill":
            return True
        if key in self._store or key in self._host_store:
            return True
        vram_room = self.budget_bytes > 0 and (self.used_bytes + nbytes <= self.budget_bytes)
        host_room = self.host_budget_bytes > 0 and (self.host_used_bytes + nbytes <= self.host_budget_bytes)
        return vram_room or host_room

    def vram_has_room(self, key: int, nbytes: int) -> bool:
        """Whether `key` is (or, under `static_fill`, could become)
        *permanently* resident in the VRAM tier specifically — as opposed to
        `should_populate`, which also counts host-only room as a reason to
        cache. Used by `QuantLinear.forward()` to decide whether a host-tier
        hit is actually worth its host->device copy: once a key is
        permanently stuck in the host tier only (no VRAM slot, ever, under
        `static_fill`), paying that copy on *every single forward call* can
        be slower than just running the fused Triton kernel directly off
        the already GPU-resident packed buffers (no transfer at all) — a
        real, measured failure mode on a many-small-layer model where a
        cache sized for ~80% of the model left the other ~20% doing a
        host->device copy every token (see `QuantLinear.forward()`'s
        "prefer the fast path" comment for the numbers). Under `"lru"`,
        always True (eviction always makes room, matching original
        behavior)."""
        if key in self._store:
            return True
        if self.admission_policy != "static_fill":
            return True
        return self.budget_bytes > 0 and (self.used_bytes + nbytes <= self.budget_bytes)

    def put(self, key: int, tensor: torch.Tensor) -> None:
        self._vram_put(key, tensor)
        if self.host_budget_bytes > 0:
            self._host_put(key, tensor)

    def clear(self) -> None:
        self._store.clear()
        self.used_bytes = 0
        self._host_store.clear()
        self.host_used_bytes = 0

    def stats(self) -> dict:
        return {
            "budget_bytes": self.budget_bytes,
            "used_bytes": self.used_bytes,
            "n_cached_layers": len(self._store),
            "host_budget_bytes": self.host_budget_bytes,
            "host_used_bytes": self.host_used_bytes,
            "n_host_cached_layers": len(self._host_store),
            "admission_policy": self.admission_policy,
        }


class QuantLinear(nn.Module):
    """Drop-in nn.Linear replacement backed by packed 4/8-bit weights.

    The weight lives in this module only as: packed codes (uint8),
    per-group (scale, min) (fp16), and a sparse outlier correction table.
    `forward()` reconstructs the weight matrix row-block by row-block
    (at most `max_dequant_rows` rows at a time) so that even for a very
    wide layer, the transient full-precision buffer used inside a single
    matmul stays bounded instead of momentarily re-inflating the entire
    layer back to fp16 size.
    """

    def __init__(self, qt: QuantizedTensor, bias: Optional[torch.Tensor] = None,
                 max_dequant_rows: Union[int, str] = DEFAULT_MAX_DEQUANT_ROWS, force_fp32_compute: bool = False,
                 cache: Optional["WeightCache"] = None, fast_dequant: bool = True, use_triton: bool = True):
        super().__init__()
        if len(qt.orig_shape) != 2:
            raise ValueError("QuantLinear requires a 2D weight (out_features, in_features)")
        self.out_features = qt.orig_shape[0]
        self.in_features = qt.orig_shape[1]
        self.bits = qt.bits
        self.group_size = qt.group_size
        self.rows = qt.rows
        self.cols = qt.cols
        self.padded_cols = qt.padded_cols
        self.n_groups = qt.n_groups
        self.orig_dtype_str = qt.orig_dtype
        if max_dequant_rows == "auto":
            max_dequant_rows = suggest_max_dequant_rows(self.out_features, self.in_features)
        self.max_dequant_rows = max(1, int(max_dequant_rows))
        # Opt-in generation-speed cache (see WeightCache's docstring): plain
        # python attributes, not registered buffers/modules, so they never
        # appear in state_dict()/parameters() and add zero serialization
        # cost. `fast_dequant` skips the fp32 intermediate round-trip in
        # `_dequantize_rows` when the compute dtype is already fp16/bf16 —
        # codes are small integers (exact in any float type) and scale/gmin
        # are already fp16-precision on disk, so this changes nothing about
        # *which* values are representable, only how many redundant
        # upcast/downcast passes happen per forward call.
        self.cache: Optional["WeightCache"] = cache
        self.fast_dequant = fast_dequant
        # Opt-out escape hatch for the fused Triton kernel path (see
        # triton_kernels.py). On by default; forward() falls back to the
        # pure-PyTorch path automatically whenever the environment/shape
        # isn't one the kernel covers (no Triton installed, CPU tensor,
        # non-default group_size, force_fp32_compute, ...), so this flag is
        # only needed to force the reference path for debugging/validation.
        self.use_triton = use_triton
        # Scratch buffers for the fused Triton path are pooled globally in
        # triton_kernels.get_buffer_pool(), so nothing to cache per-layer here.
        # force_fp32_compute: upcast dequant+matmul to fp32 and cast back for
        # output, for layers that need fp32 compute for numerical stability
        # (e.g. HF keeps T5's `wo` in fp32 — see T5_EXTRA_SKIP_PATTERNS).
        self.force_fp32_compute = force_fp32_compute

        self.register_buffer("packed", torch.from_numpy(np.ascontiguousarray(qt.packed)))

        # Double quantization (see codec.py): when qt was built with
        # double_quant=True, its scale/gmin metadata (fp16, [rows,n_groups])
        # is itself compressed to a uint8 code + one (meta_scale, meta_min)
        # fp16 pair per `dq_group_size`-element super-group -- roughly
        # halves the metadata's resident footprint (~6% of a 4-bit layer's
        # packed size down to ~3%). `qt.scale`/`qt.gmin` still come in as
        # the full reconstructed (slightly lossy) arrays either way -- see
        # quantize_tensor's comment -- so only the *resident buffers* differ
        # here; every consumer of "scale/gmin" goes through
        # `_get_scale_gmin()` below rather than the raw attributes, except
        # the one-time outlier-delta precompute a few lines down, which
        # still has `qt.scale`/`qt.gmin` (the full arrays) directly in
        # scope and uses those rather than paying for a buffer round-trip.
        self.double_quant = getattr(qt, "dq_scale_codes", None) is not None
        if self.double_quant:
            self.dq_group_size = int(qt.dq_scale_codes.shape[0] // qt.dq_scale_meta.shape[0])
            self.register_buffer("dq_scale_codes", torch.from_numpy(np.ascontiguousarray(qt.dq_scale_codes)))
            self.register_buffer("dq_scale_meta", torch.from_numpy(np.ascontiguousarray(qt.dq_scale_meta)))
            self.register_buffer("dq_gmin_codes", torch.from_numpy(np.ascontiguousarray(qt.dq_gmin_codes)))
            self.register_buffer("dq_gmin_meta", torch.from_numpy(np.ascontiguousarray(qt.dq_gmin_meta)))
            self.scale = None
            self.gmin = None
        else:
            self.dq_group_size = 0
            self.register_buffer("scale", torch.from_numpy(np.ascontiguousarray(qt.scale)))
            self.register_buffer("gmin", torch.from_numpy(np.ascontiguousarray(qt.gmin)))

        # Lloyd-Max companded codebook (see codec.py): when present, codes
        # index into this non-uniform reconstruction table instead of being
        # used directly as the (uniform) reconstruction value. The fused
        # Triton kernel (triton_kernels.py) has a `codebook_ptr`/
        # `HAS_CODEBOOK` gather for exactly this, so companded weights get
        # the same zero-cache-needed fast path as plain bits=2/4/8 --
        # `forward()` passes `self.codebook` straight into
        # `fused_quantized_linear`. `_dequantize_rows`'s own codebook gather
        # (below) is the fallback for whatever the Triton path doesn't cover
        # (CPU, no Triton installed, non-default group_size, ...), used
        # directly or, if a WeightCache is attached, once per cache fill.
        codebook = getattr(qt, 'codebook', None)
        self.companded = codebook is not None
        if self.companded:
            self.register_buffer("codebook", torch.from_numpy(np.ascontiguousarray(codebook)).to(torch.float32))
            if cache is None and not (TRITON_AVAILABLE and use_triton and self.group_size in _SUPPORTED_GROUP_SIZES):
                # The fused Triton kernel (triton_kernels.py) covers
                # companded weights directly (codebook gather fused into
                # the GEMM), so this only matters when that path isn't
                # available -- no Triton, CUDA missing at forward() time, or
                # a non-default group_size -- and forward() falls back to
                # the uncached pure-PyTorch dequant every call.
                warnings.warn(
                    "QuantLinear built with companded=True, no WeightCache, and no "
                    "fused-Triton-kernel path available (Triton/CUDA missing or "
                    "non-default group_size): every forward() call will pay the full "
                    "pure-PyTorch codebook-gather dequant, roughly 3x slower per call "
                    "than plain bits=4/8. Pass a `cache=` (or use "
                    "quantize_module(..., companded=True), which auto-sizes one) so "
                    "repeated calls reuse the cached dense weight instead.",
                    stacklevel=2,
                )
        else:
            self.codebook = None

        # int32, not int64: row/col are bounded by out_features/cols, which
        # for every quantized Linear layer in practice is well under 2**31
        # (the un-quantized embedding/lm_head layers that could exceed that
        # are excluded by min_numel/skip_name_patterns before reaching here
        # anyway). Halves this buffer's resident VRAM versus int64 for zero
        # correctness cost -- see the outlier-memory-footprint fix's note in
        # `_compute_outlier_delta`/`_apply_outlier_correction`/
        # `_dequantize_rows`, which `.long()`-cast back to int64 at their
        # (non-hot-path) point of use, since PyTorch's `index_add_` and
        # advanced-indexing require it there. The fused Triton kernel path
        # (the hot path -- see triton_kernels.py) reads these as raw
        # pointers and has no such requirement; int32 there is if anything
        # slightly cheaper than int64.
        outlier_idx = qt.outlier_idx.astype(np.int64)
        if outlier_idx.size > 0:
            outlier_row = outlier_idx // qt.cols
            outlier_col = outlier_idx % qt.cols
            # Sort by row so each chunk's outliers are a contiguous slice,
            # located via precomputed Python-int bounds (numpy.searchsorted)
            # instead of a boolean mask + `.any()` per forward() call — that
            # requires a GPU->CPU sync, which is expensive on Windows/WDDM.
            sort_perm = np.argsort(outlier_row, kind="stable")
            outlier_row = outlier_row[sort_perm].astype(np.int32)
            outlier_col = outlier_col[sort_perm].astype(np.int32)
            outlier_val_sorted = qt.outlier_val[sort_perm]
        else:
            outlier_row = np.zeros((0,), dtype=np.int32)
            outlier_col = np.zeros((0,), dtype=np.int32)
            outlier_val_sorted = qt.outlier_val
        self.register_buffer("outlier_row", torch.from_numpy(outlier_row))
        self.register_buffer("outlier_col", torch.from_numpy(outlier_col))

        # outlier_val (the outliers' true full-precision values) is NOT kept
        # as a resident buffer: once outlier_delta is precomputed below,
        # every consumer only ever needs val = base_dequant_val + delta, and
        # base_dequant_val is either already sitting in the pure-PyTorch
        # dequant's `flat` buffer right before it gets overwritten (see
        # `_dequantize_rows`) or cheap to recompute from scale/gmin+codes
        # (see `to_quantized_tensor`) -- so keeping a whole separate copy of
        # values already recoverable from delta+base was pure waste. Only
        # needed transiently here, to compute delta itself.
        outlier_val_t = torch.from_numpy(np.ascontiguousarray(outlier_val_sorted))

        # Precomputed delta between each outlier's true value and what the
        # base group-affine dequant would produce there. The fused Triton
        # kernel reconstructs the clipped weight only and adds this back as
        # a sparse correction: `y[:, row] += x[:, col] * delta[k]`.
        self.register_buffer("outlier_delta", self._compute_outlier_delta(outlier_val_t))

        # (chunk_start, chunk_end) -> (lo, hi) slice bounds into the sorted
        # outlier buffers above, as plain Python ints — not tensors, so
        # using them never touches the GPU and needs no synchronization.
        self._outlier_chunk_bounds: Dict[Tuple[int, int], Tuple[int, int]] = {}
        step = self.max_dequant_rows
        for start in range(0, self.rows, step):
            end = min(start + step, self.rows)
            lo = int(np.searchsorted(outlier_row, start, side="left"))
            hi = int(np.searchsorted(outlier_row, end, side="left"))
            self._outlier_chunk_bounds[(start, end)] = (lo, hi)

        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None

        # Pre-compute static Triton eligibility and kwargs so forward() can
        # skip the per-call condition checks and 20+ argument construction.
        # Invalidated and rebuilt by _rebuild_triton_cache() after .to(device).
        self._triton_eligible: bool = False
        self._triton_kwargs: Optional[dict] = None
        self._rebuild_triton_cache()

    def _rebuild_triton_cache(self) -> None:
        """Pre-compute whether this layer qualifies for the fused Triton path
        and, if so, all the static arguments `fused_quantized_linear` needs.
        Called once from __init__ and again from `_apply(fn)` (which `.to()`
        triggers) to pick up device changes."""
        eligible = (
            self.use_triton
            and TRITON_AVAILABLE
            and not self.force_fp32_compute
            and self.group_size in _SUPPORTED_GROUP_SIZES
            and self.bits in _TRITON_SUPPORTED_BITS
        )
        if eligible and self.packed.is_cuda:
            self._triton_eligible = True
            self._triton_kwargs = dict(
                packed=self.packed,
                scale=self.scale if not self.double_quant else None,
                gmin=self.gmin if not self.double_quant else None,
                bits=self.bits,
                group_size=self.group_size,
                cols=self.cols,
                out_features=self.out_features,
                outlier_row=self.outlier_row if self.outlier_row.numel() > 0 else None,
                outlier_col=self.outlier_col if self.outlier_col.numel() > 0 else None,
                outlier_delta=self.outlier_delta if self.outlier_delta.numel() > 0 else None,
                codebook=self.codebook if self.companded else None,
                dq_scale_codes=self.dq_scale_codes if self.double_quant else None,
                dq_scale_meta=self.dq_scale_meta if self.double_quant else None,
                dq_gmin_codes=self.dq_gmin_codes if self.double_quant else None,
                dq_gmin_meta=self.dq_gmin_meta if self.double_quant else None,
                dq_group_size=self.dq_group_size,
                n_groups=self.n_groups,
            )
            # Note: DQ scale/gmin cache (_cached_scale_gmin) is populated
            # lazily on first call to _get_scale_gmin(). For UNet workloads
            # (large M, many calls through the dequant+cuBLAS path), this
            # caches after the first forward pass. For LLM decode (small M,
            # fused kernel path), _get_scale_gmin is never called, so no
            # VRAM is wasted on scale/gmin tensors the fused kernel reads
            # directly from DQ buffers.
        else:
            self._triton_eligible = eligible  # True but not yet on CUDA
            self._triton_kwargs = None

    def _apply(self, fn, recurse=True):
        """Override Module._apply to rebuild caches after .to()
        moves buffers to a new device."""
        result = super()._apply(fn, recurse=recurse)
        self._cached_scale_gmin = None  # invalidate DQ cache
        self._rebuild_triton_cache()
        return result

    # -- introspection --------------------------------------------------

    @property
    def weight(self) -> torch.Tensor:
        """A zero-cost (`device="meta"`, no storage) tensor exposed purely
        for compatibility with library code that introspects
        `some_linear.weight` without checking the module type first.

        e.g. `transformers`' T5 dense layer checks
        `hidden.dtype != self.wo.weight.dtype and weight.dtype != torch.int8`
        as an fp16-overflow guard, skipping the cast when the weight looks
        already-quantized (the same check bitsandbytes' Linear8bitLt relies
        on). `dtype=torch.int8` here is that "already quantized" sentinel
        regardless of actual bit-width — `forward()` never reads this
        property, it dequantizes from the packed buffers directly.
        """
        return torch.empty(self.out_features, self.in_features, dtype=torch.int8, device="meta")

    def packed_nbytes(self) -> int:
        n = self.packed.numel() * self.packed.element_size()
        if self.double_quant:
            n += self.dq_scale_codes.numel() * self.dq_scale_codes.element_size()
            n += self.dq_scale_meta.numel() * self.dq_scale_meta.element_size()
            n += self.dq_gmin_codes.numel() * self.dq_gmin_codes.element_size()
            n += self.dq_gmin_meta.numel() * self.dq_gmin_meta.element_size()
        else:
            n += self.scale.numel() * self.scale.element_size()
            n += self.gmin.numel() * self.gmin.element_size()
        n += self.outlier_row.numel() * self.outlier_row.element_size()
        n += self.outlier_col.numel() * self.outlier_col.element_size()
        n += self.outlier_delta.numel() * self.outlier_delta.element_size()
        return n

    def _get_scale_gmin(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full `[rows, n_groups]` fp32 scale/gmin. When `double_quant` is
        on, materializes them from the compact DQ buffers. Cached after the
        first call (the DQ codes never change during inference) to avoid
        repeated small-tensor ops — significant in UNet workloads where
        this is called 700+ times per forward pass through the large-M
        dequant+cuBLAS path."""
        # Fast path: return cached result (set by first call or _apply)
        cached = getattr(self, "_cached_scale_gmin", None)
        if cached is not None:
            return cached

        if not self.double_quant:
            result = self.scale.to(torch.float32), self.gmin.to(torch.float32)
        else:
            def _mat(codes: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
                n_super = meta.shape[0]
                dq_gs = codes.numel() // n_super
                c = codes.to(torch.float32).reshape(n_super, dq_gs)
                m = meta.to(torch.float32)
                recon = c * m[:, 0:1] + m[:, 1:2]
                return recon.reshape(-1)[: self.rows * self.n_groups].reshape(self.rows, self.n_groups)
            result = _mat(self.dq_scale_codes, self.dq_scale_meta), _mat(self.dq_gmin_codes, self.dq_gmin_meta)

        self._cached_scale_gmin = result
        return result

    @staticmethod
    def estimate_resident_nbytes(qt: QuantizedTensor) -> int:
        """Alias for `codec.estimate_resident_nbytes` — see that function's
        docstring. Kept here too since call sites already spell it as
        `QuantLinear.estimate_resident_nbytes(qt)`."""
        return _estimate_resident_nbytes(qt)

    def dense_equiv_nbytes(self, itemsize: int = 2) -> int:
        return self.out_features * self.in_features * itemsize

    def to_quantized_tensor(self) -> QuantizedTensor:
        """Reconstruct a QuantizedTensor (e.g. to re-save to a container).

        When `double_quant` is on, passes the compact dq_* fields straight
        through instead of round-tripping through materialized full
        scale/gmin arrays — preserves the compression instead of quietly
        undoing it on every save, and `scale`/`gmin` are still filled in
        (materialized once here) since QuantizedTensor's dataclass declares
        them as required fields that other consumers (e.g.
        `compressed_nbytes`'s non-double-quant branch, or code that doesn't
        know about double_quant) may still read directly."""
        row = self.outlier_row.cpu().numpy()
        col = self.outlier_col.cpu().numpy()
        idx = (row * self.cols + col).astype(np.int32)
        scale_full, gmin_full = self._get_scale_gmin()
        # outlier_val isn't kept resident (see __init__'s comment) --
        # reconstruct it as base_val + delta, the exact inverse of how
        # outlier_delta was computed in the first place.
        outlier_val = (self._outlier_base_val() + self.outlier_delta.to(torch.float32)).to(torch.float16)
        kwargs = dict(
            orig_shape=(self.out_features, self.in_features),
            orig_dtype=self.orig_dtype_str,
            bits=self.bits,
            group_size=self.group_size,
            rows=self.rows,
            cols=self.cols,
            padded_cols=self.padded_cols,
            n_groups=self.n_groups,
            packed=self.packed.cpu().numpy(),
            scale=scale_full.to(torch.float16).cpu().numpy(),
            gmin=gmin_full.to(torch.float16).cpu().numpy(),
            outlier_idx=idx,
            outlier_val=outlier_val.cpu().numpy(),
        )
        if self.double_quant:
            kwargs.update(
                dq_scale_codes=self.dq_scale_codes.cpu().numpy(),
                dq_scale_meta=self.dq_scale_meta.cpu().numpy(),
                dq_gmin_codes=self.dq_gmin_codes.cpu().numpy(),
                dq_gmin_meta=self.dq_gmin_meta.cpu().numpy(),
            )
        if self.companded:
            # Pre-existing bug, found via this session's round-trip test:
            # `codebook` was never included here, so saving a companded
            # QuantLinear to a container and reloading it silently dropped
            # back to plain uniform-grid reconstruction -- the codes stayed
            # the same but their meaning (index into the Lloyd-Max
            # codebook) was lost, corrupting every companded weight's
            # dequantized value.
            kwargs.update(codebook=self.codebook.cpu().numpy())
        return QuantizedTensor(**kwargs)

    # -- outlier correction (used by the fused-kernel forward path) ------

    def _outlier_base_val(self) -> torch.Tensor:
        """The plain group-affine dequant value (fp32) at each outlier's
        (row, col) — i.e. what that position would decode to *before* the
        outlier correction is applied. Shared by `_compute_outlier_delta`
        (delta = outlier_val - this, at construction) and
        `to_quantized_tensor` (outlier_val = this + delta, on demand,
        now that outlier_val isn't kept resident — see its docstring)."""
        n = self.outlier_row.numel()
        if n == 0:
            return torch.zeros(0, dtype=torch.float32, device=self.packed.device)
        # outlier_row/col are stored as int32 (see __init__) to halve their
        # resident VRAM; PyTorch's advanced indexing below requires int64,
        # so upcast here -- not a hot path (construction-time or container
        # round-trip only), so the transient int64 copy costs nothing at
        # steady state.
        row = self.outlier_row.to(torch.int64)
        group = (self.outlier_col // self.group_size).to(torch.int64)
        within = (self.outlier_col % self.group_size).to(torch.int64)
        if self.bits == 2:
            byte_idx = (group * self.group_size + within) // 4
            crumb_idx = within % 4  # which of the 4 codes packed into this byte
            bytes_ = self.packed[row, byte_idx]
            c0 = (bytes_ & 0x03).to(torch.float32)
            c1 = ((bytes_ >> 2) & 0x03).to(torch.float32)
            c2 = ((bytes_ >> 4) & 0x03).to(torch.float32)
            c3 = ((bytes_ >> 6) & 0x03).to(torch.float32)
            code = torch.where(
                crumb_idx == 0, c0, torch.where(crumb_idx == 1, c1, torch.where(crumb_idx == 2, c2, c3))
            )
        elif self.bits == 4:
            byte_idx = (group * self.group_size + within) // 2
            is_odd = (within % 2 == 1)
            bytes_ = self.packed[row, byte_idx]
            low = (bytes_ & 0x0F).to(torch.float32)
            high = ((bytes_ >> 4) & 0x0F).to(torch.float32)
            code = torch.where(is_odd, high, low)
        else:
            byte_idx = group * self.group_size + within
            code = self.packed[row, byte_idx].to(torch.float32)
        if self.companded:
            # Companded weights reconstruct as codebook[code], not code
            # itself (see codec.py) — the fused kernel's base value (what
            # this delta corrects *from*) uses the same lookup, so this
            # must match it exactly or the outlier correction would be
            # computed against the wrong baseline.
            code = self.codebook[code.to(torch.long)].to(torch.float32)
        scale_full, gmin_full = self._get_scale_gmin()
        sc = scale_full[row, group]
        gm = gmin_full[row, group]
        return code * sc + gm

    def _compute_outlier_delta(self, outlier_val: torch.Tensor) -> torch.Tensor:
        """`outlier_val - base_dequant_val` at each outlier's (row, col),
        computed once from the already-packed buffers. See the
        `outlier_delta` buffer's docstring in __init__ for why this is
        precomputed instead of recomputed per forward() call."""
        if outlier_val.numel() == 0:
            return torch.zeros(0, dtype=torch.float16)
        base_val = self._outlier_base_val()
        # fp16, not fp32: every consumer (the fused Triton kernel's
        # _add_outlier_tile, and _apply_outlier_correction's PyTorch
        # fallback below) casts this straight back up to fp32 immediately
        # after loading/reading it, so storing it at fp32 resident
        # precision bought nothing -- it's a small correction term already
        # derived from fp16-precision inputs (outlier_val, scale, gmin).
        # Halves this buffer's resident VRAM for zero numerical difference.
        return (outlier_val.to(base_val.device).to(torch.float32) - base_val).to(torch.float16).contiguous()

    def _apply_outlier_correction(self, out: torch.Tensor, x2d: torch.Tensor) -> torch.Tensor:
        """`out` is the fused kernel's result (dequantized from clipped
        codes only, no outliers). Adds back each outlier's exact
        contribution: replacing a clipped weight with its true value changes
        dot-product output row `row` by `x[:, col] * delta` — see
        `outlier_delta`'s docstring.

        Tries the Triton atomic-add kernel first (`apply_outlier_correction`
        in triton_kernels.py) — measured to matter: PyTorch's `index_add_`
        path below has real per-call overhead plus atomic-contention cost
        from many outliers landing in the same output row (~6/row at the
        default 0.3% `outlier_frac`), making it comparable in cost to the
        fused GEMM kernel itself on some shapes. Falls back to the
        gather+`index_add_` path when Triton isn't available/applicable
        (CPU tensors, no CUDA) — same result either way, just slower.
        """
        n = self.outlier_row.numel()
        if n == 0:
            return out
        triton_result = _triton_apply_outlier_correction(
            out, x2d, self.outlier_row, self.outlier_col, self.outlier_delta
        )
        if triton_result is not None:
            return triton_result
        contrib = x2d[:, self.outlier_col.long()].to(torch.float32) * self.outlier_delta.to(torch.float32).unsqueeze(0)
        out2d = out.reshape(-1, self.out_features)
        # index_add_ requires an int64 index tensor -- outlier_row is int32
        # (see __init__), so cast here. Not the hot path: this PyTorch
        # fallback only runs when the Triton kernel is unavailable/not
        # applicable (see apply_outlier_correction's None-return contract).
        out2d.index_add_(1, self.outlier_row.long(), contrib.to(out2d.dtype))
        return out2d.reshape(out.shape)

    # -- core dequant + forward ------------------------------------------

    def _dequantize_rows(self, row_start: int, row_end: int, compute_dtype: torch.dtype) -> torch.Tensor:
        n_rows = row_end - row_start
        packed_chunk = self.packed[row_start:row_end]

        # Run the affine reconstruction (codes*scale+gmin) directly in fp16/bf16
        # when requested, instead of always upcasting to fp32: scale/gmin are
        # already stored at fp16 precision, and integer codes are exact in
        # fp16, so this saves a redundant upcast pass with no precision loss.
        calc_dtype = (
            compute_dtype
            if (self.fast_dequant and compute_dtype in (torch.float16, torch.bfloat16))
            else torch.float32
        )
        scale_full, gmin_full = self._get_scale_gmin()
        scale_chunk = scale_full[row_start:row_end].to(calc_dtype)
        gmin_chunk = gmin_full[row_start:row_end].to(calc_dtype)

        if self.bits == 2:
            codes_flat = _unpack_crumbs(packed_chunk, self.padded_cols)
        elif self.bits == 4:
            codes_flat = _unpack_nibbles(packed_chunk, self.padded_cols)
        else:
            codes_flat = packed_chunk

        if self.companded:
            cb = self.codebook.to(calc_dtype)
            codes = cb[codes_flat.reshape(-1).to(torch.long)].reshape(
                n_rows, self.n_groups, self.group_size)
        else:
            codes = codes_flat.reshape(n_rows, self.n_groups, self.group_size).to(calc_dtype)
        deq = codes * scale_chunk.unsqueeze(-1) + gmin_chunk.unsqueeze(-1)
        deq = deq.reshape(n_rows, self.padded_cols)[:, : self.cols]
        flat = deq.reshape(-1)

        # Precomputed slice bounds from __init__ — avoids a boolean mask +
        # `.any()`, which would force a GPU->CPU sync here.
        lo, hi = self._outlier_chunk_bounds[(row_start, row_end)]
        if hi > lo:
            local_row = self.outlier_row[lo:hi] - row_start
            local_col = self.outlier_col[lo:hi]
            # outlier_row/col are int32 (see __init__); cast to int64 for
            # the index-assignment below, PyTorch's expected index dtype.
            local_idx = (local_row.to(torch.int64) * self.cols + local_col.to(torch.int64))
            # outlier_val isn't kept resident (see __init__'s comment):
            # `flat[local_idx]` right now still holds the plain group-affine
            # dequant value at each outlier position (nothing has
            # overwritten it yet this call) -- exactly `_outlier_base_val`'s
            # definition -- so val = that + delta, computed in fp32 and cast
            # down, same precision path the removed buffer would have taken.
            base = flat[local_idx].to(torch.float32)
            flat[local_idx] = (base + self.outlier_delta[lo:hi].to(torch.float32)).to(calc_dtype)

        return flat.reshape(n_rows, self.cols).to(compute_dtype)

    def _dequantize_full(self, compute_dtype: torch.dtype) -> torch.Tensor:
        """Dequantize the entire weight in one shot, chunked only internally
        by `max_dequant_rows` to bound transient memory, then concatenated.
        Used for the `WeightCache` path, where the result is meant to be
        kept around (so there's no reason to also chunk the *matmul*, unlike
        the uncached path in `forward()`)."""
        rows = self.rows
        step = self.max_dequant_rows
        if rows <= step:
            return self._dequantize_rows(0, rows, compute_dtype)
        chunks = [self._dequantize_rows(s, min(s + step, rows), compute_dtype) for s in range(0, rows, step)]
        return torch.cat(chunks, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        compute_dtype = torch.float32 if self.force_fp32_compute else in_dtype
        if self.force_fp32_compute and x.dtype != torch.float32:
            x = x.to(torch.float32)
        bias = self.bias.to(compute_dtype) if self.bias is not None else None

        # Generation-speed fast path: if this weight is already dequantized in
        # the shared cache from a previous call, skip straight to the matmul.
        # See WeightCache's docstring for the VRAM-budget tradeoff.
        if self.cache is not None:
            cache_key = id(self)
            itemsize = 2 if compute_dtype in (torch.float16, torch.bfloat16) else 4
            nbytes = self.out_features * self.in_features * itemsize

            # Only prefer the cache when this key has (or, under
            # `static_fill`, could get) a permanent VRAM slot. A key stuck in
            # the host tier only would otherwise pay a host->device copy on
            # every call, which is slower than running the fused Triton
            # kernel straight off the GPU-resident packed buffers — but only
            # when that fast path is actually usable (CUDA + Triton).
            triton_fast_path_usable = (
                self._triton_eligible
                and self._triton_kwargs is not None
                and compute_dtype in (torch.float16, torch.bfloat16)
            )
            prefer_cache = self.cache.vram_has_room(cache_key, nbytes) or not triton_fast_path_usable

            if prefer_cache:
                w = self.cache.get(cache_key, device=x.device)
                if w is not None and w.dtype == compute_dtype:
                    out = F.linear(x, w, bias)
                    return out.to(in_dtype) if out.dtype != in_dtype else out
                # Cache miss (or dtype changed since last cached). Only pay
                # for a full dequant when the cache has somewhere to put the
                # result — see WeightCache's "Thrashing failure mode" section.
                if self.cache.should_populate(cache_key, nbytes):
                    w = self._dequantize_full(compute_dtype)
                    self.cache.put(cache_key, w)
                    out = F.linear(x, w, bias)
                    return out.to(in_dtype) if out.dtype != in_dtype else out
            # else: fall through to the fast Triton/chunked path below, uncached.

        # Fused Triton kernel path: unpack + per-group affine reconstruct +
        # GEMM + outlier correction in one kernel launch, dense weight never
        # materialized. Pre-computed eligibility and kwargs from
        # _rebuild_triton_cache() make this path near-zero Python overhead.
        #
        # IMPORTANT: the fused kernel only wins for small M (decode-shaped,
        # M <= ~64). For large M (UNet attention with spatial tokens, prefill
        # with long prompts), each of the `ceil(M/BLOCK_M)` row-tiles
        # independently re-reads and re-dequantizes the same packed weight.
        # At M=16384 (1024x1024 SDXL latent), that's 128x redundant weight
        # reads, which *loses* to dequant-once + cuBLAS (F.linear). Measured
        # on RTX 4060: fused gave 0.71 it/s vs 2.1 it/s for cuBLAS on the
        # same UNet. Threshold 64 keeps the fused path for all LLM decode
        # shapes and small-batch UNet passes.
        # M = total rows in the 2D view (product of all dims except last).
        # Use Python ints to avoid touching the GPU.
        M = 1
        for d in x.shape[:-1]:
            M *= d
        if (
            self._triton_eligible
            and self._triton_kwargs is not None
            and compute_dtype in (torch.float16, torch.bfloat16)
            and M <= 64
        ):
            x2d = x.reshape(-1, self.in_features)
            fused = fused_quantized_linear(
                x2d, bias=bias, compute_dtype=compute_dtype,
                **self._triton_kwargs,
            )
            if fused is not None:
                out = fused.reshape(*x.shape[:-1], self.out_features)
                return out.to(in_dtype) if out.dtype != in_dtype else out

        # Large-M path: fused Triton dequant (1 kernel) + cuBLAS GEMM.
        # Falls through to the chunked PyTorch path when Triton is unavailable.
        if (
            self._triton_eligible
            and self.packed.is_cuda
            and compute_dtype in (torch.float16, torch.bfloat16)
        ):
            scale_full, gmin_full = self._get_scale_gmin()
            w = _triton_dequantize_weight(
                self.packed, scale_full, gmin_full,
                bits=self.bits, rows=self.rows, cols=self.cols,
                padded_cols=self.padded_cols, n_groups=self.n_groups,
                group_size=self.group_size, compute_dtype=compute_dtype,
                codebook=self.codebook if self.companded else None,
            )
            if w is not None:
                # Apply outlier correction directly to the weight
                if self.outlier_row.numel() > 0:
                    flat = w.reshape(-1)
                    row64 = self.outlier_row.to(torch.int64)
                    col64 = self.outlier_col.to(torch.int64)
                    idx = row64 * self.cols + col64
                    base = flat[idx].to(torch.float32)
                    flat[idx] = (base + self.outlier_delta.to(torch.float32)).to(compute_dtype)
                out = F.linear(x, w, bias)
                del w
                return out.to(in_dtype) if out.dtype != in_dtype else out

        # Fallback: chunked PyTorch dequant + F.linear.
        rows = self.rows
        step = self.max_dequant_rows
        if rows <= step:
            w = self._dequantize_rows(0, rows, compute_dtype)
            out = F.linear(x, w, bias)
        else:
            chunks = []
            for start in range(0, rows, step):
                end = min(start + step, rows)
                w_chunk = self._dequantize_rows(start, end, compute_dtype)
                b_chunk = bias[start:end] if bias is not None else None
                chunks.append(F.linear(x, w_chunk, b_chunk))
                del w_chunk
            out = torch.cat(chunks, dim=-1)

        return out.to(in_dtype) if out.dtype != in_dtype else out

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int = 4,
        group_size: int = 128,
        outlier_frac: float = 0.001,
        adaptive: bool = False,
        error_threshold: float = 0.10,
        max_dequant_rows: Union[int, str] = DEFAULT_MAX_DEQUANT_ROWS,
        force_fp32_compute: bool = False,
        cache: Optional["WeightCache"] = None,
        fast_dequant: bool = True,
        use_triton: bool = True,
        companded: bool = False,
        double_quant: bool = True,
        dq_group_size: int = DEFAULT_DQ_GROUP_SIZE,
        nf_quant: bool = True,
    ) -> Tuple["QuantLinear", float, bool]:
        w = linear.weight.data
        if adaptive:
            # allowed_bits=(2, 4, 8): QuantLinear's forward-pass dequant
            # supports these three bit-widths (2-bit crumb-packed added in
            # v0.4 -- see modules.py's _dequantize_rows/_compute_outlier_delta).
            # 6-bit (RVQ) is excluded: it needs a second packed/scale/gmin
            # buffer set QuantLinear doesn't build yet, and the container
            # format doesn't serialize those fields either -- see
            # codec.quantize_tensor_adaptive's allowed_bits docstring.
            qt, err, escalated = quantize_tensor_adaptive(
                w, group_size=group_size, outlier_frac=outlier_frac, error_threshold=error_threshold,
                allowed_bits=(2, 4, 8), companded=companded,
                double_quant=double_quant, dq_group_size=dq_group_size,
                nf_quant=nf_quant,
            )
        else:
            qt = quantize_tensor(w, bits=bits, group_size=group_size, outlier_frac=outlier_frac,
                                  companded=companded,
                                  double_quant=double_quant, dq_group_size=dq_group_size,
                                  nf_quant=nf_quant)
            err = relative_error(w, qt)
            escalated = False
        bias = linear.bias.data if linear.bias is not None else None
        qlin = cls(
            qt, bias, max_dequant_rows=max_dequant_rows, force_fp32_compute=force_fp32_compute,
            cache=cache, fast_dequant=fast_dequant, use_triton=use_triton,
        )
        # The codec always hands back packed/scale/gmin/outlier buffers as
        # numpy arrays (CPU) -- see codec.py's `.cpu().numpy()` calls, needed
        # for the .tsk container format -- so a freshly built QuantLinear is
        # CPU-resident regardless of where `linear` itself lived. Move it
        # back to `linear`'s device so replacing a layer in an
        # already-GPU-resident model doesn't silently strand its weights on
        # the host: the fused Triton path requires CUDA tensors and quietly
        # falls back when it doesn't get them (see fused_quantized_linear),
        # and the pure-PyTorch fallback outright crashes on the resulting
        # cross-device matmul once a bias (also CUDA) is involved. Mirrors
        # the `.to(target_device)` hf_integration.py already does for the
        # transformers quantization_config path.
        device = linear.weight.device
        if device.type != "cpu":
            qlin = qlin.to(device)
        return qlin, err, escalated

    @classmethod
    def from_quantized_tensor(
        cls,
        qt: QuantizedTensor,
        bias: Optional[torch.Tensor] = None,
        max_dequant_rows: Union[int, str] = DEFAULT_MAX_DEQUANT_ROWS,
        force_fp32_compute: bool = False,
        cache: Optional["WeightCache"] = None,
        fast_dequant: bool = True,
        use_triton: bool = True,
    ) -> "QuantLinear":
        return cls(
            qt, bias, max_dequant_rows=max_dequant_rows, force_fp32_compute=force_fp32_compute,
            cache=cache, fast_dequant=fast_dequant, use_triton=use_triton,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, group_size={self.group_size}, "
            f"packed_kB={self.packed_nbytes()/1024:.1f}, dense_fp16_kB={self.dense_equiv_nbytes()/1024:.1f}"
        )


# ---------------------------------------------------------------------------
# whole-model helpers
# ---------------------------------------------------------------------------

def _set_module_by_name(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parts = dotted_name.split(".")
    obj = root
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], new_module)


def _iter_target_linears(
    model: nn.Module, min_numel: int, skip_name_patterns: Iterable[str]
) -> Iterable[Tuple[str, nn.Linear]]:
    skip_name_patterns = tuple(p.lower() for p in skip_name_patterns)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not isinstance(module, QuantLinear):
            if module.weight.numel() < min_numel:
                continue
            if any(p in name.lower() for p in skip_name_patterns):
                continue
            yield name, module


def model_weight_bytes(model: nn.Module) -> int:
    """Theoretical (allocator-independent) byte footprint of every
    parameter + buffer currently attached to the model. Used as a
    cross-check alongside measured torch.cuda memory counters."""
    total = 0
    for p in model.parameters(recurse=True):
        total += p.numel() * p.element_size()
    for b in model.buffers(recurse=True):
        total += b.numel() * b.element_size()
    return total


def _parallel_windowed(
    items: List[Optional[Tuple[str, nn.Linear]]],
    worker_fn,
    max_workers: int,
    window_multiplier: int = 2,
):
    """Runs `worker_fn(name, lin)` over `items` (a `[(name, nn.Linear), ...]`
    list, mutated in place) on a `ThreadPoolExecutor`, yielding
    `(index, result)` pairs as each completes.

    Unlike submitting every task upfront (`executor.map` / a dict
    comprehension of `ex.submit(...)` over the whole list), this caps how
    many tasks are in flight at once to `max_workers * window_multiplier`
    and only submits task `i` once an earlier slot frees up. That bound
    matters here specifically because each task's args include a
    `nn.Linear` (i.e. its fp16 weight tensor) — `ThreadPoolExecutor`'s
    internal work queue holds a task's args alive from the moment it's
    submitted, not from when a worker thread actually starts it, so
    submitting all `n` tasks at once keeps *every* layer's fp16 weight
    referenced simultaneously regardless of how many threads are actually
    running — silently reintroducing the "whole model resident at once"
    problem this module's per-layer `targets[i] = None` freeing is meant to
    avoid, just moved from the `targets` list into the executor's queue.
    Bounding the window keeps at most ~`window_multiplier` layers per
    worker thread's weight alive at a time instead of all of them.
    """
    n = len(items)
    window = max(1, max_workers) * max(1, window_multiplier)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        next_idx = 0

        def _submit(i):
            name, lin = items[i]
            fut = ex.submit(worker_fn, name, lin)
            futures[fut] = i

        while next_idx < min(window, n):
            _submit(next_idx)
            next_idx += 1

        while futures:
            done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                i = futures.pop(fut)
                yield i, fut.result()
                if next_idx < n:
                    _submit(next_idx)
                    next_idx += 1


def _quantize_module_budget_aware(
    model: nn.Module,
    targets: List[Tuple[str, nn.Linear]],
    force_fp32_patterns: Tuple[str, ...],
    group_size: int,
    outlier_frac: float,
    vram_budget_mb: float,
    max_dequant_rows: Union[int, str],
    cache: Optional["WeightCache"],
    fast_dequant: bool,
    use_triton: bool,
    quantize_workers: Optional[int],
    stats: dict,
    verbose: bool,
) -> dict:
    """`quantize_module(..., vram_budget_mb=...)`'s implementation — see
    that docstring. Two passes over `targets`: first quantize every layer at
    *both* 4-bit and 8-bit to collect (bytes, error) for each, run the
    greedy allocator once over the whole model, then build each QuantLinear
    from whichever of the two already-computed QuantizedTensors the
    allocator picked (no third quantize pass).

    Both passes free each `nn.Linear`'s dense fp16 weight (the only large
    thing here — `computed[name]`'s two QuantizedTensors are already 2-4x
    smaller) the moment they're done with it: `targets[i]` is overwritten in
    place with `None` right after use, dropping the last remaining
    reference (the model itself no longer points at it once
    `_set_module_by_name` swaps in the QuantLinear) so CPython's refcounting
    frees the tensor immediately instead of only at function return. See
    `quantize_module`'s matching comment for why this matters — leaving
    stale entries in `targets` was silently keeping the *entire* original
    fp16 model resident in host RAM for the whole quantize pass.

    The first pass (independent per layer — every layer's own weight, no
    shared state) runs on a `ThreadPoolExecutor`: `quantize_tensor`'s hot
    loop is `torch` tensor ops (topk/round/reshape/matmul-adjacent), which
    release the GIL for the underlying C++/BLAS/vectorized work same as any
    other torch kernel, so real wall-clock parallelism is available across
    CPU cores without the pickling cost multiprocessing would pay to ship
    each already-resident weight tensor across a process boundary.
    """
    n = len(targets)
    layer_stats_by_name: Dict[str, LayerBitStats] = {}
    computed: Dict[str, dict] = {}  # name -> {4: (qt, err), 8: (qt, err)}
    biases: Dict[str, Optional[torch.Tensor]] = {}
    itemsizes: Dict[str, int] = {}
    devices: Dict[str, torch.device] = {}
    names_order = [name for name, _ in targets]

    def _quantize_both(name: str, lin: nn.Linear):
        w = lin.weight.data
        qt4 = quantize_tensor(w, bits=4, group_size=group_size, outlier_frac=outlier_frac)
        err4 = relative_error(w, qt4)
        qt8 = quantize_tensor(w, bits=8, group_size=group_size, outlier_frac=outlier_frac)
        err8 = relative_error(w, qt8)
        # QuantLinear.estimate_resident_nbytes, not qt.compressed_nbytes() --
        # the allocator needs to plan against the byte cost that actually
        # ends up resident on GPU (wider outlier tables than the compact
        # on-disk form), or the resulting plan can quietly exceed
        # vram_budget_mb once built. See that method's docstring.
        bytes4 = _estimate_resident_nbytes(qt4)
        bytes8 = _estimate_resident_nbytes(qt8)
        bias = lin.bias.data if lin.bias is not None else None
        itemsize = lin.weight.element_size()
        device = lin.weight.device
        return name, qt4, err4, bytes4, qt8, err8, bytes8, bias, itemsize, device

    max_workers = min(quantize_workers or 1, max(1, os.cpu_count() or 1))
    if n > 1 and max_workers > 1:
        for i, result in _parallel_windowed(targets, _quantize_both, max_workers):
            name, qt4, err4, bytes4, qt8, err8, bytes8, bias, itemsize, device = result
            layer_stats_by_name[name] = LayerBitStats(name, bytes4, bytes8, err4, err8)
            computed[name] = {4: (qt4, err4), 8: (qt8, err8)}
            biases[name] = bias
            itemsizes[name] = itemsize
            devices[name] = device
            targets[i] = None  # drop the (name, nn.Linear) tuple -> frees the fp16 weight now
    else:
        for i, (name, lin) in enumerate(targets):
            name, qt4, err4, bytes4, qt8, err8, bytes8, bias, itemsize, device = _quantize_both(name, lin)
            layer_stats_by_name[name] = LayerBitStats(name, bytes4, bytes8, err4, err8)
            computed[name] = {4: (qt4, err4), 8: (qt8, err8)}
            biases[name] = bias
            itemsizes[name] = itemsize
            devices[name] = device
            targets[i] = None

    layer_stats = [layer_stats_by_name[name] for name in names_order]
    budget_bytes = vram_budget_mb * 1024 * 1024
    bit_plan = budget_allocate_bits(layer_stats, budget_bytes)

    for name in names_order:
        chosen_bits = bit_plan[name]
        qt, err = computed[name][chosen_bits]
        force_fp32 = any(p in name.lower() for p in force_fp32_patterns)
        bias = biases[name]
        qlin = QuantLinear(
            qt, bias, max_dequant_rows=max_dequant_rows, force_fp32_compute=force_fp32,
            cache=cache, fast_dequant=fast_dequant, use_triton=use_triton,
        )
        # See from_linear's matching comment: qt's buffers are numpy (CPU),
        # so move the built QuantLinear back to the original layer's device
        # before swapping it in.
        device = devices[name]
        if device.type != "cpu":
            qlin = qlin.to(device)
        _set_module_by_name(model, name, qlin)

        orig_bytes = qlin.dense_equiv_nbytes(itemsize=itemsizes[name])
        packed_bytes = qlin.packed_nbytes()
        stats["n_replaced"] += 1
        stats["orig_bytes"] += orig_bytes
        stats["packed_bytes"] += packed_bytes
        stats["layers"].append(
            {
                "name": name, "bits": chosen_bits, "rel_error": err, "escalated": chosen_bits == 8,
                "orig_bytes": orig_bytes, "packed_bytes": packed_bytes, "force_fp32_compute": force_fp32,
            }
        )
        if verbose:
            flag = "*" if chosen_bits == 8 else " "
            fp32_flag = " fp32" if force_fp32 else ""
            print(
                f"  [{chosen_bits}b{flag}{fp32_flag}] {name:60s} err={err:.4f} "
                f"{orig_bytes/1e6:8.3f}MB -> {packed_bytes/1e6:8.3f}MB"
            )

    stats["ratio"] = stats["orig_bytes"] / max(1, stats["packed_bytes"]) if stats["orig_bytes"] > 0 else 1.0
    stats["vram_budget_mb"] = vram_budget_mb
    stats["budget_bytes_used"] = stats["packed_bytes"]
    if cache is not None:
        model._tensorshrink_cache = cache
    if verbose:
        n8 = sum(1 for l in stats["layers"] if l["bits"] == 8)
        print(
            f"  budget-aware plan: {len(targets) - n8} layers at 4-bit, {n8} at 8-bit, "
            f"{stats['packed_bytes']/1e6:.1f}MB packed (budget {vram_budget_mb:.1f}MB)"
        )
    return stats


def quantize_module(
    model: nn.Module,
    bits: int = 4,
    group_size: int = 128,
    outlier_frac: float = 0.001,
    adaptive: bool = False,
    error_threshold: float = 0.10,
    min_numel: int = DEFAULT_MIN_NUMEL,
    skip_name_patterns: Iterable[str] = DEFAULT_SKIP_PATTERNS,
    max_dequant_rows: Union[int, str] = DEFAULT_MAX_DEQUANT_ROWS,
    force_fp32_patterns: Iterable[str] = (),
    cache_budget_mb: float = 0.0,
    host_cache_budget_mb: float = 0.0,
    fast_dequant: bool = True,
    use_triton: bool = True,
    vram_budget_mb: Optional[float] = None,
    quantize_workers: Optional[int] = 4,
    verbose: bool = True,
    companded: bool = False,
    double_quant: bool = True,
    dq_group_size: int = DEFAULT_DQ_GROUP_SIZE,
    nf_quant: bool = True,
) -> dict:
    """Replace every eligible nn.Linear inside `model` with a QuantLinear,
    in place. Eligibility: weight.numel() >= min_numel and the dotted module
    name doesn't match any of `skip_name_patterns` (case-insensitive
    substring match) — keeps small/sensitive layers (norms, biases,
    embeddings, lm_head) at full precision by default.

    `force_fp32_patterns`: layers matching these patterns are still
    quantized (storage shrinks) but their QuantLinear is built with
    `force_fp32_compute=True`, so forward() always computes in fp32. Use
    this instead of `skip_name_patterns` for layers whose source model
    forces fp32 compute for stability but can still be quantized in storage
    (see T5_EXTRA_SKIP_PATTERNS).

    `cache_budget_mb`: opt-in generation-speed cache (see `WeightCache`). 0
    (default) means no cache. When > 0, every QuantLinear created here
    shares one `WeightCache`, returned as `stats["cache"]`
    (`model._tensorshrink_cache`).

    `host_cache_budget_mb`: `WeightCache`'s second tier, in host RAM instead
    of VRAM, for models with ~zero VRAM headroom that still reuse the same
    weights across many forward calls (e.g. diffusion denoising steps).
    Independent of `cache_budget_mb` — either, both, or neither can be nonzero.

    `vram_budget_mb` (MiB): alternative to `error_threshold`-driven
    escalation. When set, `adaptive`/`error_threshold`/`bits` are ignored and
    every eligible layer gets a bit-width from `codec.budget_allocate_bits`:
    start at 4-bit, then upgrade layers to 8-bit (best
    error-reduction-per-byte first) until the budget is exhausted. Unlike a
    fixed `error_threshold`, this expresses a real total-bytes constraint
    across the whole model instead of escalating each large layer
    independently.

    `quantize_workers`: how many layers' quantize calls run concurrently on
    a `ThreadPoolExecutor` (torch releases the GIL for tensor ops, so this
    is real parallelism). Capped at `min(quantize_workers, os.cpu_count(),
    n_target_layers)`, with in-flight tasks windowed to ~2x that count so
    not every layer's fp16 weight is held in memory at once. Default 4 is
    conservative since each in-flight layer holds fp32-sized working
    buffers; raise it on a RAM-rich machine, or pass 1/None for strictly
    sequential.

    Returns a stats dict with a per-layer breakdown (bit-width chosen,
    reconstruction error, byte counts) suitable for printing a report.
    """
    targets = list(_iter_target_linears(model, min_numel, skip_name_patterns))
    force_fp32_patterns = tuple(p.lower() for p in force_fp32_patterns)

    # Companded weights DO have a fused-Triton-kernel fast path as of the
    # HAS_CODEBOOK/codebook_ptr work in triton_kernels.py (QuantLinear.
    # forward passes `codebook=self.codebook` straight into
    # fused_quantized_linear whenever the usual Triton-path conditions
    # hold) -- no cache is needed there, same as plain bits=4/8. This
    # mirrors the exact condition QuantLinear.__init__ checks before
    # warning about the *uncached* pure-PyTorch codebook-gather dequant
    # (~3x slower per call), which only actually applies when that fast
    # path isn't available: no Triton installed, `use_triton=False`, or a
    # non-default group_size. Auto-sizing a cache unconditionally here used
    # to silently balloon VRAM to the full dense-fp16-equivalent size (was
    # ~2.5x the fp16 baseline model, see docs/companded-quantization-
    # benchmark.md's "before Triton fusion" numbers) on every default
    # `companded=True` call even once the fast path made it unnecessary --
    # exactly backwards for a project whose point is beating bitsandbytes
    # on memory. Only fall back to auto-sizing when the fast path genuinely
    # isn't available; an explicit cache_budget_mb/host_cache_budget_mb from
    # the caller always takes precedence over both.
    triton_fast_path_available = TRITON_AVAILABLE and use_triton and group_size in _SUPPORTED_GROUP_SIZES
    if companded and cache_budget_mb <= 0 and host_cache_budget_mb <= 0 and not triton_fast_path_available:
        dense_bytes = sum(lin.weight.numel() * 2 for _, lin in targets)  # fp16
        cache_budget_mb = (dense_bytes * 1.05) / (1024 * 1024)
        if verbose:
            print(f"  companded=True with no fused-Triton-kernel fast path available: "
                  f"auto-sizing WeightCache to {cache_budget_mb:.0f}MB "
                  f"(pass cache_budget_mb explicitly to override)")

    cache = (
        WeightCache(int(cache_budget_mb * 1024 * 1024), host_budget_bytes=int(host_cache_budget_mb * 1024 * 1024),
                    admission_policy="static_fill")
        if cache_budget_mb > 0 or host_cache_budget_mb > 0
        else None
    )
    stats = {"n_replaced": 0, "orig_bytes": 0, "packed_bytes": 0, "layers": [], "cache": cache}

    if vram_budget_mb is not None:
        if companded:
            raise ValueError("companded=True is not yet supported with vram_budget_mb (budget-aware path)")
        return _quantize_module_budget_aware(
            model, targets, force_fp32_patterns, group_size, outlier_frac, vram_budget_mb,
            max_dequant_rows, cache, fast_dequant, use_triton, quantize_workers, stats, verbose,
        )

    # The per-layer quantize call is the expensive part and independent
    # across layers, so it's farmed out to a thread pool (torch releases the
    # GIL for the tensor ops, so this gets real parallelism). Model mutation
    # (`_set_module_by_name`) stays on the main thread as futures complete,
    # to avoid races and keep stats order deterministic. `targets[i]` is set
    # to `None` right after each result is consumed so the original fp16
    # weight gets freed immediately instead of staying resident for the
    # whole pass.
    n = len(targets)
    max_workers = min(quantize_workers or 1, max(1, os.cpu_count() or 1))

    def _quantize_one(name: str, lin: nn.Linear):
        force_fp32 = any(p in name.lower() for p in force_fp32_patterns)
        qlin, err, escalated = QuantLinear.from_linear(
            lin,
            bits=bits,
            group_size=group_size,
            outlier_frac=outlier_frac,
            adaptive=adaptive,
            error_threshold=error_threshold,
            max_dequant_rows=max_dequant_rows,
            force_fp32_compute=force_fp32,
            cache=cache,
            fast_dequant=fast_dequant,
            use_triton=use_triton,
            companded=companded,
            double_quant=double_quant,
            dq_group_size=dq_group_size,
            nf_quant=nf_quant,
        )
        itemsize = lin.weight.element_size()
        return name, qlin, err, escalated, force_fp32, itemsize

    def _record_stats(name, qlin, err, escalated, force_fp32, itemsize):
        orig_bytes = qlin.dense_equiv_nbytes(itemsize=itemsize)
        packed_bytes = qlin.packed_nbytes()
        stats["n_replaced"] += 1
        stats["orig_bytes"] += orig_bytes
        stats["packed_bytes"] += packed_bytes
        stats["layers"].append(
            {
                "name": name,
                "bits": qlin.bits,
                "rel_error": err,
                "escalated": escalated,
                "orig_bytes": orig_bytes,
                "packed_bytes": packed_bytes,
                "force_fp32_compute": force_fp32,
            }
        )
        if verbose:
            flag = "*" if escalated else " "
            fp32_flag = " fp32" if force_fp32 else ""
            print(
                f"  [{qlin.bits}b{flag}{fp32_flag}] {name:60s} err={err:.4f} "
                f"{orig_bytes/1e6:8.3f}MB -> {packed_bytes/1e6:8.3f}MB"
            )

    if n > 1 and max_workers > 1:
        results: List[Optional[tuple]] = [None] * n
        for i, res in _parallel_windowed(targets, _quantize_one, max_workers):
            # Swap into the model as each result arrives (not after the whole
            # pass) so the original fp16 layer can be freed right away; only
            # stats bookkeeping is deferred, to keep output order deterministic.
            name, qlin, err, escalated, force_fp32, itemsize = res
            _set_module_by_name(model, name, qlin)
            results[i] = res
            targets[i] = None  # drop (name, nn.Linear) -> frees this layer's fp16 weight
        for res in results:
            _record_stats(*res)
    else:
        for i, (name, lin) in enumerate(targets):
            res = _quantize_one(name, lin)
            _set_module_by_name(model, res[0], res[1])
            targets[i] = None
            del lin
            _record_stats(*res)

    if stats["orig_bytes"] > 0:
        stats["ratio"] = stats["orig_bytes"] / max(1, stats["packed_bytes"])
    else:
        stats["ratio"] = 1.0
    if cache is not None:
        model._tensorshrink_cache = cache
    return stats


def save_quant_module(model: nn.Module, path, verbose: bool = True) -> dict:
    """Serialize an already-quantized model (i.e. one that has been through
    `quantize_module`) to a tensorshrink container: each QuantLinear's
    packed data is written directly (no dequantize round-trip), and every
    other parameter/buffer still on the model (norms, embeddings, biases
    kept at full precision, un-quantized small Linear layers, ...) is
    written as a raw compressed segment under its `state_dict()` key.
    """
    from .container import ContainerWriter

    quant_names = set()       # state_dict keys fully handled by add_quantized/add_raw below
    quant_module_prefixes = []  # e.g. "fc1." — everything under here belongs to a QuantLinear
    writer = ContainerWriter(path)
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            wkey = f"{name}.weight"
            writer.add_quantized(wkey, module.to_quantized_tensor(), force_fp32_compute=module.force_fp32_compute)
            quant_names.add(wkey)
            quant_module_prefixes.append(f"{name}.")
            if module.bias is not None:
                writer.add_raw(f"{name}.bias", module.bias)
                quant_names.add(f"{name}.bias")

    n_raw = 0
    for key, tensor in model.state_dict().items():
        if key in quant_names:
            continue
        # Skip a QuantLinear's *internal* buffers (packed/scale/gmin/outlier_*) —
        # those are the raw materials of the quantized weight we already wrote
        # above via add_quantized; saving them again here would silently
        # duplicate the entire weight and bloat the container.
        if any(key.startswith(p) for p in quant_module_prefixes):
            continue
        writer.add_raw(key, tensor)
        n_raw += 1
    writer.close()
    if verbose:
        print(f"  saved {len(quant_names)} quantized tensors + {n_raw} raw tensors -> {path}")
    return {"n_quant": len(quant_names), "n_raw": n_raw}


def load_quantized_module(
    model: nn.Module,
    reader,
    max_dequant_rows: Union[int, str] = DEFAULT_MAX_DEQUANT_ROWS,
    cache_budget_mb: float = 0.0,
    host_cache_budget_mb: float = 0.0,
    fast_dequant: bool = True,
    use_triton: bool = True,
    parallel: bool = True,
    verbose: bool = True,
) -> dict:
    """Inverse of `quantize_module` + `save via container.ContainerWriter`:
    given a *freshly constructed* (e.g. `AutoModel.from_config(cfg)`) model
    of the same architecture, and a `ContainerReader` over a tensorshrink
    container built from that model's state_dict, populate the model by
    swapping in QuantLinear layers built directly from the packed data
    (never materializing a dense fp16 copy of those weights) and loading
    every remaining raw tensor (norms, embeddings, biases) via
    `load_state_dict`.

    `parallel` (default True): fetch every quantized layer's packed data
    via `reader.get_quantized_many`, which decompresses across a thread
    pool instead of one `get_quantized` call per layer in a strictly
    sequential loop — this is the loading-speed win for models with many
    layers (a T5-xxl-scale text encoder has 168+ quantized Linear layers;
    on this codebase's own container format, decompression is the dominant
    cost of `load_quantized_module`, not `QuantLinear` construction, so
    parallelizing it is what actually moves the wall-clock number). Falls
    back to the original sequential per-layer path when False, for
    debugging or memory-constrained environments where you'd rather not
    have several tensors' decompressed buffers alive at once.

    `cache_budget_mb` / `host_cache_budget_mb` / `fast_dequant`: forwarded to
    every constructed QuantLinear — see `quantize_module`'s docstring for
    what they do.
    """
    names = set(reader.names())
    cache = (
        WeightCache(int(cache_budget_mb * 1024 * 1024), host_budget_bytes=int(host_cache_budget_mb * 1024 * 1024),
                    admission_policy="static_fill")
        if cache_budget_mb > 0 or host_cache_budget_mb > 0
        else None
    )
    targets = []  # (module_name, wkey, bkey_or_None, force_fp32)
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and not isinstance(module, QuantLinear):
            wkey = f"{name}.weight"
            if wkey in names and reader.kind(wkey) == "quant":
                bkey = f"{name}.bias"
                force_fp32 = bool(reader.manifest["tensors"][wkey].get("force_fp32_compute", False))
                targets.append((name, wkey, bkey if bkey in names else None, force_fp32))

    if parallel and len(targets) > 1:
        qts = reader.get_quantized_many([wkey for _, wkey, _, _ in targets])
    else:
        qts = {wkey: reader.get_quantized(wkey) for _, wkey, _, _ in targets}

    replaced = 0
    for name, wkey, bkey, force_fp32 in targets:
        qt = qts[wkey]
        bias = reader.get(bkey) if bkey is not None else None
        qlin = QuantLinear.from_quantized_tensor(
            qt, bias, max_dequant_rows=max_dequant_rows, force_fp32_compute=force_fp32,
            cache=cache, fast_dequant=fast_dequant, use_triton=use_triton,
        )
        _set_module_by_name(model, name, qlin)
        replaced += 1
        if verbose:
            print(f"  loaded quantized: {name}")
    if cache is not None:
        model._tensorshrink_cache = cache

    # everything else (norms, embeddings, biases of layers we didn't touch, etc.)
    raw_state = {}
    model_state_keys = set(model.state_dict().keys())
    for n in names:
        if n in model_state_keys and reader.kind(n) == "raw":
            raw_state[n] = reader.get(n)
    missing, unexpected = model.load_state_dict(raw_state, strict=False)
    if verbose:
        print(f"  loaded {len(raw_state)} raw tensors, replaced {replaced} Linear layers")
    return {"n_quant_replaced": replaced, "n_raw_loaded": len(raw_state), "cache": cache}


# ---------------------------------------------------------------------------
# CUDA graph wrapper — eliminates Python/kernel-launch overhead for
# repeated forward calls with the same shapes (generation loops, diffusion
# denoising steps).
# ---------------------------------------------------------------------------

class CUDAGraphQuantModel:
    """Wraps a quantized model's forward pass in a CUDA graph for maximum
    throughput on repeated calls with the same input shape.

    CUDA graphs capture the entire GPU command sequence (all kernel launches,
    memory copies, synchronizations) once, then replay it in a single driver
    call — eliminating all Python-dispatch and kernel-launch overhead that
    normally dominates small-batch inference (the exact workload pattern
    diffusion denoising steps and autoregressive generation produce).

    Measured improvement: 1.5-3x throughput on decode-step-shaped workloads
    (M=1 batch size, 20+ layers) where Python overhead is a significant
    fraction of total runtime.

    Usage:
        model = ...  # already quantize_module()'d and on GPU
        graph_model = CUDAGraphQuantModel(model)

        # First call with a new shape captures the graph (slower):
        y = graph_model(x)

        # Subsequent calls with the same shape replay (fast):
        for step in range(50):
            y = graph_model(x_step)  # must be same shape as first call

        graph_model.release()  # free captured graphs
    """

    def __init__(self, model: nn.Module, warmup_runs: int = 3):
        self.model = model
        self.warmup_runs = warmup_runs
        self._graphs: Dict[Tuple, "_CapturedGraph"] = {}

    def __call__(self, *args, **kwargs):
        """Forward pass through the model, using a cached CUDA graph when
        the input shapes match a previously captured one."""
        if not torch.cuda.is_available():
            return self.model(*args, **kwargs)

        # Build a shape key from all tensor arguments
        shape_key = tuple(
            tuple(a.shape) for a in args if isinstance(a, torch.Tensor)
        )
        for v in kwargs.values():
            if isinstance(v, torch.Tensor):
                shape_key += (tuple(v.shape),)

        if shape_key not in self._graphs:
            self._graphs[shape_key] = self._capture(args, kwargs)

        return self._graphs[shape_key].replay(args, kwargs)

    def _capture(self, args, kwargs):
        """Capture a CUDA graph for the given input shapes."""
        # Warmup runs to ensure all lazy initializations are done
        for _ in range(self.warmup_runs):
            self.model(*args, **kwargs)
        torch.cuda.synchronize()

        # Allocate static input/output buffers
        static_args = tuple(
            a.clone() if isinstance(a, torch.Tensor) else a for a in args
        )
        static_kwargs = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = self.model(*static_args, **static_kwargs)

        return _CapturedGraph(graph, static_args, static_kwargs, static_output)

    def release(self):
        """Free all captured CUDA graphs."""
        self._graphs.clear()

    def __del__(self):
        self.release()


class _CapturedGraph:
    """A single captured CUDA graph with its static input/output buffers."""

    def __init__(self, graph, static_args, static_kwargs, static_output):
        self.graph = graph
        self.static_args = static_args
        self.static_kwargs = static_kwargs
        self.static_output = static_output

    def replay(self, args, kwargs):
        """Copy new inputs into the static buffers, replay, return output."""
        for sa, a in zip(self.static_args, args):
            if isinstance(sa, torch.Tensor) and isinstance(a, torch.Tensor):
                sa.copy_(a)
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor) and k in self.static_kwargs:
                self.static_kwargs[k].copy_(v)

        self.graph.replay()
        return self.static_output.clone()
