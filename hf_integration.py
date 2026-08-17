"""
tensorshrink.hf_integration
=============================

A real `transformers` `HfQuantizer` plugin — the same extension point every
built-in backend (bitsandbytes, GPTQ, AWQ, HQQ, ...) uses — so tensorshrink
can be selected exactly like bitsandbytes is:

    from transformers import AutoModelForCausalLM
    from tensorshrink import TensorShrinkConfig

    model = AutoModelForCausalLM.from_pretrained(
        "some/model", quantization_config=TensorShrinkConfig(adaptive=True)
    )

Without this, using tensorshrink meant loading the full dense fp16 model
into host RAM first, then calling `quantize_module()` on it. bitsandbytes
avoids that by hooking `from_pretrained`'s state-dict loader and quantizing
each tensor as it's read off disk. This module does the same:
`TensorShrinkHfQuantizer` implements `requires_parameters_quantization`, so
`transformers` calls `create_quantized_param(...)` once per tensor straight
from the checkpoint shard reader, building each `QuantLinear` as the
checkpoint streams in.

`bits` (fixed) and `adaptive`/`error_threshold` modes are supported — both
can decide a bit-width from a single tensor with no cross-layer info.
`vram_budget_mb` is not supported here since that allocator needs every
layer's cost up front; use `stream_quantize_safetensors_dir` for that case
instead (`vram_budget_mb` here raises `NotImplementedError` pointing to it).

`save_pretrained` round-tripping is also not implemented — persistence goes
through `save_quant_module`/`load_quantized_module` instead, and
`is_serializable` is `False` here to make that explicit.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

_TRANSFORMERS_HOOKED = False
_HOOK_ERROR: Optional[str] = None


class _ImmediateFuture:
    """Duck-types `concurrent.futures.Future`'s `.result()` for an
    already-computed value, so `_drain_one` (written against real Futures
    for the threaded path) works unchanged when `quantize_workers<=1` and
    there's no executor to submit to."""

    __slots__ = ("_value",)

    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class TensorShrinkQuantizationMethod(str, Enum):
    """Mirrors `transformers.utils.quantization_config.QuantizationMethod`'s
    own `str, Enum` shape (every third-party quantizer that predates
    `transformers`' `register_quantization_config` helper does the same,
    since `AutoHfQuantizer` reads `quantization_config.quant_method.value`
    directly) -- tensorshrink just isn't a member of transformers' own
    closed enum, so it needs its own with the same interface.
    """

    TENSORSHRINK = "tensorshrink"


class TensorShrinkConfig:
    """`quantization_config=` value for `transformers.AutoModel*.from_pretrained`.

    Mirrors `quantize_module`'s own parameters (see that function's
    docstring in `tensorshrink/modules.py` for what each one does) --
    this is the same codec, just invoked per-tensor as the checkpoint
    streams in instead of on an already-fully-loaded model.
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        outlier_frac: float = 0.001,
        adaptive: bool = False,
        error_threshold: float = 0.10,
        min_numel: Optional[int] = None,
        skip_name_patterns: Optional[Tuple[str, ...]] = None,
        force_fp32_patterns: Tuple[str, ...] = (),
        max_dequant_rows=None,
        cache_budget_mb: float = 0.0,
        host_cache_budget_mb: float = 0.0,
        fast_dequant: bool = True,
        use_triton: bool = True,
        vram_budget_mb: Optional[float] = None,
        quantize_workers: Optional[int] = 4,
        verbose: bool = True,
        companded: bool = False,
        double_quant: bool = True,
        dq_group_size: Optional[int] = None,
        nf_quant: bool = True,
        **kwargs,
    ):
        from .modules import DEFAULT_MIN_NUMEL, DEFAULT_SKIP_PATTERNS, DEFAULT_MAX_DEQUANT_ROWS
        from .codec import DEFAULT_DQ_GROUP_SIZE

        _install()  # registers with transformers' quantizer dispatch tables, once
        self.quant_method = TensorShrinkQuantizationMethod.TENSORSHRINK
        self.bits = bits
        self.group_size = group_size
        self.outlier_frac = outlier_frac
        self.adaptive = adaptive
        self.error_threshold = error_threshold
        self.min_numel = DEFAULT_MIN_NUMEL if min_numel is None else min_numel
        self.skip_name_patterns = DEFAULT_SKIP_PATTERNS if skip_name_patterns is None else tuple(skip_name_patterns)
        self.force_fp32_patterns = tuple(p.lower() for p in force_fp32_patterns)
        self.max_dequant_rows = DEFAULT_MAX_DEQUANT_ROWS if max_dequant_rows is None else max_dequant_rows
        self.cache_budget_mb = cache_budget_mb
        self.host_cache_budget_mb = host_cache_budget_mb
        self.fast_dequant = fast_dequant
        self.use_triton = use_triton
        self.vram_budget_mb = vram_budget_mb
        self.quantize_workers = quantize_workers
        self.verbose = verbose
        self.companded = companded
        self.double_quant = double_quant
        self.dq_group_size = DEFAULT_DQ_GROUP_SIZE if dq_group_size is None else dq_group_size
        self.nf_quant = nf_quant
        self.modules_to_not_convert = kwargs.pop("modules_to_not_convert", None)
        self.post_init()

    def post_init(self):
        if self.bits not in (4, 8):
            raise ValueError(f"TensorShrinkConfig.bits must be 4 or 8, got {self.bits}")
        if self.group_size <= 0:
            raise ValueError(f"TensorShrinkConfig.group_size must be positive, got {self.group_size}")
        if self.vram_budget_mb is not None:
            raise NotImplementedError(
                "TensorShrinkConfig.vram_budget_mb isn't supported through `from_pretrained` yet -- "
                "the budget-aware bit allocator needs every eligible layer's byte/error cost before "
                "picking any single layer's bit-width, which needs a real look at the checkpoint's "
                "on-disk shards, not just a single streaming pass. Use "
                "`tensorshrink.stream_quantize_safetensors_dir(model_dir, out_path, vram_budget_mb=...)` "
                "+ `tensorshrink.load_quantized_module(...)` for that case instead (see README)."
            )

    # -- QuantizationConfigMixin-compatible surface (duck-typed rather than
    # subclassed: subclassing pulls in a dataclass field-ordering fight over
    # `quant_method` having no default in the base class; every method
    # `transformers` actually calls on a quantization_config -- to_dict,
    # to_json_string, update, __repr__ -- is small enough to reimplement
    # directly and keep this file's dependency on transformers' internals
    # to the documented HfQuantizer contract only.) ------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["quant_method"] = self.quant_method.value
        return d

    def to_diff_dict(self) -> Dict[str, Any]:
        return self.to_dict()

    def to_json_string(self, use_diff: bool = True) -> str:
        import json

        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def __iter__(self):
        for k, v in self.to_dict().items():
            yield k, v

    def __repr__(self) -> str:
        return f"TensorShrinkConfig {self.to_json_string()}"

    def update(self, **kwargs) -> Dict[str, Any]:
        unused = {}
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                unused[k] = v
        return unused

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], return_unused_kwargs: bool = False, **kwargs):
        config_dict = dict(config_dict)
        config_dict.pop("quant_method", None)
        config = cls(**config_dict)
        unused = config.update(**kwargs)
        return (config, unused) if return_unused_kwargs else config


def _install():
    """Registers `TensorShrinkConfig`/`TensorShrinkHfQuantizer` into
    `transformers`' quantizer dispatch tables. Called once, lazily, the
    first time `TensorShrinkConfig` is actually needed (see
    `tensorshrink/__init__.py`) rather than unconditionally at
    `import tensorshrink` time, so a `transformers`-less install of
    tensorshrink (the codec/container/modules pieces have no such
    dependency) never pays an import cost or risks an incompatible-version
    error for a feature it isn't using.
    """
    global _TRANSFORMERS_HOOKED, _HOOK_ERROR
    if _TRANSFORMERS_HOOKED:
        return
    try:
        import transformers.quantizers.auto as auto_mod
        from transformers.quantizers.base import HfQuantizer
        from transformers.quantizers.quantizers_utils import get_module_from_name
    except Exception as exc:  # pragma: no cover - exercised only without transformers installed
        _HOOK_ERROR = (
            f"tensorshrink's transformers integration requires `transformers` (and `accelerate`) "
            f"to be installed: {exc!r}"
        )
        raise ImportError(_HOOK_ERROR) from exc

    class TensorShrinkHfQuantizer(HfQuantizer):
        requires_parameters_quantization = True
        requires_calibration = False
        required_packages = None

        def __init__(self, quantization_config: TensorShrinkConfig, **kwargs):
            super().__init__(quantization_config, **kwargs)
            self._target_weight_keys: set = set()
            self._pending_bias: Dict[str, Optional[torch.Tensor]] = {}
            self._cache = None
            self._n_replaced = 0
            self._orig_bytes = 0
            self._packed_bytes = 0
            # Pipelined quantize: transformers calls create_quantized_param
            # synchronously, one tensor at a time, with no batching hook — but
            # each tensor is already fully in memory, so only the CPU quantize
            # math needs parallelizing. Jobs go to a bounded thread pool;
            # once the window fills, the oldest outstanding job is drained.
            self._executor: Optional[ThreadPoolExecutor] = None
            self._pending: "OrderedDict[str, Tuple[Any, int]]" = OrderedDict()
            self._window = 1
            # Tracks how many target keys have been submitted so the last one
            # can force a synchronous drain of everything still pending — this
            # transformers version calls dispatch_model *before* our
            # after-weight-loading hook, so anything left un-drained would
            # still be on the meta device when dispatch_model runs.
            self._n_target_weight_keys_submitted = 0

        def validate_environment(self, *args, **kwargs):
            try:
                import accelerate  # noqa: F401
            except ImportError:
                raise ImportError(
                    "tensorshrink's `from_pretrained(quantization_config=TensorShrinkConfig(...))` "
                    "integration requires `accelerate` (for meta-device model construction): "
                    "`pip install accelerate`"
                )

        def update_torch_dtype(self, torch_dtype: Optional["torch.dtype"]) -> "torch.dtype":
            # Default checkpoint-read dtype to fp16 instead of fp32 to halve
            # the bytes moved per tensor before quantization even sees it.
            return torch.float16 if torch_dtype is None else torch_dtype

        def _process_model_before_weight_loading(self, model: "nn.Module", **kwargs):
            from .modules import _iter_target_linears, WeightCache

            cfg = self.quantization_config
            skip_patterns = cfg.skip_name_patterns
            if cfg.modules_to_not_convert:
                skip_patterns = tuple(skip_patterns) + tuple(cfg.modules_to_not_convert)
            # Safe on a meta-device model: numel() only reads shape metadata.
            # Same eligibility rule as quantize_module (min_numel + skip
            # patterns), so both paths always agree on the same model+config.
            self._target_weight_keys = {
                f"{name}.weight" for name, _ in _iter_target_linears(model, cfg.min_numel, skip_patterns)
            }
            if cfg.cache_budget_mb > 0 or cfg.host_cache_budget_mb > 0:
                self._cache = WeightCache(
                    int(cfg.cache_budget_mb * 1024 * 1024),
                    host_budget_bytes=int(cfg.host_cache_budget_mb * 1024 * 1024),
                    admission_policy="static_fill",
                )
            import os as _os

            n_workers = min(cfg.quantize_workers or 1, max(1, _os.cpu_count() or 1))
            if n_workers > 1:
                self._executor = ThreadPoolExecutor(max_workers=n_workers)
                self._window = n_workers * 2
            self._model_ref = model
            model.config.quantization_config = cfg
            return model

        def _submit_quantize(self, module_name: str, w: torch.Tensor, target_device):
            """Kick off `module_name`'s quantize job on the thread pool and,
            once `self._window` jobs are outstanding, drain the oldest one
            first — so at most ~2x quantize_workers dense weights are alive
            at once, not every layer submitted so far."""
            from .codec import quantize_tensor, quantize_tensor_adaptive

            cfg = self.quantization_config

            def _job():
                if cfg.adaptive:
                    # (2, 4, 8): what QuantLinear's forward-pass dequant
                    # supports (6-bit RVQ excluded, see modules.py).
                    return quantize_tensor_adaptive(
                        w, group_size=cfg.group_size, outlier_frac=cfg.outlier_frac,
                        error_threshold=cfg.error_threshold, allowed_bits=(2, 4, 8),
                        companded=cfg.companded,
                        double_quant=cfg.double_quant, dq_group_size=cfg.dq_group_size,
                        nf_quant=cfg.nf_quant,
                    )
                qt = quantize_tensor(w, bits=cfg.bits, group_size=cfg.group_size, outlier_frac=cfg.outlier_frac,
                                      companded=cfg.companded,
                                      double_quant=cfg.double_quant, dq_group_size=cfg.dq_group_size,
                                      nf_quant=cfg.nf_quant)
                return qt, None, False

            future = self._executor.submit(_job)
            self._pending[module_name] = (future, w.element_size(), target_device)
            self._n_target_weight_keys_submitted += 1
            while len(self._pending) >= self._window:
                oldest_name = next(iter(self._pending.keys()))
                self._drain_one(oldest_name)
            # Last target weight key: force a synchronous drain now rather
            # than leaving stragglers un-built until after dispatch_model
            # runs, which would crash with "Cannot copy out of meta tensor".
            if self._n_target_weight_keys_submitted >= len(self._target_weight_keys):
                self._drain_all()

        def _drain_one(self, module_name: str):
            """Block on `module_name`'s in-flight quantize job (if any) and
            finish building/swapping in its QuantLinear. A no-op if the
            layer has already been drained."""
            from .modules import QuantLinear, _set_module_by_name

            entry = self._pending.pop(module_name, None)
            if entry is None:
                return
            future, itemsize, target_device = entry
            qt, err, escalated = future.result()
            cfg = self.quantization_config
            force_fp32 = any(p in module_name.lower() for p in cfg.force_fp32_patterns)
            bias = self._pending_bias.pop(module_name, None)
            qlin = QuantLinear(
                qt, bias, max_dequant_rows=cfg.max_dequant_rows, force_fp32_compute=force_fp32,
                cache=self._cache, fast_dequant=cfg.fast_dequant, use_triton=cfg.use_triton,
            ).to(target_device)
            _set_module_by_name(self._model_ref, module_name, qlin)
            self._n_replaced += 1
            self._orig_bytes += qlin.dense_equiv_nbytes(itemsize=itemsize)
            self._packed_bytes += qlin.packed_nbytes()
            if cfg.verbose:
                tag = f"{qlin.bits}b" + ("*" if escalated else "")
                err_s = f" err={err:.4f}" if err is not None else ""
                print(f"  [{tag}] {module_name:60s}{err_s}")

        def _drain_all(self):
            for name in list(self._pending.keys()):
                self._drain_one(name)
            if self._executor is not None:
                self._executor.shutdown(wait=True)

        def check_quantized_param(
            self, model, param_value, param_name, state_dict, **kwargs
        ) -> bool:
            if param_name in self._target_weight_keys:
                return True
            if param_name.endswith(".bias"):
                weight_key = param_name[: -len("bias")] + "weight"
                return weight_key in self._target_weight_keys
            return False

        def create_quantized_param(
            self,
            model,
            param_value: torch.Tensor,
            param_name: str,
            target_device,
            state_dict: Dict[str, Any],
            unexpected_keys: Optional[List[str]] = None,
        ):
            from .modules import QuantLinear

            if param_name.endswith(".bias"):
                module_name = param_name[: -len(".bias")]
                bias_t = None if param_value is None else param_value.detach().to(target_device)
                module, _ = get_module_from_name(model, param_name)
                if isinstance(module, QuantLinear):
                    module.bias = torch.nn.Parameter(bias_t, requires_grad=False) if bias_t is not None else None
                else:
                    # Weight for this layer hasn't been swapped in yet --
                    # either not-yet-drained (in flight or still queued
                    # behind `self._window`) or the checkpoint happened to
                    # order `bias` before `weight` (`nn.Linear.state_dict()`
                    # never does, but a hand-built checkpoint theoretically
                    # could). Either way, stash it; `_drain_one` consumes it
                    # once that layer's QuantLinear is actually built.
                    self._pending_bias[module_name] = bias_t
                return

            module_name = param_name[: -len(".weight")]
            w = param_value.detach()
            if self._executor is not None:
                self._submit_quantize(module_name, w, target_device)
            else:
                # quantize_workers<=1 (or single logical core): quantize and
                # apply inline, same as the threaded path's `_drain_one` just
                # without the Future indirection.
                from .codec import quantize_tensor, quantize_tensor_adaptive

                cfg = self.quantization_config
                if cfg.adaptive:
                    # allowed_bits=(2, 4, 8): see the matching comment in _job() above.
                    qt, err, escalated = quantize_tensor_adaptive(
                        w, group_size=cfg.group_size, outlier_frac=cfg.outlier_frac,
                        error_threshold=cfg.error_threshold, allowed_bits=(2, 4, 8),
                        companded=cfg.companded,
                        double_quant=cfg.double_quant, dq_group_size=cfg.dq_group_size,
                        nf_quant=cfg.nf_quant,
                    )
                else:
                    qt = quantize_tensor(w, bits=cfg.bits, group_size=cfg.group_size, outlier_frac=cfg.outlier_frac,
                                          companded=cfg.companded,
                                          double_quant=cfg.double_quant, dq_group_size=cfg.dq_group_size,
                                          nf_quant=cfg.nf_quant)
                    err, escalated = None, False
                self._pending[module_name] = (_ImmediateFuture((qt, err, escalated)), w.element_size(), target_device)
                self._drain_one(module_name)
            del param_value, w

        def _process_model_after_weight_loading(self, model, **kwargs):
            self._drain_all()
            model.is_tensorshrink_quantized = True
            if self._cache is not None:
                model._tensorshrink_cache = self._cache
            if self.quantization_config.verbose and self._orig_bytes > 0:
                ratio = self._orig_bytes / max(1, self._packed_bytes)
                print(
                    f"  tensorshrink: replaced {self._n_replaced} layers, "
                    f"{self._orig_bytes/1e6:.1f}MB -> {self._packed_bytes/1e6:.1f}MB ({ratio:.2f}x)"
                )
            return model

        @property
        def is_serializable(self) -> bool:
            # `save_pretrained` on a model quantized through this path would
            # write ordinary `packed`/`scale`/`gmin`/`outlier_*` buffers as if
            # they were plain state_dict tensors -- `from_pretrained` could
            # reload the shapes but has no `create_quantized_param` branch
            # that reconstructs a QuantLinear *from* those buffers (only from
            # a fresh dense `.weight`). Use `save_quant_module` /
            # `load_quantized_module` (the tensorshrink container format) for
            # persistence instead -- explicitly unsupported here rather than
            # silently producing a checkpoint that can't be reloaded.
            return False

        @property
        def is_trainable(self) -> bool:
            return False

    auto_mod.AUTO_QUANTIZATION_CONFIG_MAPPING[TensorShrinkQuantizationMethod.TENSORSHRINK.value] = TensorShrinkConfig
    auto_mod.AUTO_QUANTIZER_MAPPING[TensorShrinkQuantizationMethod.TENSORSHRINK.value] = TensorShrinkHfQuantizer
    _TRANSFORMERS_HOOKED = True
