"""
tensorshrink.streaming
========================

`model.to("cuda")` normally moves every `QuantLinear`'s packed buffers to
the GPU at once — the whole quantized model has to fit in VRAM. For a model
whose packed size alone exceeds the card (e.g. FLUX's transformer at
13.66GB packed on an 8GB card), that's a hard wall even after 4x/2x
quantization already shrank it.

`stream_offload_module` keeps every `QuantLinear`'s packed buffers pinned in
host RAM and streams only the layer(s) about to run onto the GPU, freeing
each layer's GPU buffers immediately after its forward() call returns. A
second CUDA stream prefetches the next layer's packed bytes while the
current layer computes, so PCIe transfer overlaps compute instead of
stalling it — the same "double buffering" pattern `accelerate`'s
`disk_offload`/`cpu_offload` use, except what's transferred here is already
2-4x smaller (packed bytes, not dequantized fp16 weights), which is a
structural transfer-size advantage over bitsandbytes+accelerate offload for
the same layer over the same PCIe link.

This intentionally only ever touches `QuantLinear` buffers (`packed`,
`scale`, `gmin`, `outlier_*`) — everything else on the model (norms,
embeddings, small un-quantized layers) is assumed small enough to live on
GPU permanently, same as today.

Usage:

    model = ...                       # already quantize_module()'d
    streamer = stream_offload_module(model, vram_budget_mb=2000)
    with torch.no_grad():
        y = model(x)                  # QuantLinear buffers stream in/out on demand
    streamer.disable()                # optional: remove hooks, restore normal behavior
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

import torch

from .modules import QuantLinear

# The full set of a QuantLinear's buffers that scale with layer size (the
# "packed weight" itself). `bias` is deliberately excluded — it's tiny
# relative to `packed`/`scale`/`gmin` for any layer big enough to bother
# quantizing, so it's kept GPU-resident permanently instead of adding a
# stream/free cycle for negligible VRAM savings.
# NOTE: "outlier_val" is intentionally absent -- QuantLinear no longer keeps
# it as a resident buffer (its value is reconstructed on demand as
# base_dequant_val + outlier_delta, see modules.py's outlier_delta
# docstring); this list previously still named it, which crashed
# _prepare_host_buffers with AttributeError unconditionally (every
# QuantLinear, every model) the moment streaming was actually used.
#
# KNOWN GAP (not fixed here, out of scope for this session): a QuantLinear
# built with double_quant=True has self.scale=None/self.gmin=None and its
# real metadata in dq_scale_codes/dq_scale_meta/dq_gmin_codes/dq_gmin_meta
# instead -- _prepare_host_buffers below would hit the same class of
# AttributeError (or silently stream `None`) for such a layer. Not
# triggered by the FLUX checkpoint this session quantized (double_quant
# defaults to False and wasn't turned on), but a real pre-existing gap for
# any caller combining streaming with double_quant=True.
_STREAM_BUFFER_NAMES: Tuple[str, ...] = (
    "packed", "scale", "gmin", "outlier_row", "outlier_col", "outlier_delta",
)


@dataclasses.dataclass
class _LayerState:
    host: Dict[str, torch.Tensor]
    gpu: Optional[Dict[str, torch.Tensor]] = None
    event: Optional["torch.cuda.Event"] = None
    prefetched: bool = False


class StreamingQuantModel:
    """Attaches forward pre/post hooks to every `QuantLinear` in `model` so
    its packed buffers live in pinned host RAM by default and are streamed
    to `device` just before that layer's `forward()` runs, then freed from
    GPU immediately after.

    `prefetch_depth` (default 1): how many layers *ahead* of the one
    currently computing get their H2D copy kicked off on the side CUDA
    stream. 1 matches the roadmap's "prefetch layer N+1 while layer N
    computes" — higher values trade more resident VRAM (each additional
    depth level costs roughly one more layer's packed size) for more slack
    against a slow prefetch not finishing in time, which mostly matters if
    per-layer compute is very fast relative to PCIe transfer time.

    Execution-order assumption: prefetch targets are chosen by each
    QuantLinear's index in `model.named_modules()` order, since the actual
    forward-call order isn't known ahead of time without tracing. This
    matches real execution order for the common case (a stack of
    transformer/diffusion blocks called in definition order); if a model
    calls its Linear layers out of that order, prefetching still can't
    cause wrong results (the pre-hook always waits on *its own* layer's
    transfer, synchronously prefetching on the spot if it wasn't already
    started), it just loses some of the overlap benefit for that call.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        prefetch_depth: int = 1,
        pin_memory: bool = True,
        verbose: bool = True,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("StreamingQuantModel requires a CUDA device")
        self.model = model
        self.device = torch.device(device)
        self.prefetch_depth = max(1, int(prefetch_depth))
        self._pin_memory = pin_memory
        self._verbose = verbose

        self._layers: List[Tuple[str, QuantLinear]] = [
            (n, m) for n, m in model.named_modules() if isinstance(m, QuantLinear)
        ]
        if not self._layers:
            raise ValueError("model has no QuantLinear layers to stream")

        self._copy_stream = torch.cuda.Stream(device=self.device)
        self._handles: List = []
        self._enabled = False
        self._states: List[_LayerState] = []
        self._prepare_host_buffers()

    # -- setup -------------------------------------------------------

    def _prepare_host_buffers(self) -> None:
        total_bytes = 0
        for _, qlin in self._layers:
            host: Dict[str, torch.Tensor] = {}
            for bname in _STREAM_BUFFER_NAMES:
                t = getattr(qlin, bname).detach().to("cpu")
                if self._pin_memory:
                    t = t.pin_memory()
                host[bname] = t
                total_bytes += t.numel() * t.element_size()
                setattr(qlin, bname, t)  # start CPU-resident; GPU copy only exists while streamed in
            if qlin.bias is not None:
                qlin.bias = qlin.bias.detach().to(self.device)
            self._states.append(_LayerState(host=host))
        if self._verbose:
            print(
                f"  streaming: {len(self._layers)} QuantLinear layers pinned in host RAM "
                f"({total_bytes / 1e6:.1f}MB total packed, prefetch_depth={self.prefetch_depth})"
            )

    # -- per-layer prefetch / release ---------------------------------

    def _prefetch(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._layers) or self._states[idx].prefetched:
            return
        _, qlin = self._layers[idx]
        state = self._states[idx]
        with torch.cuda.stream(self._copy_stream):
            gpu = {name: t.to(self.device, non_blocking=True) for name, t in state.host.items()}
        ev = torch.cuda.Event()
        ev.record(self._copy_stream)
        # These buffers are allocated on `_copy_stream` but read later on the
        # compute stream. `record_stream` tells the caching allocator that,
        # so it defers reuse until both streams are done — otherwise a later
        # prefetch could overwrite the buffer while compute is still reading it.
        compute_stream = torch.cuda.current_stream(self.device)
        for t in gpu.values():
            t.record_stream(compute_stream)
        state.gpu, state.event, state.prefetched = gpu, ev, True
        for name, t in gpu.items():
            setattr(qlin, name, t)

    def _release(self, idx: int) -> None:
        _, qlin = self._layers[idx]
        state = self._states[idx]
        if not state.prefetched:
            return
        for name, t in state.host.items():
            setattr(qlin, name, t)  # drop GPU refs; caching allocator reclaims once its stream work drains
        state.gpu, state.event, state.prefetched = None, None, False

    # -- hooks ---------------------------------------------------------

    def _pre_hook(self, idx: int):
        def hook(module, inputs):
            self._prefetch(idx)  # no-op if already prefetched; synchronous kick-off otherwise
            torch.cuda.current_stream(self.device).wait_event(self._states[idx].event)
            for j in range(idx + 1, min(idx + 1 + self.prefetch_depth, len(self._layers))):
                self._prefetch(j)
            return None

        return hook

    def _post_hook(self, idx: int):
        def hook(module, inputs, output):
            self._release(idx)
            return None

        return hook

    # -- lifecycle -------------------------------------------------------

    def enable(self) -> "StreamingQuantModel":
        if self._enabled:
            return self
        for idx, (_, qlin) in enumerate(self._layers):
            self._handles.append(qlin.register_forward_pre_hook(self._pre_hook(idx)))
            self._handles.append(qlin.register_forward_hook(self._post_hook(idx)))
        self._enabled = True
        return self

    def disable(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._enabled = False
        for idx in range(len(self._layers)):
            self._release(idx)

    def stats(self) -> dict:
        host_bytes = sum(sum(t.numel() * t.element_size() for t in st.host.values()) for st in self._states)
        resident = sum(1 for st in self._states if st.prefetched)
        resident_bytes = sum(
            sum(t.numel() * t.element_size() for t in st.gpu.values())
            for st in self._states
            if st.gpu is not None
        )
        return {
            "n_layers": len(self._layers),
            "host_bytes": host_bytes,
            "layers_currently_on_gpu": resident,
            "gpu_resident_bytes": resident_bytes,
        }


def stream_offload_module(
    model: torch.nn.Module,
    device: str = "cuda",
    vram_budget_mb: Optional[float] = None,
    prefetch_depth: Optional[int] = None,
    pin_memory: bool = True,
    verbose: bool = True,
) -> StreamingQuantModel:
    """Convenience entry point: build and `enable()` a `StreamingQuantModel`.

    `vram_budget_mb`, if given, picks `prefetch_depth` from it instead of
    requiring the caller to reason about layer counts directly: how many
    whole layers' worth of packed bytes (average across the model) fit in
    that budget, minus one slot reserved for the layer currently computing.
    `prefetch_depth` (explicit int) overrides this when both are given.
    Defaults to `prefetch_depth=1` when neither is given.
    """
    layers = [m for _, m in model.named_modules() if isinstance(m, QuantLinear)]
    if not layers:
        raise ValueError("model has no QuantLinear layers to stream")

    depth = prefetch_depth
    if depth is None:
        if vram_budget_mb is not None:
            avg_bytes = sum(l.packed_nbytes() for l in layers) / len(layers)
            budget_bytes = vram_budget_mb * 1024 * 1024
            depth = max(1, int(budget_bytes // max(1, avg_bytes)) - 1)
        else:
            depth = 1

    streamer = StreamingQuantModel(
        model, device=device, prefetch_depth=depth, pin_memory=pin_memory, verbose=verbose,
    )
    streamer.enable()
    return streamer
