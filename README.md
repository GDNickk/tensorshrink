# tensorshrink

Weight-only quantization for transformer and diffusion models. Shrinks
VRAM/RAM footprint with minimal quality loss, no calibration data or
activations required.

## Contents

* [What it does](#what-it-does)
* [How it works](#how-it-works)

  * [GOAP](#goap--grouped-outlier-aware-packing-the-core-codec)
  * [Companded (Lloyd-Max) quantization](#companded-lloyd-max-quantization)
  * [Adaptive bit-width](#adaptive-bit-width)
  * [AVQ](#avq--additive-vector-quantization-experimental)
* [Install](#install)
* [Quick start](#quick-start)
* [Usage recipes](#usage-recipes)
* [Choosing settings](#choosing-settings)
* [Benchmarks](#benchmarks)
* [Project layout](#project-layout)
* [Status](#status)

## What it does

* Quantizes weights to 2, 4, 6, or 8 bits with grouped, outlier-aware
affine quantization. Outliers get pulled into a small fp16 side table
so a handful of large values don't blow out the range for everyone else.
* Optional companded (Lloyd-Max) quantization: a non-uniform grid fit
per-tensor to its own values instead of round-to-nearest. Free codebook
lookup, fused into the GPU kernel — no VRAM or speed cost.
* Packs sub-byte weights and zstd-compresses them into a `.tsk` container.
* Picks a bit-width per layer by reconstruction error, or fits a
bit-width mix to a VRAM budget across the whole model.
* Runs quantized layers directly (`QuantLinear`) through a fused Triton
kernel (dequant + matmul + outlier correction, one launch), plus a
CUDA quantize path and CPU fallback.
* Drops into `transformers` via `quantization_config`.
* Streams straight from a safetensors checkpoint on disk
(`tensorshrink.cli quantize`) without loading the full model into RAM,
and can stream layers on/off the GPU for models bigger than your VRAM
(`StreamingQuantModel`).

## How it works

### GOAP — Grouped Outlier-Aware Packing (the core codec)

Used by every quantized tensor:

1. **Outliers first.** The top `outlier_frac` (default 0.1%) of elements
by magnitude get pulled out losslessly into a sparse fp16 table before
anything else touches the tensor.
2. **Grouped affine quantization.** Rows are split into groups of
`group_size` (default 128) elements, each with its own `(scale, min)`,
mapped onto a `2**bits`-level grid. Fine grouping keeps error local
instead of averaging it across a whole row or tensor.
3. **Double-quant (optional).** The per-group scale/min metadata gets
quantized again
4. **Pack + zstd.** Codes are bit-packed (4 per byte at 2-bit, 2 at
4-bit) and zstd-compressed. Measured within ~0.5% of the Shannon
entropy bound.

### Companded (Lloyd-Max) quantization

`companded=True` fits a non-uniform grid per tensor with Lloyd's
algorithm (1-D k-means) directly on the weights — still no calibration
data. This is the MSE-optimal grid for the tensor's own distribution, so
it never regresses uniform quantization, and the win grows at low
bit-widths: a 4-level (2-bit) uniform grid wastes levels on tails a
bell-shaped distribution barely touches; companded spends them where the
mass is.

The fitted codebook (≤256 fp16 entries) travels in the `.tsk`
container/`QuantLinear` buffers, and the lookup is one extra `tl.load`
fused into the same kernel every other bit-width uses.

Validated and fast on transformer/LLM models. On diffusion UNets
(hundreds of small distinct-shape `Linear` layers per forward pass) it
currently uses more peak VRAM than plain quantization — leave
`companded=False` (the default) there for now.

### Adaptive bit-width

`adaptive=True` escalates each layer through 2→4→8 bits until its
reconstruction error drops under `error_threshold`, so easy layers stay
small. Default is `adaptive=False` (forced 4-bit + double-quant), which
gives the most predictable VRAM footprint. `vram_budget_mb` fits a
bit-width mix across the whole model to a target size instead.

### AVQ — Additive Vector Quantization (experimental)

`vq.py` is a second, independent codec: vector-quantizes small groups of
weights (default 8 elements) against a k-means codebook, with a second
residual codebook stage (in the spirit of AQLM/QuIP#, implemented from
scratch). On real weights it measurably underperforms the scalar codec
(~0.44 relative error at 1.5 bits/weight vs. scalar 4-bit's ~0.10)
weight columns weren't spatially correlated enough for a shared vector
codebook to pay off. Not part of the public API, not used by
`quantize_module` or the HF integration. Kept for reference, import
directly (`from tensorshrink.vq import vq_quantize_tensor`) if curious,
otherwise use GOAP.

## Install

```bash
pip install torch numpy zstandard
```

Optional:

```bash
pip install triton                    # fused GPU kernels
pip install accelerate                # HF integration (meta-device loading)
pip install safetensors transformers  # HF checkpoint streaming / quantization_config
```

## Quick start

Quantize a live model in place:

```python
import torch
from tensorshrink import quantize_module

model = ...  # an nn.Module, e.g. a loaded HF model
stats = quantize_module(model)
print(f"{stats\['orig_bytes'] / 1e9:.2f} GB -> {stats\['packed_bytes'] / 1e9:.2f} GB")
```

Quantize a checkpoint directory straight to disk, no full model load:

```bash
python -m tensorshrink.cli quantize --model-dir ./my-model --out model.tsk
python -m tensorshrink.cli inspect model.tsk --list
```

Use it with `transformers`:

```python
from transformers import AutoModelForCausalLM
from tensorshrink import TensorShrinkConfig

model = AutoModelForCausalLM.from_pretrained(
    "some/model", quantization_config=TensorShrinkConfig()
)
```

## Usage recipes

**Default:** 4-bit, double-quant, NF4-style codebook.

```python
from tensorshrink import quantize_module

quantize_module(model)  # bits=4, double_quant=True, outlier_frac=0.001, nf_quant=True
```

**Best quality:** `group_size=64` halves the group width (~3% more VRAM
for extra metadata) for a further error reduction on top of the NF4
codebook. Both 64 and 128 run through the fused kernel at the same speed.

```python
quantize_module(model, group_size=64)
```

**Adaptive bit-width** (per-layer auto-selection):

```python
quantize_module(model, adaptive=True, error_threshold=0.10)
```

**2-bit with companded quantization** (plain 2-bit tends to collapse
quality — companded stays usable):

```python
quantize_module(model, bits=2, companded=True)
```

**Fit to a VRAM budget:**

```python
quantize_module(model, adaptive=True, vram_budget_mb=1800)
```

**Save/load a portable `.tsk` file:**

```python
from tensorshrink import save_container, ContainerReader

save_container(model, "model.tsk")

reader = ContainerReader("model.tsk")
print(reader.stats())
```

**Stream-quantize a checkpoint bigger than your RAM:**

```python
from tensorshrink import stream_quantize_safetensors_dir

stream_quantize_safetensors_dir("./my-model", "model.tsk", vram_budget_mb=4000)
```

**Run a model larger than your VRAM:**

```python
from tensorshrink import StreamingQuantModel

streamed = StreamingQuantModel(model)  # wraps an already-quantized model
```

## Choosing settings

|Goal|Setting|
|-|-|
|General use, best default|leave defaults (4-bit, double-quant, NF4)|
|Best quality, small VRAM cost|`group_size=64`|
|Per-layer bit-width selection|`adaptive=True`|
|Hard VRAM ceiling|`vram_budget_mb=<target>`|
|2-bit without collapsing|`companded=True` (with `adaptive=True`)|
|Diffusion / UNet models|`companded=False` (default)|
|Fastest inference|`use_triton=True` (default), needs `triton` + NVIDIA GPU|
|No GPU / no Triton|works automatically via the CPU fallback, just slower|

## Benchmarks

Benched on an RTX 4060 (8GB), default settings (4-bit, double-quant, `outlier_frac=0.001`,`nf_quant=True`), compared against bitsandbytes NF4 + double-quant as a
reference point:



**LLM (Qwen2.5-1.5B-Instruct):**

||tok/s|model VRAM|host RAM|
|-|-|-|-|
|fp16 baseline|26.1|3558 MB|—|
|bitsandbytes NF4|14.1|1631 MB|1048 MB|
|tensorshrink|14.8|1644 MB|981 MB|

Mean relative reconstruction error: 0.0955 (0.0910 with `group_size=64`)
vs. bitsandbytes' ~0.06. Both produce coherent, usable generations.



**Diffusion (Pony Diffusion V6 XL / SDXL UNet, 1024×1024 latent):**

||it/s|model VRAM|
|-|-|-|
|bitsandbytes NF4|2.10|—|
|tensorshrink|2.38|1801 MB|



The UNet path routes large-batch layers (spatial attention) through a
Triton dequant-only kernel + cuBLAS GEMM instead of the fused
dequant+GEMM kernel, which otherwise re-reads weights per row-tile at
large M. Double-quant scale/gmin reconstruction is cached across calls.

Speed numbers fluctuate ~5-10% run to run (Windows/WDDM scheduling) —
treat single measurements as approximate.



## Project layout

|File|What's in it|
|-|-|
|`codec.py`|Core GOAP codec|
|`container.py`|`.tsk` on-disk container format |
|`modules.py`|`QuantLinear`, `quantize_module` (whole-model in-place quantization), `WeightCache`|
|`triton_kernels.py`|Fused Triton GPU kernels |
|`gpu_quantize.py`|CUDA-accelerated quantize path |
|`streaming.py`|`StreamingQuantModel` - run models bigger than VRAM by streaming layers on/off the GPU|
|`hf_integration.py`|`TensorShrinkConfig` - drop-in `quantization_config` for `transformers`|
|`vq.py`|AVQ - vector-quantization codec|
|`cli.py`|`python -m tensorshrink.cli quantize / inspect`|
|`docs/companded-quantization-benchmark.md`|Full benchmark numbers and methodology|

## Status

v0.4.0. Actively developed by a single maintainer. API may still shift
between minor versions.



