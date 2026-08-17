"""
tensorshrink — an original, from-scratch weight-compression toolkit for local
transformer / diffusion models.

Public API:
    quantize_tensor / dequantize_tensor       -> core codec (codec.py)
    save_container / load_container / QuantizedContainerReader -> disk format (container.py)
    QuantLinear, quantize_module, quantize_pretrained -> live nn.Module swap (modules.py)
"""

from .codec import (
    QuantizedTensor,
    quantize_tensor,
    dequantize_tensor,
    quantize_tensor_adaptive,
)
from .container import (
    save_container,
    ContainerReader,
    stream_quantize_safetensors_dir,
    CONTAINER_FORMAT_VERSION,
)
from .modules import (
    QuantLinear,
    quantize_module,
    load_quantized_module,
    DEFAULT_SKIP_PATTERNS,
    T5_EXTRA_SKIP_PATTERNS,
    WeightCache,
    suggest_max_dequant_rows,
    CUDAGraphQuantModel,
)
from .streaming import StreamingQuantModel, stream_offload_module
from .hf_integration import TensorShrinkConfig
from .gpu_quantize import (
    quantize_tensor_gpu,
    quantize_tensor_gpu_adaptive,
    quantize_module_gpu,
)

__version__ = "0.4.0"

__all__ = [
    "QuantizedTensor",
    "quantize_tensor",
    "dequantize_tensor",
    "quantize_tensor_adaptive",
    "save_container",
    "ContainerReader",
    "stream_quantize_safetensors_dir",
    "CONTAINER_FORMAT_VERSION",
    "QuantLinear",
    "quantize_module",
    "load_quantized_module",
    "DEFAULT_SKIP_PATTERNS",
    "T5_EXTRA_SKIP_PATTERNS",
    "WeightCache",
    "suggest_max_dequant_rows",
    "StreamingQuantModel",
    "stream_offload_module",
    "TensorShrinkConfig",
    "CUDAGraphQuantModel",
    "quantize_tensor_gpu",
    "quantize_tensor_gpu_adaptive",
    "quantize_module_gpu",
]
