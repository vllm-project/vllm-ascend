import torch
from torch.nn.parameter import Parameter
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils import GiB_bytes

logger = init_logger(__name__)


def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int,
                   output_partition_sizes: list[int], input_size: int,
                   output_size: int, params_dtype: torch.dtype,
                   **extra_weight_attrs):
    # This method creates unquantized linear weights.
    # The weights are not quantized, and they are not sharded.
    # The amount of memory allocated for the weights is
    # sum(output_partition_sizes) * input_size_per_partition.
    try:
        weight = Parameter(torch.empty(sum(output_partition_sizes),
                                       input_size_per_partition,
                                       dtype=params_dtype),
                           requires_grad=False)
    except torch.cuda.OutOfMemoryError as e:
        logger.error("Failed to create unquantized linear weights: %s", e)
        if torch.cuda.is_available():
            logger.debug("CUDA device: %s", torch.cuda.current_device())
            logger.debug("Allocated: %.2f GiB",
                         torch.cuda.memory_allocated() / GiB_bytes)
            logger.debug("Reserved: %.2f GiB",
                         torch.cuda.memory_reserved() / GiB_bytes)
        raise RuntimeError(
            "Failed to create unquantized linear weights. "
            "This may be caused by insufficient memory to allocate "
            "the weight.") from e
    set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
    layer.register_parameter("weight", weight)
    set_weight_attrs(weight, extra_weight_attrs)


UnquantizedLinearMethod.create_weights = create_weights
