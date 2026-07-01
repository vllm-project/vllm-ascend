import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    _BUFFER_PREFIX,
    _LOCK_FILE_PREFIX,
    RoutedExpertsCapturer,
    _create_or_attach_shared_memory,
    logger,
)
from vllm.platforms import current_platform


def init_buffer(
    self,
    max_num_batched_tokens: int,
    max_num_kv_tokens: int,
    vllm_config: VllmConfig,
) -> None:
    """
    Initialize the device buffer and optionally shared memory buffer.

    Args:
        max_num_batched_tokens: Maximum number of tokens in a batch.
        max_num_kv_tokens: Maximum number of KV tokens for shared memory.
        vllm_config: vllm configuration containing layer and expert info.
    """

    if self._device_buffer is not None:
        raise RuntimeError("Device buffer has already been initialized")

    hf_config = vllm_config.model_config.hf_text_config
    num_layers = hf_config.num_hidden_layers
    num_experts_per_tok = hf_config.num_experts_per_tok

    # Initialize device buffer
    self._device_buffer = torch.zeros(
        (max_num_batched_tokens, num_layers, num_experts_per_tok),
        dtype=torch.int32,
        device=current_platform.device_name,
    )
    self.dp_rank = vllm_config.parallel_config.data_parallel_rank

    if get_tensor_model_parallel_rank() != 0:
        return

    # Initialize shared memory
    shape = (max_num_kv_tokens, num_layers, num_experts_per_tok)
    buffer_size = int(np.prod(shape)) * np.dtype(np.int32).itemsize
    instance_id = vllm_config.instance_id
    self._lock_file = f"{_LOCK_FILE_PREFIX}_{instance_id}_{self.dp_rank}.lock"
    shm_name = f"{_BUFFER_PREFIX}_{instance_id}_{self.dp_rank}"

    self._shm = _create_or_attach_shared_memory(shm_name, buffer_size, self._lock_file)
    self._host_buffer_view = np.ndarray(shape, dtype=np.int32, buffer=self._shm.buf)
    self._host_buffer_view.fill(0)

    logger.debug(
        "Created shared memory buffer '%s' with shape %s",
        shm_name,
        shape,
    )


# Patch for _device_buffer's initialization(device="cuda" -> device=current_platform.device_name).
# TODO Remove this patch when pr(https://github.com/vllm-project/vllm/pull/34336) is merged.
RoutedExpertsCapturer.init_buffer = init_buffer
