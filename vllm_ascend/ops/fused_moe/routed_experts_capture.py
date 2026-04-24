import math

import torch

from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.forward_context import get_forward_context


class AscendRoutedExpertsCapturer(RoutedExpertsCapturer):
    """
    Capturer for routed experts with device and optional shared memory buffer.

    This class captures expert routing decisions during model forward passes
    and optionally stores them in shared memory for cross-process access.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
    

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        """
        Capture expert routing decisions for a specific layer.

        Args:
            layer_id: The layer index.
            topk_ids: Tensor of shape (batch_size, num_routed_experts).
        """

        if self._device_buffer is None:
            raise RuntimeError("Buffer not initialized. Call init_buffer() first.")

        ctx = get_forward_context()
        if ctx.dp_metadata is None:  # single dp
            start_loc = 0
            end_loc = topk_ids.shape[0]
            token_num_per_dp = topk_ids.shape[0]
        else:  # multi dp
            num_tokens_dp = ctx.dp_metadata.num_tokens_across_dp_cpu
            token_num_per_dp = int(num_tokens_dp[self.dp_rank].item())
            total = int(num_tokens_dp.sum().item())
            n = topk_ids.shape[0]

            if n == total:
                # Naive dispatch: all DP ranks' tokens concatenated before routing.
                cumsum = torch.cumsum(num_tokens_dp, dim=0)
                end_loc = int(cumsum[self.dp_rank].item())
                start_loc = end_loc - token_num_per_dp
            elif n == math.ceil(token_num_per_dp / self.tp_size):
                # multi dp & tp
                # when use multi dp, token_per_dp will be padded,
                # we should to unpad it
                full_topk_ids = tensor_model_parallel_all_gather(
                    topk_ids,
                    dim=0
                )
                topk_ids = full_topk_ids[:token_num_per_dp]
                start_loc = 0
                end_loc = token_num_per_dp
            else:
                raise AssertionError(
                    "AscendRoutedExpertsCapturer: unexpected topk_ids batch dim "
                    f"{n} (expected {total} or {token_num_per_dp} "
                    f"for dp_rank={self.dp_rank})"
                )
            
        if layer_id >= self._device_buffer.shape[1]:
            return
        
        self._device_buffer[:token_num_per_dp, layer_id, :] = topk_ids[
            start_loc:end_loc, :
        ]

