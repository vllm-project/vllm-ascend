import torch
from torch.nn import functional as F
from vllm.distributed import (get_tensor_model_parallel_world_size,
                              get_tp_group, tensor_model_parallel_all_gather,
                              tensor_model_parallel_reduce_scatter)
from vllm.forward_context import get_forward_context

from vllm_ascend.platform import NPUPlatform


class MetadataForPadding:

    def __init__(self,
                 padding_flag=False,
                 lengths_sum_padding=0,
                 lengths_sum_unpadding=0,
                 pad_size=0,
                 not_dummy_and_is_prefill=False):
        self.padding_flag = padding_flag
        self.not_dummy_and_is_prefill = not_dummy_and_is_prefill

        self.lengths_sum_padding = lengths_sum_padding
        self.lengths_sum_unpadding = lengths_sum_unpadding
        self.pad_size = pad_size

        self.tp_size = get_tp_group().world_size
        self.tp_rank_in_group = get_tp_group().rank_in_group

        assert self.lengths_sum_padding % self.tp_size == 0
        self.slice_size = self.lengths_sum_padding // self.tp_size

        self.mc2_mask = torch.zeros(
            self.lengths_sum_padding,
            dtype=torch.bool,
            device=NPUPlatform.device_type,
        )
        self.mc2_mask[:lengths_sum_unpadding] = True

    def padding_aligned_reduce_scatter(self,
                                       data: torch.Tensor) -> torch.Tensor:
        if self.padding_flag:
            pad_size = self.pad_size
            padded_data = F.pad(data, (0, 0, 0, pad_size))
        else:
            padded_data = data
        padded_data_reduce_scatter = tensor_model_parallel_reduce_scatter(
            padded_data, 0)

        return padded_data_reduce_scatter

    def allgather_unpadding_aligned(self,
                                    padded_data: torch.Tensor) -> torch.Tensor:
        padded_data_allgather = tensor_model_parallel_all_gather(
            padded_data, 0)
        if self.padding_flag:
            lengths_sum_unpadding = self.lengths_sum_unpadding
            unpadding_data = padded_data_allgather[:lengths_sum_unpadding]
        else:
            unpadding_data = padded_data_allgather
        return unpadding_data

    def padding_slice(self, data: torch.Tensor) -> torch.Tensor:

        padded_data = F.pad(data, (0, 0, 0, self.pad_size))
        start = self.tp_rank_in_group * self.slice_size
        end = start + self.slice_size
        slice_data = padded_data[start:end]

        return slice_data

    def padding_aligned_scatter(self, data: torch.Tensor) -> torch.Tensor:
        if self.padding_flag:
            pad_size = self.pad_size
            padded_data = F.pad(data, (0, 0, 0, pad_size))
        else:
            padded_data = data
        # padded_data = data
        padded_data = torch.tensor_split(padded_data, self.tp_size, dim=0)

        padded_data_reduce_scatter = padded_data[self.tp_rank_in_group]

        return padded_data_reduce_scatter


def init_metadata_for_sp(input_ids, enable_sequence_parallelism):
    if not enable_sequence_parallelism:
        return MetadataForPadding(padding_flag=False,
                                  not_dummy_and_is_prefill=False)

    is_perifll = 0
    attn_metadata = get_forward_context().attn_metadata
    tp_size = get_tensor_model_parallel_world_size()
    if attn_metadata is not None:
        if hasattr(attn_metadata,
                   'is_only_prefill') and attn_metadata.is_only_prefill:
            is_perifll = 1
        if hasattr(attn_metadata,
                   'num_prefills') and attn_metadata.num_prefills > 0:
            is_perifll = 1

        if is_perifll:
            lengths_sum_unpadding = input_ids.shape[0]
            lengths_sum_padding = (
                (lengths_sum_unpadding + tp_size - 1) // tp_size) * tp_size
            if lengths_sum_unpadding == lengths_sum_padding:
                padding_flag = False
            else:
                padding_flag = True
            pad_size = lengths_sum_padding - lengths_sum_unpadding
            _metadata_for_padding = MetadataForPadding(
                lengths_sum_unpadding=lengths_sum_unpadding,
                lengths_sum_padding=lengths_sum_padding,
                padding_flag=padding_flag,
                pad_size=pad_size,
                not_dummy_and_is_prefill=True)

            return _metadata_for_padding

    return MetadataForPadding(padding_flag=False,
                              not_dummy_and_is_prefill=False)
