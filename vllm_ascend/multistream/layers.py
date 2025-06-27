from typing import List, Optional, Tuple, Union

import torch
from vllm.forward_context import get_forward_context

from .base import MSEventKey
from .context import (get_multistream_layer_context,
                      reset_multistream_layer_context,
                      set_multistream_layer_context)
from .metadata import MultiStreamMetadata


class MultiStreamPreTransformerLayer(torch.nn.Module):

    def __init__(self, multistream_metadata: MultiStreamMetadata):
        super().__init__()
        self.multistream_metadata = multistream_metadata

    def forward(
        self,
        intput_tensors: List[torch.Tensor],
    ):
        attn_metadata = get_forward_context().attn_metadata
        if self.multistream_metadata is None or attn_metadata is None:
            set_multistream_layer_context(-1, None, None)
            return attn_metadata, intput_tensors
        # TODO add attn_metadata management
        do_ms, attn_metadata, intput_tensors, _ = self.multistream_metadata.split_micro_batch(
            attn_metadata, intput_tensors)
        if do_ms:
            set_multistream_layer_context(
                self.multistream_metadata.start_layer,
                self.multistream_metadata, attn_metadata)
        else:
            set_multistream_layer_context(-1, None, None)
        return attn_metadata, intput_tensors


class MultiStreamPostTransformerLayer(torch.nn.Module):

    def __init__(self, multistream_metadata: MultiStreamMetadata):
        super().__init__()
        self.multistream_metadata = multistream_metadata

    def forward(self,
                input_tensors: Union[List[Tuple[torch.Tensor]],
                                     List[torch.Tensor],
                                     List[List[torch.Tensor]]],
                wait_layer_index: Optional[int] = None):
        if self.multistream_metadata is None or self.multistream_metadata.ms_config is None:
            return input_tensors
        layer_index, ms_metadata, ms_attn_metadata = get_multistream_layer_context(
        )
        if layer_index >= 0:
            true_wait_layer = self.multistream_metadata.end_layer - 1 if wait_layer_index is None else wait_layer_index
            self.multistream_metadata.try_wait_event(
                true_wait_layer,
                self.multistream_metadata.ms_config.num_micro_batches - 1,
                MSEventKey.FFN_AR_FINISH)
            reset_multistream_layer_context()
        return self.multistream_metadata.merge_micro_batches(input_tensors)
