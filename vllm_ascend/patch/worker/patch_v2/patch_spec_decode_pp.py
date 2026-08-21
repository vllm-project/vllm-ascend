# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Speculative decoding support for Model Runner V2 PP."""

from itertools import islice

import numpy as np
import torch
from vllm.distributed.parallel_state import get_pp_group
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu

_AUX_PREFIX = "eagle3_aux_"
_FORWARD_PATCHED = "_vllm_ascend_eagle3_pp_forward_patched"
_BROADCAST_PATCHED = "_vllm_ascend_spec_decode_pp_broadcast_patched"


def _get_aux_states(
    intermediate_tensors: IntermediateTensors | None,
) -> list[torch.Tensor]:
    if intermediate_tensors is None:
        return []
    keys = sorted(
        (key for key in intermediate_tensors.tensors if key.startswith(_AUX_PREFIX)),
        key=lambda key: int(key.removeprefix(_AUX_PREFIX)),
    )
    return [intermediate_tensors[key] for key in keys]


def _qwen3_eagle3_pp_forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
):
    pp_group = get_pp_group()
    aux_hidden_states = _get_aux_states(intermediate_tensors)

    if pp_group.is_first_rank:
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        residual = None
        self._maybe_add_hidden_state(aux_hidden_states, 0, hidden_states, residual)
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    for layer_idx, layer in enumerate(
        islice(self.layers, self.start_layer, self.end_layer),
        start=self.start_layer,
    ):
        hidden_states, residual = layer(positions, hidden_states, residual)
        self._maybe_add_hidden_state(
            aux_hidden_states,
            layer_idx + 1,
            hidden_states,
            residual,
        )

    if not pp_group.is_last_rank:
        tensors = {"hidden_states": hidden_states, "residual": residual}
        tensors.update({f"{_AUX_PREFIX}{idx}": aux_state for idx, aux_state in enumerate(aux_hidden_states)})
        return IntermediateTensors(tensors)

    if len(aux_hidden_states) != len(self.aux_hidden_state_layers):
        raise RuntimeError(
            "Eagle3 PP did not collect every auxiliary hidden state: "
            f"expected {len(self.aux_hidden_state_layers)}, got {len(aux_hidden_states)}."
        )
    hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states, aux_hidden_states


def prepare_qwen3_eagle3_pp_forward() -> None:
    """Patch before model loading so torch.compile captures the PP forward."""
    from vllm.model_executor.models.qwen3 import Qwen3Model

    if getattr(Qwen3Model, _FORWARD_PATCHED, False):
        return
    Qwen3Model.forward = _qwen3_eagle3_pp_forward
    setattr(Qwen3Model, _FORWARD_PATCHED, True)


def install_qwen3_eagle3_pp_aux(model):
    inner_model = model.model
    original_make_empty = inner_model.make_empty_intermediate_tensors
    incoming_aux_count = sum(layer_idx <= inner_model.start_layer for layer_idx in inner_model.aux_hidden_state_layers)

    def make_empty_intermediate_tensors(batch_size, dtype, device):
        result = original_make_empty(batch_size, dtype, device)
        for idx in range(incoming_aux_count):
            result.tensors[f"{_AUX_PREFIX}{idx}"] = torch.zeros_like(result["hidden_states"])
        return result

    inner_model.make_empty_intermediate_tensors = make_empty_intermediate_tensors
    return inner_model


def install_spec_decode_pp_token_broadcast(pp_handler) -> None:
    """Send accepted and next-draft tokens through the same V2 PP slot."""
    if getattr(pp_handler, _BROADCAST_PATCHED, False):
        return

    max_sample_len = pp_handler.max_sample_len
    draft_width = max_sample_len - 1
    token_payload_width = max_sample_len + draft_width
    original_get_prev_sampled_outputs = pp_handler.get_prev_sampled_outputs
    original_broadcast = pp_handler.broadcast
    pending_send = None

    def get_prev_sampled_outputs():
        slot = pp_handler.queue[0] if pp_handler.queue else None
        outputs = original_get_prev_sampled_outputs()
        if outputs is None:
            return None
        assert slot is not None
        token_payload = outputs["sampled_tokens"]
        outputs["sampled_tokens"] = token_payload[:, :max_sample_len]
        outputs["draft_tokens"] = token_payload[:, max_sample_len:]

        # Preserve valid rows on CPU; NPU bool indexing lowers to NonzeroV2.
        freed = pp_handler.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
        exclude_mask = freed | ~slot.need_sampled_mask
        draft_update_indices = None
        if exclude_mask.any():
            valid_rows = np.flatnonzero(~exclude_mask)
            update_indices = np.stack(
                (valid_rows, slot.idx_mapping_np[valid_rows]),
            )
            draft_update_indices = async_copy_to_gpu(
                update_indices,
                device=pp_handler.device,
            )
        outputs["draft_update_indices"] = draft_update_indices
        return outputs

    def broadcast(sampled_token_ids, num_sampled, num_rejected, input_batch):
        nonlocal pending_send
        assert pp_handler.is_last_rank
        if pending_send is not None:
            raise RuntimeError("Speculative PP already has a pending sampled-token broadcast.")
        pending_send = (
            sampled_token_ids,
            num_sampled,
            num_rejected,
            input_batch,
        )

    def broadcast_draft_tokens(draft_tokens):
        nonlocal pending_send
        if pending_send is None:
            return

        sampled_token_ids, num_sampled, num_rejected, input_batch = pending_send
        pending_send = None
        num_reqs = input_batch.num_reqs
        token_payload = sampled_token_ids.new_zeros((num_reqs, token_payload_width))
        token_payload[:, : sampled_token_ids.shape[1]].copy_(sampled_token_ids)
        token_payload[:, max_sample_len:].copy_(draft_tokens)
        original_broadcast(
            token_payload,
            num_sampled,
            num_rejected,
            input_batch,
        )

    pp_handler.max_sample_len = token_payload_width
    pp_handler.get_prev_sampled_outputs = get_prev_sampled_outputs
    pp_handler.broadcast = broadcast
    pp_handler.broadcast_draft_tokens = broadcast_draft_tokens
    setattr(pp_handler, _BROADCAST_PATCHED, True)
