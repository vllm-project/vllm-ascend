from dataclasses import replace
from typing import Any

import torch
from vllm.attention.layer import Attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.speculative import SpeculativeConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle import PADDING_SLOT_ID
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm_ascend.attention.attention_v1 import AscendMetadata
from vllm_ascend.attention.utils import extend_flat_seqs
from vllm_ascend.spec_decode.eagle_proposer import SpecDecodeBaseProposer

logger = init_logger(__name__)


class DraftModelProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=False,
            runner=runner,
        )
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self._raise_if_mrope()
        self._raise_if_padded_drafter_batch()
        self._raise_if_vocab_size_mismatch()
        self._raise_if_draft_tp_mismatch()

    def generate_token_ids(
        self,
        valid_sampled_token_ids: list[list[int]],
        sampling_metadata: SamplingMetadata = None,
        scheduler_output: SchedulerOutput = None,
        spec_decode_metadata: SpecDecodeMetadata = None,
        positions: torch.Tensor = None,
        num_scheduled_tokens: int = 0,
        hidden_states: torch.Tensor = None,
        attn_metadata=None,
        aux_hidden_states: torch.Tensor = None,
    ):
        attn_metadata = self._get_atten_dict(scheduler_output)
        attn_metadata = attn_metadata[self.attn_layer_name]
        next_token_ids: list[int] = []
        for i, token_ids in enumerate(valid_sampled_token_ids):
            if token_ids:
                # Common case.
                next_token_id = token_ids[-1]
            else:
                # Partial prefill (rare case).
                # Get the next token id from the request state.
                req_id = self.runner.input_batch.req_ids[i]
                req_state = self.runner.requests[req_id]
                seq_len = (
                    req_state.num_computed_tokens
                    + scheduler_output.num_scheduled_tokens[req_id]
                )

                next_token_id = req_state.get_token_id(seq_len)
            next_token_ids.append(next_token_id)
        next_token_ids = torch.tensor(
            next_token_ids, dtype=torch.int32, device=self.device
        )

        if spec_decode_metadata is None:
            # input_ids can be None for multimodal models.
            target_token_ids = self.runner.input_ids[:num_scheduled_tokens]
            target_positions = positions[:num_scheduled_tokens]
            cu_num_tokens = attn_metadata.query_start_loc
        else:
            num_draft_tokens = spec_decode_metadata.num_draft_tokens
            num_rejected_tokens = [
                n + 1 - len(valid_sampled_token_ids[i]) if n > 0 else 0
                for i, n in enumerate(num_draft_tokens)
            ]
            num_rejected_tokens = torch.tensor(
                num_rejected_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            cu_num_tokens, token_indices = self._prepare_inputs(
                attn_metadata, num_rejected_tokens
            )
            target_token_ids = self.runner.input_ids[token_indices]
            target_positions = positions[token_indices]

        num_reqs = self.runner.input_batch.num_reqs
        (target_token_ids, target_positions, target_slot_mapping, cu_num_tokens) = merge_next_token_ids_into_token_ids(
                input_token_ids=target_token_ids,
                input_positions=target_positions,
                cad=attn_metadata,
                next_token_ids=next_token_ids,
                block_size=self.block_size,
                max_model_len=self.vllm_config.model_config.max_model_len,
                arange=self.arange,
                cu_num_tokens=cu_num_tokens,
                num_reqs=num_reqs
            )

        draft_token_ids = self._propose(
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=None,
            target_slot_mapping=target_slot_mapping.to(torch.int32),
            next_token_ids=next_token_ids,
            cu_num_tokens=cu_num_tokens,
            block_table=attn_metadata.block_tables,
            sampling_metadata=sampling_metadata,
        )
        spec_token_ids = draft_token_ids.tolist()

        return spec_token_ids

    def _raise_if_mrope(self):
        if self.draft_model_config.uses_mrope:
            raise NotImplementedError(
                "Speculative Decoding with draft models does not support M-RoPE yet"
            )

    def _raise_if_padded_drafter_batch(self):
        if not self.vllm_config.speculative_config.disable_padded_drafter_batch:
            raise NotImplementedError(
                "Speculative Decoding with draft models does not support "
                "padded drafter batch yet. Please pass --disable-padded-drafter-batch "
                "in the speculative_config."
            )

    def _raise_if_vocab_size_mismatch(self):
        speculative_config = self.vllm_config.speculative_config
        if (
            speculative_config.method == "draft_model"
            and speculative_config.target_model_config is not None
            and speculative_config.draft_model_config is not None
        ):
            target_vocab_size = speculative_config.target_model_config.get_vocab_size()
            draft_vocab_size = speculative_config.draft_model_config.get_vocab_size()
            if target_vocab_size != draft_vocab_size:
                raise ValueError(
                    f"Target and draft model should have the same vocabulary size. "
                    f"Target model vocab_size={target_vocab_size}. "
                    f"Draft model vocab_size={draft_vocab_size}. "
                    f"Using models with different tokenizers can cause out-of-bounds "
                    f"errors during speculative decoding."
                )

    def _raise_if_draft_tp_mismatch(self):
        # Note(Tomas Ruiz) If we run the target model with TP > 1 and
        # the draft model with TP = 1, then the different TP ranks collide.
        # Specifically when all ranks compile the draft model on rank 0
        # (because TP=1), then the torch compile cache is overwritten and corrupted.
        # We need a mechanism like this: https://github.com/vllm-project/vllm/pull/5414
        # To prevent this error, we assert that both TP sizes must be the same.
        spec_cfg: SpeculativeConfig = self.vllm_config.speculative_config
        tgt_tp = spec_cfg.target_parallel_config.tensor_parallel_size
        draft_tp = spec_cfg.draft_parallel_config.tensor_parallel_size
        if draft_tp != tgt_tp:
            raise ValueError(
                f"Currently, 'draft_tensor_parallel_size' and 'tensor_parallel_size' "
                f"must be the same. Got {draft_tp} and {tgt_tp}. "
                "Please pass 'draft_tensor_parallel_size' in the speculative_config."
            )

    def set_input_ids_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        num_tokens: int,
        last_token_indices: torch.Tensor,
    ) -> None:
        self.input_ids[:num_tokens] = target_token_ids

    def load_model(self, target_model: Any) -> None:
        """Takes target_model to satisfy the type checker."""

        # This must be computed before loading the draft model
        # because that mutates the forward_context of the vllm_config
        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, Attention).keys()
        )

        from vllm.compilation.backends import set_model_tag

        draft_vllm_config: VllmConfig = create_vllm_config_for_draft_model(
            target_model_vllm_config=self.vllm_config
        )
        logger.info(
            "Starting to load draft model %s. TP=%d, rank=%d",
            draft_vllm_config.model_config.model,
            draft_vllm_config.parallel_config.tensor_parallel_size,
            draft_vllm_config.parallel_config.rank,
        )
        with set_model_tag("draft_model"):
            self.model = get_model(vllm_config=draft_vllm_config, prefix="draft_model")

        # This must be computed after loading the draft model
        # because that mutates the forward_context of the vllm_config
        draft_attn_layer_names = (
            get_layers_from_vllm_config(self.vllm_config, Attention).keys()
            - target_attn_layer_names
        )
        self.attn_layer_name = next(iter(draft_attn_layer_names))


def create_vllm_config_for_draft_model(
    target_model_vllm_config: VllmConfig,
) -> VllmConfig:
    """The vllm_config is configured for the target model, e.g.
    its quant_config and parallel_config. But the draft model is potentially
    quantized differently, and has potentially different tensor_parallel_size.
    This function creates a new vllm_config configured for the draft model.
    The vllm_config is useful when loading the draft model with get_model().
    """
    old = target_model_vllm_config
    new_parallel_config = replace(
        old.speculative_config.draft_parallel_config, rank=old.parallel_config.rank
    )

    new: VllmConfig = replace(
        old,
        quant_config=None,  # quant_config is recomputed in __init__()
        model_config=old.speculative_config.draft_model_config,
        parallel_config=new_parallel_config,
    )
    return new


def merge_next_token_ids_into_token_ids(
    input_token_ids: torch.Tensor,
    input_positions: torch.Tensor,
    cad: AscendMetadata,
    next_token_ids: torch.Tensor,
    block_size: int,
    max_model_len: int,
    arange: torch.Tensor,
    cu_num_tokens,
    num_reqs
):
    """
    Merges the next token ids with the existing token ids into a flat sequence.
    Does the same for the positions, computes new slot mapping,
    and updates the common_attn_metadata. The inputs are not modified in-place.
    """
    query_end_locs = cu_num_tokens[1:] - 1
    new_token_ids = extend_flat_seqs(
        seqs=input_token_ids, end_locs=query_end_locs, new_vals=next_token_ids
    )

    # append new positions
    positions_to_append = input_positions[query_end_locs] + 1
    new_positions = extend_flat_seqs(
        seqs=input_positions, end_locs=query_end_locs, new_vals=positions_to_append
    )
    # recompute slot mapping
    batch_size, n_blocks_per_req = cad.block_tables.shape
    req_indices = torch.arange(num_reqs, device=cad.query_start_loc.device)

    query_lens = cu_num_tokens[1:] - cu_num_tokens[:-1]
    req_indices = torch.repeat_interleave(
        req_indices, query_lens.to(cad.query_start_loc.device) + 1
    )
    block_table_indices = req_indices * n_blocks_per_req + new_positions // block_size
    block_nums = cad.block_tables.view(-1)[block_table_indices]
    block_offsets = new_positions % block_size
    new_slot_mapping = block_nums * block_size + block_offsets
    # Mask out the position ids that exceed the max model length.
    exceeds_max_model_len = new_positions >= max_model_len
    new_slot_mapping.masked_fill_(exceeds_max_model_len, PADDING_SLOT_ID)

    cu_num_tokens = cu_num_tokens + arange[: len(cu_num_tokens)]
    return (new_token_ids, new_positions, new_slot_mapping, cu_num_tokens)
