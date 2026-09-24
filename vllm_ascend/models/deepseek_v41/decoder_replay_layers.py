# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_tp_group, tensor_model_parallel_all_gather
from vllm.logger import logger
from vllm.models.deepseek_v41.decoder_replay_layers import DecoderReplayLayers

from vllm_ascend.utils import enable_dsa_cp

if TYPE_CHECKING:
    from .model import DeepseekV41Model


class AscendDecoderReplayLayers(DecoderReplayLayers):
    context_factory: Callable[[], AbstractContextManager[None]] | None = None
    num_actual_rows: int = 0
    sequence_parallel: bool = False

    def _run(self, outputs: tuple[torch.Tensor, ...], *states: torch.Tensor | None) -> None:
        if self.rows is None:
            super()._run(outputs, *states)
            return
        assert self.context_factory is not None
        positions = states[2]
        assert positions is not None
        with self.context_factory():
            rows = self.rows
            for buffer in self.row_buffers:
                if enable_dsa_cp():
                    # Indexer outputs use the old CP partition; replay queries
                    # repartition the retained rows across the same TP group.
                    tp = get_tp_group()
                    local_tokens = positions.shape[0] // tp.world_size
                    global_buffer = tensor_model_parallel_all_gather(buffer[:local_tokens], dim=0)
                    replay_tokens = rows.numel() // tp.world_size
                    local_rows = rows[tp.rank_in_group * replay_tokens : (tp.rank_in_group + 1) * replay_tokens]
                    buffer[:replay_tokens].copy_(global_buffer.index_select(0, local_rows))
                else:
                    buffer[: rows.numel()].copy_(buffer.index_select(0, rows))
            replay_states = []
            for index, state in enumerate(states):
                if state is None:
                    replay_states.append(None)
                    continue
                sharded = self.sequence_parallel and index != 2
                if sharded:
                    state = tensor_model_parallel_all_gather(state, dim=0)
                state = state.index_select(0, rows)
                if sharded:
                    state = state.chunk(get_tp_group().world_size)[get_tp_group().rank_in_group]
                replay_states.append(state)
            row_outputs = self.run_layers(*replay_states)
            for index, (output, row_output) in enumerate(zip(outputs, row_outputs, strict=True)):
                # Hidden states and pre-mix are SP-local; auxiliary states
                # have already been gathered by the model's layer loop.
                if self.sequence_parallel and index < 2:
                    tp = get_tp_group()
                    row_output = tensor_model_parallel_all_gather(row_output, dim=0)
                    full_output = output.new_zeros((output.shape[0] * tp.world_size, *output.shape[1:]))
                    full_output.index_copy_(0, rows[: self.num_actual_rows], row_output[: self.num_actual_rows])
                    output.copy_(full_output.chunk(tp.world_size)[tp.rank_in_group])
                else:
                    output.index_copy_(0, rows[: self.num_actual_rows], row_output[: self.num_actual_rows])


def make_decoder_replay(model: "DeepseekV41Model", config: VllmConfig) -> AscendDecoderReplayLayers | None:
    if not config.cache_config.swa_bounded_replay:
        return None
    parallel = config.parallel_config
    speculative = config.speculative_config
    draft_config = speculative.draft_model_config if speculative is not None else None
    draft_hf_config = getattr(draft_config, "hf_config", None)
    draft_window = getattr(draft_hf_config, "sliding_window", None)
    draft_layer_types = getattr(draft_hf_config, "layer_types", None) or ()
    cut = max(model.config.kv_source_layer_ids)
    if cut >= model.end_layer - 1:
        return None
    if config.use_v2_model_runner:
        reason = "the Ascend replay adapter currently uses the MRV1 preparation hook"
    elif config.cache_config.enable_prefix_caching:
        reason = "MRV1 encoder-side prefix replay is required when prefix caching is enabled"
    elif (
        (model.use_sequence_parallel and not enable_dsa_cp())
        or parallel.pipeline_parallel_size > 1
        or parallel.prefill_context_parallel_size > 1
        or parallel.decode_context_parallel_size > 1
        or parallel.use_ubatching
    ):
        reason = "SP without DSA CP, PP, PCP, DCP and microbatching need a separate replay layout"
    elif config.compilation_config.cudagraph_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY):
        reason = "replay prefill requires eager execution; use NONE or FULL_DECODE_ONLY"
    elif any(layer > cut for layer in model.config.engram_layer_ids):
        reason = "an Engram layer follows the last KV source"
    elif speculative is not None and not speculative.use_dspark():
        reason = "the MRV1 replay adapter currently supports DSpark drafting only"
    elif draft_config is not None and (
        draft_window is None
        or draft_window > model.config.sliding_window
        or any(layer_type != "sliding_attention" for layer_type in draft_layer_types)
    ):
        reason = (
            f"the drafter (sliding window {draft_window}) reads hidden states "
            f"outside the target's {model.config.sliding_window}-token window"
        )
    elif config.lora_config is not None:
        reason = "LoRA token mappings need to follow the replay batch"
    else:
        model.decoder_replay_start = cut + 1
        logger.info_once(
            "Ascend MRV1 decoder bounded replay: layers %d-%d keep each prefill's last %d tokens.",
            cut + 1,
            model.end_layer - 1,
            model.config.sliding_window,
        )
        replay = AscendDecoderReplayLayers(
            model.config.sliding_window,
            model._run_replay_layers,
            model._new_replay_outputs,
            [model.topk_indices_buffer, model.candidate_indices_buffer],
        )
        replay.sequence_parallel = model.use_sequence_parallel
        return replay
    logger.warning_once("Ascend decoder bounded replay is disabled: %s.", reason)
    return None
