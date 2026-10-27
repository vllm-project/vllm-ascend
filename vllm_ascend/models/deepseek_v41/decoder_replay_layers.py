# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

import torch
from vllm.compilation.breakable_cudagraph import eager_break_during_capture, is_breakable_cudagraph_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_tp_group, tensor_model_parallel_all_gather
from vllm.forward_context import get_forward_context, in_piecewise_cudagraph
from vllm.logger import logger

from vllm_ascend.utils import enable_dsa_cp

if TYPE_CHECKING:
    from .model import DeepseekV41Model


class AscendDecoderReplayLayers:
    """Gather and scatter the MRV2 decoder sub-batch, including sharded states."""

    def __init__(
        self,
        window: int,
        run_layers: Callable[..., tuple[torch.Tensor, ...]],
        row_buffers: list[torch.Tensor],
        *,
        write_batch_kv: Callable[..., None] | None = None,
        first_swa_prefix: str | None = None,
    ) -> None:
        self.window = window
        self.run_layers = run_layers
        self.row_buffers = row_buffers
        self.write_batch_kv = write_batch_kv
        self.first_swa_prefix = first_swa_prefix
        self.rows: torch.Tensor | None = None
        self.context_factory: Callable[[], AbstractContextManager[None]] | None = None
        self.num_actual_rows = 0
        self.num_padded_rows: int | None = None
        self.sequence_parallel = False
        self.trim_threshold: int | None = None
        self.graph_runner: (
            Callable[[torch.Tensor, tuple[torch.Tensor | None, ...]], tuple[torch.Tensor, ...]] | None
        ) = None
        self._break_outputs: list[torch.Tensor] | None = None

    def __call__(self, *states: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
        if self.uses_replay_batch():
            # The cut layer publishes KV for the complete batch before the
            # eager graph break switches to the compact replay rows.
            if self.write_batch_kv is not None:
                self.write_batch_kv(*states)
            if in_piecewise_cudagraph():
                return self._run_in_graph_break(*states)
        return self._run(*states)

    def uses_replay_batch(self) -> bool:
        """Whether this forward uses replay rows, including during capture."""
        if in_piecewise_cudagraph():
            batch_descriptor = get_forward_context().batch_descriptor
            assert batch_descriptor is not None
            return self.graph_uses_replay_batch(batch_descriptor.num_tokens)
        return self.rows is not None

    def graph_uses_replay_batch(self, num_tokens: int) -> bool:
        return self.trim_threshold is not None and num_tokens >= self.trim_threshold

    @eager_break_during_capture
    def _run_in_graph_break(self, *states: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
        outputs = self._run(*states)
        if self._break_outputs is None:
            self._break_outputs = [torch.empty_like(output) for output in outputs]
        assert all(output.shape[0] <= buffer.shape[0] for output, buffer in zip(outputs, self._break_outputs))
        return tuple(buffer[: output.shape[0]].copy_(output) for buffer, output in zip(self._break_outputs, outputs))

    def _run(self, *states: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
        if self.rows is None:
            return self.run_layers(*states)
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
                if self.num_padded_rows is not None:
                    buffer[rows.numel() : self.num_padded_rows].zero_()
            replay_states: list[torch.Tensor | None] = []
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
                if self.num_padded_rows is not None:
                    padding = self.num_padded_rows - state.shape[0]
                    state = torch.cat((state, state.new_zeros((padding, *state.shape[1:]))))
                replay_states.append(state)
            if self.graph_runner is not None:
                row_outputs = self.graph_runner(rows, tuple(replay_states))
            else:
                row_outputs = self.run_layers(*replay_states)
            outputs = []
            for index, row_output in enumerate(row_outputs):
                # Hidden states and pre-mix are SP-local; auxiliary states
                # have already been gathered by the model's layer loop.
                sharded = self.sequence_parallel and index < 2
                if sharded:
                    row_output = tensor_model_parallel_all_gather(row_output, dim=0)
                output = row_output.new_zeros((positions.shape[0], *row_output.shape[1:]))
                output.index_copy_(0, rows[: self.num_actual_rows], row_output[: self.num_actual_rows])
                if sharded:
                    tp = get_tp_group()
                    output = output.chunk(tp.world_size)[tp.rank_in_group]
                outputs.append(output)
            return tuple(outputs)


def make_decoder_replay(model: "DeepseekV41Model", config: VllmConfig) -> AscendDecoderReplayLayers | None:
    if not getattr(config.cache_config, "swa_bounded_replay", False):
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
    if parallel.prefill_context_parallel_size > 1 and parallel.data_parallel_size > 1:
        reason = "PCP decoder replay currently requires DP=1"
    elif (
        (model.use_sequence_parallel and not enable_dsa_cp())
        or parallel.pipeline_parallel_size > 1
        or parallel.decode_context_parallel_size > 1
        or parallel.use_ubatching
    ):
        reason = "SP without DSA CP, PP, PCP, DCP and microbatching need a separate replay layout"
    elif config.compilation_config.cudagraph_mode not in (
        CUDAGraphMode.NONE,
        CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.PIECEWISE,
    ):
        reason = "replay prefill requires eager or breakable PIECEWISE execution"
    elif config.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE and not is_breakable_cudagraph_enabled():
        reason = "PIECEWISE replay requires breakable ACL graphs"
    elif any(layer > cut for layer in model.config.engram_layer_ids):
        reason = "an Engram layer follows the last KV source"
    elif speculative is not None and not speculative.use_dspark():
        reason = "the replay adapter currently supports DSpark drafting only"
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
    elif config.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE and getattr(
        getattr(config, "model_config", None), "enable_prompt_embeds", False
    ):
        reason = "prompt embeddings need a separate replay graph input layout"
    else:
        # Replay the last KV-source layer too when its input layout can be
        # reproduced without sequence/context parallelism or an Engram lookup.
        # Other supported layouts keep the existing cut+1 replay boundary.
        cut_replay_supported = (
            parallel.pipeline_parallel_size == 1
            and parallel.prefill_context_parallel_size == 1
            and parallel.decode_context_parallel_size == 1
            and not parallel.use_ubatching
            and not model.use_sequence_parallel
            and not enable_dsa_cp()
            and not any(layer >= cut for layer in model.config.engram_layer_ids)
            and hasattr(model, "layers")
        )
        cut_layer = model.layers[cut] if cut_replay_supported else None
        write_batch_kv = cut_layer.write_kv if cut_layer is not None else None
        first_swa_prefix = cut_layer.self_attn.v41_impl.swa_prefix if cut_layer is not None else None
        model.decoder_replay_start = cut if cut_layer is not None else cut + 1

        index_sources = set(getattr(model.config, "index_source_layer_ids", ()))
        candidate_source = int(getattr(model.config, "candidate_source_layer_id", -1))
        cut_is_replayed = write_batch_kv is not None
        row_buffers = []
        if (not cut_is_replayed or cut not in index_sources) and model.topk_indices_buffer is not None:
            row_buffers.append(model.topk_indices_buffer)
        if not cut_is_replayed or cut != candidate_source:
            row_buffers.extend(
                buffer
                for buffer in (model.candidate_indices_buffer, model.candidate_lengths_buffer)
                if buffer is not None
            )
        if (not cut_is_replayed or cut not in index_sources) and model.topk_lengths_buffer is not None:
            row_buffers.append(model.topk_lengths_buffer)

        if write_batch_kv is not None:
            logger.info_once(
                "Ascend MRV2 decoder bounded replay: layer %d writes full-batch KV, "
                "then layers %d-%d replay each prefill's last %d tokens.",
                cut,
                cut,
                model.end_layer - 1,
                model.config.sliding_window,
            )
        else:
            logger.info_once(
                "Ascend MRV2 decoder bounded replay: layers %d-%d keep each prefill's last %d tokens.",
                cut + 1,
                model.end_layer - 1,
                model.config.sliding_window,
            )
        replay = AscendDecoderReplayLayers(
            model.config.sliding_window,
            model._run_replay_layers,
            row_buffers,
            write_batch_kv=write_batch_kv,
            first_swa_prefix=first_swa_prefix,
        )
        replay.sequence_parallel = model.use_sequence_parallel
        if (
            config.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE
            and config.compilation_config.cudagraph_decoder_replay
        ):
            threshold = config.compilation_config.decoder_replay_trim_threshold
            if threshold <= replay.window:
                raise ValueError("decoder_replay_trim_threshold must exceed the replay window")
            replay.trim_threshold = threshold
        return replay
    logger.warning_once("Ascend decoder bounded replay is disabled: %s.", reason)
    return None
