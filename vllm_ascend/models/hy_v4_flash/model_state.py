# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch
from vllm.config import CUDAGraphMode
from vllm.logger import init_logger
from vllm.triton_utils import triton

from vllm_ascend.ops.triton.engram_lookback import _gather_lookback_kernel
from vllm_ascend.utils import lmhead_tp_enable
from vllm_ascend.worker.v2.model_states.default import AscendModelState

from .aclgraph import FlashAclGraphManager
from .pd_role import uses_prefill_ops

logger = init_logger(__name__)


class FlashModelState(AscendModelState):
    aclgraph_manager_cls = FlashAclGraphManager

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        self._graph_decode = False
        self._reuse_eager_embeddings = (
            uses_prefill_ops(vllm_config)
            and vllm_config.model_config.enforce_eager
            and vllm_config.parallel_config.prefill_context_parallel_size == 1
        )
        language_model = getattr(model, "language_model", model)
        self._flash_model = language_model.model
        self._logits_processor = language_model.logits_processor
        self._prefill_vocab_embedding = None
        if self._reuse_eager_embeddings and os.getenv("HY4_PREFILL_FUSED_VOCAB") == "1":
            embedding = self._flash_model.embed_tokens
            if (
                self.prompt_embeds_state is None
                and self.mm_pruner is None
                and not self._flash_model.uses_dfly
                and embedding.hy4_prefill_vocab_parameters() is not None
            ):
                self._prefill_vocab_embedding = embedding
            first_layer = self._flash_model.layers[0]
            first_engram = 1 in (self._flash_model.config.engram_layer_ids or ())
            if (
                self._prefill_vocab_embedding is not None
                and os.getenv("HY4_PREFILL_FIRST_IHC_BROADCAST") == "1"
                and not first_layer.sequence_parallel and not first_engram
            ):
                first_layer.hc_attn_layer.configure_prefill_broadcast(
                    embedding, vllm_config.scheduler_config.max_num_batched_tokens,
                )
        self._defer_hc_head = (
            os.getenv("HY4_PREFILL_SAMPLE_ONLY_HEAD") == "1"
            and self._reuse_eager_embeddings
            and vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
            and vllm_config.speculative_config is None
            and vllm_config.lora_config is None
            and vllm_config.parallel_config.pipeline_parallel_size == 1
            and vllm_config.parallel_config.data_parallel_size == 1
            and vllm_config.parallel_config.decode_context_parallel_size == 1
            and not vllm_config.parallel_config.enable_batch_sharded_sampling
            and not lmhead_tp_enable()
            and self._flash_model.engram_hash is not None
        )
        checks = int(os.getenv("HY4_PREFILL_HEAD_CHECK_STEPS", "0")) if self._defer_hc_head else 0
        self._hc_head_checks = checks
        self._lm_head_checks = checks
        self._graph_hashes = None
        if model.token_lookback_depth and vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs():
            engram = self._flash_model.engram_hash
            capacity = min(
                vllm_config.scheduler_config.max_num_batched_tokens,
                vllm_config.scheduler_config.max_num_seqs * vllm_config.uniform_decode_query_len,
            )
            self._graph_hashes = torch.full(
                (len(engram.layout.layer_ids), capacity, engram.offsets.shape[1]), -1, dtype=torch.int32, device=device
            )
            self._engram_embeddings = tuple(
                self._flash_model.layers[layer - 1].engram_layer.multi_head_embedding
                for layer in engram.layout.layer_ids
            )
            self._graph_rows = tuple(
                torch.zeros(
                    (capacity, embedding.head_end - embedding.head_start, embedding.dim),
                    dtype=torch.float32,
                    device=device,
                )
                for embedding in self._engram_embeddings
            )

    def _prepare_prefill_inputs(self, input_batch, req_states):
        rope_positions = getattr(input_batch, "hy4_rope_positions", None)
        if rope_positions is None:
            inputs = super().prepare_inputs(input_batch, req_states)
        else:
            inputs = {"positions": rope_positions}
            inputs.update(self.prepare_engram_inputs(input_batch, req_states))
        indices = getattr(input_batch, "hy4_vocab_indices", None)
        if indices is not None and not self.supports_mm_inputs:
            inputs["inputs_embeds"] = self._prefill_vocab_embedding.embed_hy4_local_indices(indices)
            input_batch.hy4_text_vocab_embeddings = True
        if getattr(input_batch, "hy4_text_vocab_embeddings", False):
            inputs["first_ihc_vocab"] = True
        worker = self.prompt_logprobs_worker
        if (
            self._defer_hc_head
            and not input_batch.is_dummy
            and not self.kvpp_is_dummy_run
            and input_batch.num_tokens == input_batch.num_tokens_after_padding
            and input_batch.logits_indices.numel() > 0
            and (
                input_batch.num_computed_prefill_tokens_np
                < req_states.prefill_len.np[input_batch.idx_mapping_np]
            ).all()
            and worker is not None
            and not worker.uses_prompt_logprobs[input_batch.idx_mapping_np].any()
            and not any(req_id in worker.token_id_scores for req_id in input_batch.req_ids)
        ):
            inputs["defer_hc_head"] = True
            if self._hc_head_checks:
                inputs["check_hc_head_rows"] = input_batch.logits_indices
                self._hc_head_checks -= 1
        return inputs

    def compute_sample_logits(self, hidden_states):
        if self._defer_hc_head and hidden_states.ndim == 3:
            hidden_states = self._flash_model.finish_hidden_states(hidden_states)
        logits = super().compute_sample_logits(hidden_states)
        processor = self._logits_processor
        if self._lm_head_checks and logits is not None and logits.numel() > 0 and processor.use_npu_fp32_head:
            self._lm_head_checks -= 1
            processor.use_npu_fp32_head = False
            try:
                reference = super().compute_sample_logits(hidden_states)
            finally:
                processor.use_npu_fp32_head = True
            different = torch.count_nonzero(logits != reference).item()
            delta = (logits.float() - reference.float()).abs()
            max_abs = delta.max().item()
            close = torch.isclose(logits, reference, rtol=1e-5, atol=1e-5).all().item()
            same_top1 = torch.equal(logits.argmax(dim=-1), reference.argmax(dim=-1))
            logger.info(
                "HY4 LM_HEAD_CHECK rows=%d different=%d max_abs=%g close=%s same_top1=%s",
                hidden_states.shape[0], different, max_abs, close, same_top1,
            )
        return logits

    def custom_sampler(self, sampler):
        if (
            self._reuse_eager_embeddings
            and self.vllm_config.speculative_config is None
            and os.getenv("HY4_PREFILL_FAST_GREEDY") == "1"
        ):
            from vllm.v1.worker.gpu.sample.sampler import Sampler

            if type(sampler) is Sampler:
                from vllm_ascend.worker.v2.hy4_prefill_greedy import PrefillSampler

                return PrefillSampler(sampler, fuse_counts=os.getenv("HY4_PREFILL_FUSED_COUNTS") == "1"), None
        return super().custom_sampler(sampler)


    def prepare_attn(self, input_batch, cudagraph_mode, *args, **kwargs):
        self._graph_decode = cudagraph_mode == CUDAGraphMode.FULL
        return super().prepare_attn(input_batch, cudagraph_mode, *args, **kwargs)

    def prefill_vocab_parameters(self):
        embedding = self._prefill_vocab_embedding
        return None if embedding is None else embedding.hy4_prefill_vocab_parameters()

    def prepare_inputs_embeds(self, scheduled_encoder_inputs, input_batch, req_states) -> torch.Tensor:
        num_tokens = input_batch.num_tokens
        if (
            not self._reuse_eager_embeddings
            or num_tokens != input_batch.num_tokens_after_padding
            or not self.supports_mm_inputs
            or self.prompt_embeds_state is not None
            or self.mm_pruner is not None
        ):
            return super().prepare_inputs_embeds(scheduled_encoder_inputs, input_batch, req_states)

        set_padded_tokens = getattr(self.model, "set_embedding_num_tokens_padded", None)
        if set_padded_tokens is not None:
            set_padded_tokens(num_tokens)
        input_ids = input_batch.input_ids[:num_tokens]
        self.execute_mm_encoder(scheduled_encoder_inputs)
        mm_embeds, is_mm_embed = self.gather_mm_embeddings(input_batch)
        indices = getattr(input_batch, "hy4_vocab_indices", None)
        needs_mm_token_mask = is_mm_embed is not None and getattr(self.model, "_has_oov_mm_tokens", False)
        if indices is not None and not mm_embeds and not needs_mm_token_mask:
            embeddings = self._prefill_vocab_embedding.embed_hy4_local_indices(indices)
            input_batch.hy4_text_vocab_embeddings = True
        else:
            embeddings = self.model.embed_input_ids(
                input_ids, multimodal_embeddings=mm_embeds, is_multimodal=is_mm_embed,
            )
        buffer = self.encoder_runner.inputs_embeds

        # Eager Prefill consumes the newly produced embeddings immediately.
        return embeddings.to(device=buffer.device, dtype=buffer.dtype).contiguous()

    def prepare_engram_dummy_inputs(self, num_reqs, num_tokens):
        if self._graph_hashes is None:
            return {}
        return {
            "prepared_engram": (
                self._graph_hashes[:, :num_tokens],
                tuple(rows[:num_tokens] for rows in self._graph_rows),
            )
        }

    def prepare_inputs(self, input_batch, req_states):
        if self._reuse_eager_embeddings:
            return self._prepare_prefill_inputs(input_batch, req_states)
        return super().prepare_inputs(input_batch, req_states)

    def prepare_engram_inputs(self, input_batch, req_states):
        depth = self.model.token_lookback_depth
        if depth == 0:
            return {}
        history = getattr(input_batch, "hy4_lookback_token_ids", None)
        if history is not None:
            return {"lookback_token_ids": history}
        history = torch.empty((input_batch.num_reqs, depth), dtype=torch.int32, device=self.device)
        if input_batch.is_dummy:
            history.fill_(-1)
        elif input_batch.num_reqs:
            all_token_ids = req_states.all_token_ids.gpu
            _gather_lookback_kernel[(input_batch.num_reqs,)](
                history,
                input_batch.idx_mapping,
                req_states.num_computed_tokens.gpu,
                all_token_ids,
                all_token_ids.stride(0),
                input_batch.num_reqs,
                DEPTH=depth,
                BLOCK_DEPTH=triton.next_power_of_2(depth),
            )
        if self._graph_decode:
            num_tokens = input_batch.num_tokens_after_padding
            inputs = self.prepare_engram_dummy_inputs(input_batch.num_reqs_after_padding, num_tokens)
            hashes, rows = inputs["prepared_engram"]
            if input_batch.is_dummy:
                hashes.fill_(-1)
                for output in rows:
                    output.zero_()
            else:
                self._flash_model.engram_hash.prepare_batch_hashes(
                    input_batch.input_ids,
                    history,
                    input_batch.query_start_loc_np[: input_batch.num_reqs + 1],
                    hashes,
                )
                self._prepare_engram_rows(hashes, rows)
            return inputs
        return {"lookback_token_ids": history}

    def _prepare_engram_rows(self, hashes, rows):
        embeddings = self._engram_embeddings
        if rows and all(
            output.shape == rows[0].shape
            and embedding.embedding.weight.device.type == "cpu"
            and embedding.embedding.weight.dtype == torch.bfloat16
            and not embedding.fp8
            for embedding, output in zip(embeddings, rows, strict=True)
        ):
            # Each layer keeps its existing shard and clamp rules. Transfer the
            # hashes and native BF16 rows together to avoid per-layer barriers.
            host_hashes = hashes.cpu()
            host_rows = [
                embedding.gather_rows(layer_hashes)[0]
                for embedding, layer_hashes in zip(embeddings, host_hashes, strict=True)
            ]
            device_rows = torch.stack(host_rows).to(hashes.device)
            for output, layer_rows in zip(rows, device_rows, strict=True):
                output.copy_(layer_rows)
            logger.info_once("HYV4 Decode batches CPU-offloaded BF16 Engram lookups")
            return

        for embedding, layer_hashes, output in zip(embeddings, hashes, rows, strict=True):
            output.copy_(embedding(layer_hashes))
