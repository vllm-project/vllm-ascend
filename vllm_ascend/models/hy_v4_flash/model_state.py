# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.config import CUDAGraphMode
from vllm.logger import init_logger
from vllm.triton_utils import triton

from vllm_ascend.ops.triton.engram_lookback import _gather_lookback_kernel
from vllm_ascend.worker.v2.model_states.default import AscendModelState

from .aclgraph import FlashAclGraphManager

logger = init_logger(__name__)


class FlashModelState(AscendModelState):
    aclgraph_manager_cls = FlashAclGraphManager

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        self._graph_decode = False
        self._graph_hashes = None
        if model.token_lookback_depth and vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self._flash_model = getattr(model, "language_model", model).model
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

    def prepare_attn(self, input_batch, cudagraph_mode, *args, **kwargs):
        self._graph_decode = cudagraph_mode == CUDAGraphMode.FULL
        return super().prepare_attn(input_batch, cudagraph_mode, *args, **kwargs)

    def prepare_engram_dummy_inputs(self, num_reqs, num_tokens):
        if self._graph_hashes is None:
            return {}
        return {
            "prepared_engram": (
                self._graph_hashes[:, :num_tokens],
                tuple(rows[:num_tokens] for rows in self._graph_rows),
            )
        }

    def prepare_engram_inputs(self, input_batch, req_states):
        depth = self.model.token_lookback_depth
        if depth == 0:
            return {}
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
