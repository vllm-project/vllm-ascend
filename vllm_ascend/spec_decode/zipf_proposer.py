# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The vLLM team.
#
# This file is a part of the vllm-ascend project.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import logger

from vllm_ascend.spec_decode.config import get_zipf_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class AscendZipfDecodingProposer:
    """Zipf-based speculative decoding proposer for the Ascend V1 runner."""

    def __init__(self, vllm_config: VllmConfig, runner: NPUModelRunner):
        config = vllm_config.speculative_config
        assert config is not None
        zipf_config = get_zipf_config(vllm_config)

        self.runner = runner
        self.num_speculative_tokens = zipf_config["zipf_initial_speculative_tokens"]
        self.max_speculative_tokens = config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.min_window = zipf_config["zipf_min_window"]
        self.max_window = zipf_config["zipf_ngram_size"]
        self.skip_shared = zipf_config["zipf_skip_shared"]
        self.ema_alpha = 0.3

        from vllm_ascend.spec_decode.zipf_cache import ZipfCache

        logger.info(
            "Creating Ascend Zipf cache: min_window=%d, max_window=%d, skip_shared=%s",
            self.min_window,
            self.max_window,
            self.skip_shared,
        )
        self.zipf_cache = ZipfCache(
            min_window=self.min_window,
            max_window=self.max_window,
            skip_shared=self.skip_shared,
            generalized_before_shared=zipf_config["zipf_generalized_before_shared"],
        )

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        pass

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool | None = None,
        in_graph_capturing: bool | None = None,
        num_reqs: int | None = None,
        num_tokens_across_dp: int | None = None,
        aclgraph_runtime_mode: Any | None = None,
        batch_descriptor: Any | None = None,
        dummy_compute_logits: Any = lambda hidden_states: None,
        is_profile: bool = False,
    ) -> None:
        pass

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        *args: Any,
        **kwargs: Any,
    ) -> list[list[int]]:
        input_batch = self.runner.input_batch
        batch_size = min(len(sampled_token_ids), input_batch.num_reqs)

        req_hashes: list[int] = []
        token_ids_rows = []
        num_tokens_list: list[int] = []
        num_prompt_list: list[int] = []
        valid_mask: list[bool] = []
        c_sampled: list[list[int]] = []
        c_to_orig: list[int] = []

        for req_index in range(batch_size):
            sampled_ids = sampled_token_ids[req_index]
            if not sampled_ids:
                continue

            req_id = input_batch.req_ids[req_index]
            if req_id in input_batch.spec_decode_unsupported_reqs:
                continue

            num_tokens = int(input_batch.num_tokens_no_spec[req_index])
            if num_tokens >= input_batch.max_model_len:
                continue

            req_hashes.append(hash(req_id) & 0xFFFFFFFFFFFFFFFF)
            token_ids_rows.append(input_batch.token_ids_cpu[req_index])
            num_tokens_list.append(num_tokens)
            num_prompt_list.append(int(input_batch.num_prompt_tokens[req_index]))
            c_sampled.append(sampled_ids)
            valid_mask.append(True)
            c_to_orig.append(req_index)

        draft_token_ids: list[list[int]] = [[] for _ in range(len(sampled_token_ids))]

        if req_hashes:
            batch_results = self.zipf_cache.propose_batch(
                req_hashes,
                token_ids_rows,
                num_tokens_list,
                num_prompt_list,
                c_sampled,
                valid_mask,
                self.max_model_len,
                self.num_speculative_tokens,
                self.ema_alpha,
                self.max_speculative_tokens,
            )
            for result_index, orig_index in enumerate(c_to_orig):
                draft = batch_results[result_index]
                if draft:
                    draft_token_ids[orig_index] = list(draft)

        return draft_token_ids

    def get_stats(self) -> dict[str, Any]:
        stats = self.zipf_cache.stats()
        return {
            "query_count": stats["query_count"],
            "shared_hit_count": stats["shared_hit_count"],
            "local_hit_count": stats["local_hit_count"],
            "shared_hit_rate": stats["shared_hit_rate"],
            "local_hit_rate": stats["local_hit_rate"],
            "total_hit_rate": stats["total_hit_rate"],
            "update_count": stats["update_count"],
            "shared_entries": stats["shared_entries"],
            "local_hashes": stats["local_hashes"],
            "local_entries": stats["local_entries"],
            "local_memory_mb": stats["local_memory_mb"],
        }
