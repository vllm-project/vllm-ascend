# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 MLA-only additions to the shared Ascend DSpark runtime."""

from contextlib import contextmanager

import torch
import vllm.v1.worker.gpu.spec_decode.dflash.cudagraph as dflash_cudagraph

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.utils import get_rotation_path
from vllm_ascend.worker.v2.attn_utils import (
    build_attn_metadata,
    build_attn_metadata_wrapper,
    build_draft_attn_metadata_factory,
)
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator


class AscendMLADSparkSpeculator(AscendDSparkSpeculator):
    """Reuse shared loading, input preparation, graph dispatch and replay.

    Unlike the GQA draft, K3 MLA consumes post-layer raw prefix sums and
    stores FIA query lengths in ``metadata.decode``.
    """

    def load_draft_model(self, target_model, target_attn_layer_names):
        config = self.draft_model_config.hf_config
        rotation_path = get_rotation_path(self.vllm_config)
        had_rotation = hasattr(config, "_ascend_target_rotation_path")
        previous_rotation = getattr(config, "_ascend_target_rotation_path", None)
        config._ascend_target_rotation_path = rotation_path
        try:
            model = super().load_draft_model(target_model, target_attn_layer_names)
        finally:
            if had_rotation:
                config._ascend_target_rotation_path = previous_rotation
            else:
                delattr(config, "_ascend_target_rotation_path")

        target = target_model.get_language_model() if hasattr(target_model, "get_language_model") else target_model
        setter = getattr(target, "set_dspark_aux_capture_materialized", None)
        if setter is None:
            raise ValueError("K3 MLA DSpark requires a target supporting raw-prefix-sum auxiliary capture.")
        target_layers = getattr(config, "dspark_target_layer_ids", None) or getattr(config, "target_layer_ids", None)
        if not target_layers:
            raise ValueError("K3 MLA DSpark requires target_layer_ids.")
        boundaries = tuple(int(layer) + 1 for layer in target_layers)
        if len(set(boundaries)) != len(boundaries) or any(
            layer <= 0 or layer > target.model.config.num_hidden_layers for layer in boundaries
        ):
            raise ValueError(f"Invalid K3 MLA target layer boundaries: {boundaries}.")
        if tuple(target.model.aux_hidden_state_layers) != boundaries:
            raise ValueError("K3 MLA draft and target auxiliary layer boundaries do not match.")
        if target.model.config.hidden_size != config.target_hidden_size:
            raise ValueError("K3 MLA draft and target hidden sizes do not match.")
        if getattr(config, "num_target_layers", len(boundaries)) != len(boundaries):
            raise ValueError("K3 MLA num_target_layers does not match target_layer_ids.")
        setter(False)
        return model

    @contextmanager
    def draft_capture_context(self):
        """Supply the MLA RoPE positions omitted by upstream graph capture."""
        original = dflash_cudagraph.build_attn_metadata

        def build_mla_metadata(*args, **kwargs):
            kwargs["positions"] = self.input_buffers.positions[: kwargs["num_tokens"]]
            kwargs["is_prefilling"] = torch.zeros(kwargs["num_reqs"], dtype=torch.bool)
            kwargs["attn_state"] = AscendAttentionState.SpecDecoding
            return build_attn_metadata(*args, **kwargs)

        try:
            dflash_cudagraph.build_attn_metadata = build_mla_metadata
            yield
        finally:
            dflash_cudagraph.build_attn_metadata = original

    def build_draft_attn_metadatas(self, num_reqs_padded, seq_lens_cpu_upper_bound):
        assert self.input_batch is not None
        # The query graph contains only the speculative query block, even
        # when its context was produced by a target prefill. Padded rows must
        # not inherit prefill flags from a previous batch.
        with (
            build_attn_metadata_wrapper(),
            build_draft_attn_metadata_factory(
                self.input_buffers.positions,
                num_reqs_padded * self.num_query_per_req,
                torch.zeros(num_reqs_padded, dtype=torch.bool),
            ),
        ):
            metadata = self._build_draft_attn_metadata(
                num_reqs=self.input_batch.num_reqs,
                num_reqs_padded=num_reqs_padded,
                num_tokens_padded=num_reqs_padded * self.num_query_per_req,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=self.num_query_per_req,
                causal=self._group_causal,
            )
        return [self._update_draft_attn_metadata(metadata, num_reqs_padded)]

    def _update_draft_attn_metadata(self, attn_metadata, num_reqs_padded):
        query_lengths = [(i + 1) * self.num_query_per_req for i in range(num_reqs_padded)]
        if not attn_metadata:
            raise RuntimeError("K3 MLA draft graph requires attention metadata.")
        for name, metadata in attn_metadata.items():
            decode = getattr(metadata, "decode", None)
            if decode is None or not hasattr(decode, "actual_seq_lengths_q"):
                raise TypeError(f"K3 MLA draft layer {name} has no decode query-length metadata.")
            decode.actual_seq_lengths_q = query_lengths
        return attn_metadata
