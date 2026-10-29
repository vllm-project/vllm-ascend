# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from typing import Any, cast

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.worker.gpu.spec_decode.mamba_attn_hybrid.speculator import MambaAttnHybridSpeculator

from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator


class AscendMambaAttnHybridSpeculator(AscendDSparkSpeculator, MambaAttnHybridSpeculator):
    """H-Spec latent anchors and Markov sampling with Ascend attention/graphs.

    Cooperative inheritance runs upstream H-Spec initialization and model
    loading through Ascend's DSpark hooks. Sampling and the recurrent backbone
    remain the upstream implementations.
    """

    _speculator_name = "MambaAttnHybrid"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self._validate_config(vllm_config)
        super().__init__(vllm_config, device)
        self._exact_seq_lens_cpu: torch.Tensor | None = None

    @staticmethod
    def _validate_config(vllm_config: VllmConfig) -> None:
        spec = vllm_config.speculative_config
        assert spec is not None
        parallel = vllm_config.parallel_config
        if not vllm_config.use_v2_model_runner:
            raise ValueError("H-Spec on Ascend requires Model Runner V2.")
        if (
            parallel.pipeline_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
        ):
            raise ValueError("H-Spec shared target KV currently requires PP=PCP=DCP=1 on Ascend.")
        if spec.draft_parallel_config.tensor_parallel_size != parallel.tensor_parallel_size:
            raise ValueError("H-Spec must use the target tensor parallel size to share its KV cache.")
        if spec.kv_cache_dtype is not None and spec.kv_cache_dtype != vllm_config.cache_config.cache_dtype:
            raise ValueError("H-Spec must use the target KV cache dtype.")
        config = spec.draft_model_config.hf_config
        if spec.draft_model_config.get_hidden_size() != vllm_config.model_config.get_hidden_size():
            raise ValueError("H-Spec latent fusion requires equal target and draft hidden sizes.")
        if getattr(config, "sample_from_anchor", None) is not False:
            raise ValueError("H-Spec requires the 1+K query layout (sample_from_anchor=False).")
        if not config.eagle_aux_hidden_state_layer_ids:
            raise ValueError("H-Spec requires latent fusion auxiliary layer IDs.")
        if "attention" not in config.block_pattern:
            raise ValueError("H-Spec requires a target-KV attention layer.")

    def load_draft_model(self, target_model: torch.nn.Module, target_attn_layer_names: set[str]) -> torch.nn.Module:
        model = super().load_draft_model(target_model, target_attn_layer_names)
        attn_layers = get_layers_from_vllm_config(self.vllm_config, cast(type[Any], AttentionLayerBase))
        for layer in model.model.layers:
            if not hasattr(layer, "self_attn"):
                continue
            attention = layer.self_attn.attn
            target_name = attention.kv_sharing_target_layer_name
            if target_name not in target_attn_layer_names:
                raise ValueError(f"H-Spec cannot share unavailable target attention layer {target_name!r}.")
            target = attn_layers[target_name]
            if (
                attention.get_attn_backend() is not AscendAttentionBackend
                or target.get_attn_backend() is not AscendAttentionBackend
            ):
                raise ValueError("H-Spec target KV sharing requires the Ascend GQA attention backend.")
            for field in ("num_kv_heads", "head_size", "head_size_v", "kv_cache_dtype"):
                if getattr(attention, field) != getattr(target, field):
                    raise ValueError(f"H-Spec target/draft KV layout mismatch: {field}.")
            if target.impl.sliding_window is not None and attention.impl.sliding_window != target.impl.sliding_window:
                raise ValueError("H-Spec cannot read full history from a different target sliding window.")
            # The outer Attention alias binds the target cache. Unlike a
            # read-only KV-sharing layer, H-Spec writes candidate K/V into
            # lookahead slots following the last accepted target position.
            # Ascend's implementation must keep its own metadata/task name
            # and perform those writes; accepted target slots remain intact.
            attention.impl.kv_sharing_target_layer_name = None
        return model

    def _configure_decode_metadata_update(self) -> None:
        # ACL replay updates FIA host parameters using the exact KV lengths.
        self.supports_decode_metadata_update = False

    def propose(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._exact_seq_lens_cpu = None
        try:
            return super().propose(*args, **kwargs)
        finally:
            self._exact_seq_lens_cpu = None

    def _prepare_draft_dcp_metadata_inputs(
        self, num_reqs: int, num_reqs_padded: int, step: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Non-causal FIA consumes host KV lengths. Optimistic target lengths
        # include rejected tokens and would expose unwritten cache entries.
        # Copy the exact device lengths once per proposal, shared by the eager
        # metadata build and ACL graph update. No scalar device .item().
        if self._exact_seq_lens_cpu is None:
            self._exact_seq_lens_cpu = self.input_buffers.seq_lens[:num_reqs].to(device="cpu", copy=True)
        seq_lens = torch.zeros(num_reqs_padded, dtype=torch.int32)
        seq_lens[:num_reqs].copy_(self._exact_seq_lens_cpu)
        return seq_lens, torch.zeros(num_reqs_padded, dtype=torch.bool)
