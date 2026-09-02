#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend._310p.attention.attention_mask import (
    AttentionMaskBuilder310,
    is_compressed_mask_supported,
)
from vllm_ascend._310p.attention.dflash_hybrid_draft_graph_safe_attention import (
    DFlashHybridDraftAttentionInputs310,
    create_dflash_hybrid_draft_attention_inputs_310,
    update_dflash_hybrid_draft_attention_inputs_310,
)
from vllm_ascend._310p.dflash_full_and_piecewise import (
    is_310p_dflash_full_and_piecewise,
)
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

QUERY_LENS_CPU_ATTR = "query_lens_cpu"
DFLASH_HYBRID_DRAFT_ATTENTION_INPUTS_ATTR = (
    "_dflash_hybrid_draft_attention_inputs_310"
)


@dataclass(frozen=True)
class _DFlashHybridDraftCaptureScope310:
    real_num_reqs: int
    capacity_tokens: int


_DFLASH_HYBRID_DRAFT_CAPTURE_SCOPE_310: ContextVar[
    _DFlashHybridDraftCaptureScope310 | None
] = ContextVar(
    "dflash_hybrid_draft_capture_scope_310",
    default=None,
)


@contextmanager
def dflash_hybrid_draft_capture_scope_310(
    *,
    real_num_reqs: int,
    capacity_tokens: int,
) -> Iterator[None]:
    """Mark only the 310P Hybrid Draft dummy-capture metadata build."""
    if real_num_reqs <= 0 or capacity_tokens <= 0:
        raise ValueError("Draft FULL capture capacities must be positive")
    token = _DFLASH_HYBRID_DRAFT_CAPTURE_SCOPE_310.set(
        _DFlashHybridDraftCaptureScope310(
            real_num_reqs=real_num_reqs,
            capacity_tokens=capacity_tokens,
        )
    )
    try:
        yield
    finally:
        _DFLASH_HYBRID_DRAFT_CAPTURE_SCOPE_310.reset(token)


def set_query_lens_cpu(attn_metadata: AscendMetadata, query_lens_cpu: torch.Tensor) -> None:
    """Attach host qLens for ATB splitfuse without extending upstream AscendMetadata."""
    setattr(attn_metadata, QUERY_LENS_CPU_ATTR, query_lens_cpu)


def get_query_lens_cpu(attn_metadata: AscendMetadata) -> torch.Tensor | None:
    value = getattr(attn_metadata, QUERY_LENS_CPU_ATTR, None)
    if value is None:
        return None
    return value


def get_dflash_hybrid_draft_attention_inputs_310(
    attn_metadata: AscendMetadata,
) -> DFlashHybridDraftAttentionInputs310 | None:
    return getattr(
        attn_metadata,
        DFLASH_HYBRID_DRAFT_ATTENTION_INPUTS_ATTR,
        None,
    )


class AscendAttentionMetadataBuilder310(AscendAttentionMetadataBuilder):
    """
    Metadata builder specialized for the Huawei Ascend 310P NPU.

    This class extends the base Ascend attention metadata builder to use
    the 310P-specific attention mask builder, ensuring that masks are
    generated in the correct format (FRACTAL_NZ) and logic required by
    the 310P hardware.
    """

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        """
        Initializes the metadata builder and the 310P-specific mask builder.

        Args:
            kv_cache_spec (AttentionSpec): Specification for the KV cache (block size, etc.).
            layer_names (list[str]): List of layer names in the model.
            vllm_config (VllmConfig): Global vLLM configuration object.
            device (torch.device): The device (NPU) to run operations on.
        """
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # The public builder does not expose the config as an instance field.
        # Keep a private 310P reference so the exact Hybrid-only adapter can be
        # selected without changing any public builder contract.
        self._vllm_config_310 = vllm_config

        # Override the mask builder with the 310P-specific version
        max_model_len = vllm_config.model_config.max_model_len
        self.attn_mask_builder: Any = AttentionMaskBuilder310(self.device, max_model_len)

        self._query_lens_cpu_buffer: torch.Tensor | None = None
        if device.type != "cpu":
            max_num_seqs = vllm_config.scheduler_config.max_num_seqs
            self._query_lens_cpu_buffer = torch.empty(max_num_seqs, dtype=torch.int32, device="cpu", pin_memory=True)

    def _prepare_dflash_hybrid_draft_attention_inputs_310(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        *,
        draft_step: int,
        real_num_reqs_override: int | None = None,
        capacity_tokens_override: int | None = None,
    ) -> DFlashHybridDraftAttentionInputs310:
        """Build one step's runtime source for the shared Draft FULL graph."""
        valid_num_tokens = int(common_attn_metadata.num_actual_tokens)
        query_width = int(
            getattr(common_attn_metadata, "decode_token_per_req", 0)
            or getattr(common_attn_metadata, "max_query_len", 0)
        )
        if real_num_reqs_override is None:
            if query_width <= 0 or valid_num_tokens % query_width:
                raise RuntimeError(
                    "310P DFlash Hybrid Draft cannot derive logical request "
                    f"count: tokens={valid_num_tokens}, "
                    f"query_width={query_width}"
                )
            valid_num_reqs = valid_num_tokens // query_width
        else:
            valid_num_reqs = int(real_num_reqs_override)
        capacity_tokens = int(
            common_attn_metadata.num_input_tokens
            if capacity_tokens_override is None
            else capacity_tokens_override
        )
        capacity_reqs = int(common_attn_metadata.block_table_tensor.shape[0])
        max_blocks = int(common_attn_metadata.block_table_tensor.shape[1])
        if valid_num_reqs <= 0:
            raise RuntimeError("310P DFlash Hybrid Draft has no logical request")
        if common_attn_metadata.query_start_loc.shape[0] < valid_num_reqs + 1:
            raise RuntimeError(
                "310P DFlash Hybrid Draft query_start_loc does not cover "
                "logical requests"
            )

        device = common_attn_metadata.query_start_loc.device
        cache_key = (
            int(draft_step),
            capacity_reqs,
            capacity_tokens,
            max_blocks,
            device.type,
            device.index,
        )
        cache = getattr(
            self,
            "_dflash_hybrid_draft_attention_input_cache_310",
            None,
        )
        if cache is None:
            cache = {}
            self._dflash_hybrid_draft_attention_input_cache_310 = cache
        inputs = cache.get(cache_key)
        if inputs is None:
            inputs = create_dflash_hybrid_draft_attention_inputs_310(
                capacity_reqs=capacity_reqs,
                capacity_tokens=capacity_tokens,
                max_blocks=max_blocks,
                device=device,
            )
            cache[cache_key] = inputs

        query_lens = (
            common_attn_metadata.query_start_loc[1 : valid_num_reqs + 1]
            - common_attn_metadata.query_start_loc[:valid_num_reqs]
        ).to(torch.int32)
        seq_lens = common_attn_metadata.seq_lens[:valid_num_reqs].to(
            torch.int32
        )
        block_table = common_attn_metadata.block_table_tensor[
            :valid_num_reqs
        ].to(torch.int32)
        update_dflash_hybrid_draft_attention_inputs_310(
            inputs,
            query_lens=query_lens,
            seq_lens=seq_lens,
            block_table=block_table,
            valid_num_reqs=valid_num_reqs,
            valid_num_tokens=valid_num_tokens,
        )
        return inputs

    def _fill_query_lens_cpu(
        self, num_reqs: int, query_start_loc_cpu: torch.Tensor, is_drafting: bool = False
    ) -> torch.Tensor:
        """Pinned CPU per-request query lengths for ATB splitfuse (host qLensTensor)."""
        if self._query_lens_cpu_buffer is None:
            return (query_start_loc_cpu[1 : num_reqs + 1] - query_start_loc_cpu[:num_reqs]).contiguous()
        if is_drafting:
            # We are using the same buffer for multi step drafting,
            # so we have to clone the buffer or the q lens of step 0
            # will be overwritten by the following steps.
            buffer = self._query_lens_cpu_buffer[:num_reqs].clone()
        else:
            buffer = self._query_lens_cpu_buffer[:num_reqs]
        torch.sub(
            query_start_loc_cpu[1 : num_reqs + 1],
            query_start_loc_cpu[:num_reqs],
            out=buffer,
        )
        return buffer

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        is_drafting: bool = False,
        dflash_hybrid_draft_step: int = -1,
    ) -> AscendMetadata:
        private_draft_inputs = None
        capture_scope = _DFLASH_HYBRID_DRAFT_CAPTURE_SCOPE_310.get()
        hybrid_draft = (
            (is_drafting or capture_scope is not None)
            and not common_attn_metadata.causal
            and is_310p_dflash_full_and_piecewise(self._vllm_config_310)
        )
        logger.debug(
            "[310p-dflash-full-and-piecewise/draft-metadata] "
            "event=builder-route builder=%s is_drafting=%s causal=%s "
            "hybrid=%s step=%d",
            type(self).__name__,
            is_drafting,
            common_attn_metadata.causal,
            hybrid_draft,
            dflash_hybrid_draft_step,
        )
        if hybrid_draft:
            private_draft_inputs = (
                self._prepare_dflash_hybrid_draft_attention_inputs_310(
                    common_attn_metadata,
                    draft_step=dflash_hybrid_draft_step,
                    real_num_reqs_override=(
                        capture_scope.real_num_reqs
                        if capture_scope is not None
                        else None
                    ),
                    capacity_tokens_override=(
                        capture_scope.capacity_tokens
                        if capture_scope is not None
                        else None
                    ),
                )
            )
        attn_metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)

        if private_draft_inputs is not None:
            setattr(
                attn_metadata,
                DFLASH_HYBRID_DRAFT_ATTENTION_INPUTS_ATTR,
                private_draft_inputs,
            )

        num_reqs = common_attn_metadata.num_reqs

        # ATB flash attention (PrefillNoCache) reads seqLen from host data to build
        # its tiling. The base builder's parallel-drafting branch overrides both
        # seq_lens and seq_lens_cpu with a device tensor (kept device-side for graph
        # replay on other platforms), which has no hostData and crashes ATB with
        # "tensor.hostData is null". Re-attach the host seq_lens that already exists
        # on CPU (no extra D2H sync) so forward_prefill_310 can feed ATB directly.
        # For the main model seq_lens_cpu is already host, so this is a no-op.
        if common_attn_metadata._seq_lens_cpu is not None:
            attn_metadata.seq_lens_cpu = common_attn_metadata._seq_lens_cpu[:num_reqs]
        elif common_attn_metadata.seq_lens_cpu is not None:
            attn_metadata.seq_lens_cpu = common_attn_metadata.seq_lens_cpu[:num_reqs]

        splitfuse_states = (
            AscendAttentionState.SpecDecoding,
            AscendAttentionState.ChunkedPrefill,
        )
        if attn_metadata.attn_state not in splitfuse_states:
            return attn_metadata

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]
        # ATB splitfuse qLensTensor must be host; filled here (outside graph forward).
        set_query_lens_cpu(
            attn_metadata,
            self._fill_query_lens_cpu(num_reqs, query_start_loc_cpu, is_drafting),
        )

        # Bind device-side views for in-place graph replay updates.
        attn_metadata.seq_lens = common_attn_metadata.seq_lens[:num_reqs]
        attn_metadata.query_start_loc = common_attn_metadata.query_start_loc[: num_reqs + 1]

        if is_compressed_mask_supported():
            if common_attn_metadata.causal:
                attn_metadata.attn_mask = AttentionMaskBuilder310.get_compressed_splitfuse_mask(self.device)
            else:
                attn_metadata.attn_mask = AttentionMaskBuilder310.get_compressed_non_causal_splitfuse_mask(
                    self.device
                )

        return attn_metadata

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
    ):
        # override build_for_drafting for passing status.
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            fast_build=True,
            is_drafting=True,
            dflash_hybrid_draft_step=draft_index,
        )
