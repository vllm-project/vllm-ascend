# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN backend for the shared RecoverSSM metadata lifecycle."""

from dataclasses import dataclass, field, fields

import torch
from vllm.config import VllmConfig
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.attention.gdn_attn_builder import (
    AscendGDNAttentionBackend,
    AscendGDNAttentionMetadataBuilder,
)
from vllm_ascend.attention.recoverssm_metadata import (
    recoverssm_request_indices,
    recoverssm_spec_rows,
)
from vllm_ascend.models.glm5next.recoverssm_metadata import AscendKDARecoverSSMMetadataBuilder
from vllm_ascend.ops.triton.gdn.recoverssm_gdn import GDNRecoverSSMCommitContext


@dataclass
class AscendGDNRecoverSSMCommitMetadata:
    state_indices: torch.Tensor
    query_start_loc: torch.Tensor
    request_indices: torch.Tensor | None
    block_table: torch.Tensor | None
    num_computed_tokens: torch.Tensor | None
    block_size: int | None


@dataclass
class AscendGDNRecoverSSMMetadata(GDNAttentionMetadata, RecoverSSMMetadata):
    recoverssm_commit: AscendGDNRecoverSSMCommitMetadata | None = None
    recoverssm_context: GDNRecoverSSMCommitContext | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def commit_recoverssm_state(
        self,
        num_accepted_tokens: torch.Tensor,
    ) -> RecoverSSMPostprocessMetadata | None:
        commit = self.recoverssm_commit
        if commit is None:
            return None
        context = self.recoverssm_context
        assert context is not None
        num_spec_decodes = self.num_spec_decodes
        context.commit(
            num_accepted_tokens,
            commit.state_indices[:num_spec_decodes, 0],
            commit.query_start_loc[: num_spec_decodes + 1],
            request_indices=commit.request_indices,
            block_table=commit.block_table,
            num_computed_tokens=commit.num_computed_tokens,
            mamba_block_size=commit.block_size,
        )
        if commit.block_table is None:
            return None
        assert commit.num_computed_tokens is not None
        assert commit.block_size is not None
        return RecoverSSMPostprocessMetadata(
            num_spec_decodes=num_spec_decodes,
            request_indices=commit.request_indices,
            block_table=commit.block_table,
            num_computed_tokens=commit.num_computed_tokens,
            block_size=commit.block_size,
        )


class AscendGDNRecoverSSMMetadataBuilder(AscendGDNAttentionMetadataBuilder):
    """Build GDN RecoverSSM metadata without inheriting the KDA builder."""

    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    # TODO: Remove this temporary compatibility binding once the pinned vLLM
    # includes #61309 or equivalent native GDN RecoverSSM metadata support.
    _normalize_graph_padding = AscendKDARecoverSSMMetadataBuilder._normalize_graph_padding

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._ones = torch.ones(
            self.spec_state_indices_tensor.shape[0],
            dtype=torch.int32,
            device=self.spec_state_indices_tensor.device,
        )
        self._context: GDNRecoverSSMCommitContext | None = None
        self._capturing = False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
        num_actual_reqs: int | None = None,
    ) -> AscendGDNRecoverSSMMetadata:
        m, num_decode_draft_tokens_cpu = self._normalize_graph_padding(
            common_attn_metadata,
            num_actual_reqs,
            num_decode_draft_tokens_cpu,
        )
        rows = None
        if self.use_spec_decode and num_decode_draft_tokens_cpu is not None:
            if self._capturing and m.is_prefilling is None:
                m = m.replace(is_prefilling=torch.zeros(m.num_reqs, dtype=torch.bool))
            if m.is_prefilling is None or m.is_prefilling.device.type != "cpu":
                raise ValueError("GDN RecoverSSM needs the CPU is_prefilling mask")
            num_reqs = m.num_reqs if num_actual_reqs is None else num_actual_reqs
            num_decode_draft_tokens_cpu, rows = recoverssm_spec_rows(
                m.is_prefilling,
                m.query_start_loc_cpu,
                num_decode_draft_tokens_cpu,
                num_reqs,
            )

        metadata = super().build(
            common_prefix_len,
            m,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            fast_build=fast_build,
            num_actual_reqs=num_actual_reqs,
        )
        base = {item.name: getattr(metadata, item.name) for item in fields(metadata)}
        commit = None
        if metadata.num_spec_decodes > 0:
            if rows is None:
                raise ValueError("GDN RecoverSSM: spec rows without draft counts")
            assert metadata.spec_state_indices_tensor is not None
            assert metadata.spec_query_start_loc is not None
            state_indices = metadata.spec_state_indices_tensor[:, :1]
            accepted_tokens = self._ones[: state_indices.shape[0]]
            base["spec_state_indices_tensor"] = state_indices
            base["num_accepted_tokens"] = accepted_tokens

            # The Ascend base builder creates convolution descriptors before
            # this RecoverSSM specialization. Point them at the checkpoint-only
            # state column and the one-token accepted-count buffer as well.
            spec_decode_metadata = getattr(metadata, "spec_decode_metadata", None)
            if spec_decode_metadata is not None:
                conv_metadata = spec_decode_metadata.spec_causal_conv1d
                num_rows = conv_metadata.query_start_loc.shape[0] - 1
                conv_metadata.cache_indices = state_indices[:num_rows, 0]
                conv_metadata.num_accepted_tokens = accepted_tokens[:num_rows]

            request_indices = recoverssm_request_indices(
                rows,
                metadata.num_spec_decodes,
                m.query_start_loc.device,
            )
            align = self.vllm_config.cache_config.mamba_cache_mode == "align"
            commit = AscendGDNRecoverSSMCommitMetadata(
                state_indices=state_indices,
                query_start_loc=metadata.spec_query_start_loc,
                request_indices=request_indices,
                block_table=m.block_table_tensor if align else None,
                num_computed_tokens=m.compute_num_computed_tokens() if align else None,
                block_size=self.kv_cache_spec.block_size if align else None,
            )

        result = AscendGDNRecoverSSMMetadata(
            **base,
            recoverssm_commit=commit,
            recoverssm_context=(self._get_context() if commit is not None else None),
        )
        # Ascend GDN attaches device-specific convolution and PCP descriptors
        # dynamically rather than declaring them on the upstream dataclass.
        for name, value in vars(metadata).items():
            if name not in base:
                setattr(result, name, value)
        result.pcp_context = getattr(common_attn_metadata, "pcp_context", None)
        return result

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ):
        self._capturing = True
        try:
            return super().build_for_cudagraph_capture(common_attn_metadata)
        finally:
            self._capturing = False

    def _get_context(self):
        if self._context is None:
            layers = self.vllm_config.compilation_config.static_forward_context
            self._context = GDNRecoverSSMCommitContext.create(
                [layers[name] for name in self.layer_names],
                spec_query_len=self.num_spec + 1,
                max_num_reqs=self.vllm_config.scheduler_config.max_num_seqs,
            )
        return self._context


class AscendGDNRecoverSSMAttentionBackend(AscendGDNAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_RECOVERSSM"

    @staticmethod
    def get_builder_cls() -> type[AscendGDNRecoverSSMMetadataBuilder]:
        return AscendGDNRecoverSSMMetadataBuilder
