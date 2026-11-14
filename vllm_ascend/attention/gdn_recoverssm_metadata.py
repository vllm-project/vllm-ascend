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


# TODO: Once the pinned vLLM revision includes #61309, might replace this class
# with upstream GDNRecoverSSMCommitMetadata.
# Keep supplying the Ascend convolution descriptors from the builder.
@dataclass
class AscendGDNRecoverSSMCommitMetadata:
    state_indices: torch.Tensor
    query_start_loc: torch.Tensor
    request_indices: torch.Tensor | None
    block_table: torch.Tensor | None
    num_computed_tokens: torch.Tensor | None
    block_size: int | None


# TODO: Once the pinned vLLM revision includes #61309, might replace this class
# with upstream GDNRecoverSSMMetadata.
# Keep supplying the Ascend NPU commit context through the same commit() interface.
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

        meta = super().build(
            common_prefix_len,
            m,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            fast_build=fast_build,
            num_actual_reqs=num_actual_reqs,
        )
        base = {f.name: getattr(meta, f.name) for f in fields(meta)}
        commit = None
        if meta.num_spec_decodes > 0:
            n = meta.num_spec_decodes
            assert meta.spec_state_indices_tensor is not None
            assert meta.spec_query_start_loc is not None
            base["spec_state_indices_tensor"] = meta.spec_state_indices_tensor[:, :1]
            # The checkpoint already holds the accepted state. Size the
            # one-token counts to the potentially graph-padded indices.
            base["num_accepted_tokens"] = self._ones[: base["spec_state_indices_tensor"].shape[0]]
            if rows is None:
                raise ValueError("GDN RecoverSSM: spec rows without draft counts")
            request_indices = recoverssm_request_indices(rows, n, m.query_start_loc.device)
            align = self.vllm_config.cache_config.mamba_cache_mode == "align"
            commit = AscendGDNRecoverSSMCommitMetadata(
                state_indices=base["spec_state_indices_tensor"],
                query_start_loc=meta.spec_query_start_loc,
                request_indices=request_indices,
                block_table=m.block_table_tensor if align else None,
                num_computed_tokens=m.compute_num_computed_tokens() if align else None,
                block_size=self.kv_cache_spec.block_size if align else None,
            )

            # Reuse the checkpoint column and one-token counts in the Ascend
            # convolution descriptor, which may have a different row count.
            spec_decode_metadata = getattr(meta, "spec_decode_metadata", None)
            if spec_decode_metadata is not None:
                conv_metadata = spec_decode_metadata.spec_causal_conv1d
                num_rows = conv_metadata.query_start_loc.shape[0] - 1
                conv_metadata.cache_indices = base["spec_state_indices_tensor"][:num_rows, 0]
                conv_metadata.num_accepted_tokens = base["num_accepted_tokens"][:num_rows]

        result_meta = AscendGDNRecoverSSMMetadata(
            **base,
            recoverssm_commit=commit,
            recoverssm_context=(self._get_context() if commit is not None else None),
        )
        self._attach_ascend_metadata(result_meta, meta, base, common_attn_metadata)
        return result_meta

    @staticmethod
    def _attach_ascend_metadata(
        result_meta: AscendGDNRecoverSSMMetadata,
        meta: GDNAttentionMetadata,
        base: dict[str, object],
        common_attn_metadata: CommonAttentionMetadata,
    ) -> None:
        # Ascend GDN attaches device-specific convolution and PCP descriptors
        # dynamically rather than declaring them on the upstream dataclass.
        for name, value in vars(meta).items():
            if name not in base:
                setattr(result_meta, name, value)
        result_meta.pcp_context = getattr(common_attn_metadata, "pcp_context", None)

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ):
        self._capturing = True
        try:
            return super().build_for_cudagraph_capture(common_attn_metadata)
        finally:
            self._capturing = False

    # TODO: Once the pinned vLLM revision includes #61309, reuse upstream context
    # creation and caching. Remove this override once the upstream path can
    # instantiate Ascend's GDNRecoverSSMCommitContext.
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
