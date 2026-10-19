from copy import copy
from dataclasses import dataclass
from enum import IntFlag
from typing import Any

import torch

from vllm_ascend.attention import dsa_v1
from vllm_ascend.attention.context_parallel.dsa_cp import (
    AscendDSAPCPMetadata,
    AscendDSAPCPMetadataBuilder,
)
from vllm_ascend.attention.dsa_attn_kv_plan import get_dsa_attn_kv_plan
from vllm_ascend.attention.dsa_v1 import build_dspark_swa_indices, get_dspark_sparse_sas_window
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    split_decodes_and_prefills,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.rope_dsv4 import (
    get_cos_and_sin_dsa,
    get_full_cos_and_sin_dsa,
)
from vllm_ascend.quantization.methods.kv_cache.turboquant import is_turboquant


class MetadataUse(IntFlag):
    STATE = 1
    SWA = 2
    ATTENTION = 4
    INDEXER = 8


@dataclass
class BatchScope:
    common: AscendCommonAttentionMetadata
    actual_reqs: int
    num_decodes: int
    num_prefills: int
    num_decode_tokens: int
    num_prefill_tokens: int

    rope: Any = None
    start_pos: torch.Tensor | None = None
    limits: tuple[int, int] | None = None
    qli: Any = None
    sas: dict | None = None
    op_cache: dict | None = None


class PCPMetadataPlan:
    def __init__(self, builders: list[dsa_v1.AscendDSAMetadataBuilder], uses: dict[int, MetadataUse], *, draft=False):
        self.is_draft = draft
        self.builders = tuple(sorted(builders, key=lambda b: tuple(b.layer_names)))
        self.uses = uses

        self.context_owner = self.builders[0]
        self.vllm_config = self.context_owner.vllm_config

        self.sas_owners: dict[int, AscendDSAPCPMetadataBuilder] = {}
        self.qli_owner = None

        for builder in self.builders:
            use = uses[id(builder)]
            ratio = builder.compressor_ratio

            if use & (MetadataUse.ATTENTION | MetadataUse.SWA):
                self.sas_owners.setdefault(ratio, builder)

            if use & MetadataUse.INDEXER:
                if ratio != 4:
                    raise ValueError("PCP metadata fast path requires C4 indexer queries")
                if self.qli_owner is None:
                    self.qli_owner = builder

        # The ordinary and DSpark SWA variants may both be live in one
        # forward. Give each a persistent output instead of overwriting
        # the same builder buffer when a different mask/window is built.
        self.sas_variant_owners = {}
        for ratio, owner in self.sas_owners.items():
            normal_window = (owner.model_config.hf_config.sliding_window - 1, 0)
            self.sas_variant_owners[(ratio, False, *normal_window)] = owner
            speculative = owner.vllm_config.speculative_config
            if speculative is not None and speculative.method == "dspark":
                special_owner = copy(owner)
                special_owner.sas_metadata_buffer = torch.zeros_like(owner.sas_metadata_buffer)
                special_window = get_dspark_sparse_sas_window(owner.vllm_config)
                self.sas_variant_owners[(ratio, True, *special_window)] = special_owner

        local_capacity = max(b.start_pos_prefill.numel() for b in self.builders)
        prototype = self.context_owner.start_pos_prefill
        self.start_buffers = {"local": prototype.new_zeros(local_capacity)}
        self.global_rope_buffers: dict = {}
        self.draft_rope_buffers: dict = {}
        if draft:
            self.restore_capacity = max(b.slot_mapping.shape[0] for b in self.builders)
        else:
            global_capacity = max(b._global_metadata_builder.start_pos_prefill.numel() for b in self.builders)
            self.start_buffers["global"] = prototype.new_zeros(global_capacity)
            self.restore_capacity = self.context_owner._hidden_restore_idx_buffer.numel()

    def enable(self) -> bool:
        return not any(
            b._device_metadata_enabled or (not self.is_draft and b._global_metadata_builder._device_metadata_enabled)
            for b in self.builders
        )

    def _scope(
        self,
        common: AscendCommonAttentionMetadata,
        actual_reqs: int | None,
    ) -> BatchScope:
        num_reqs: int = common.num_reqs
        resolved_actual_reqs: int = num_reqs if actual_reqs is None else min(actual_reqs, num_reqs)

        counts = split_decodes_and_prefills(
            common,
            decode_threshold=self.context_owner.decode_threshold,
        )
        num_decode, npref, ndtokens, nptokens = counts

        return BatchScope(
            common=common,
            actual_reqs=resolved_actual_reqs,
            num_decodes=num_decode,
            num_prefills=npref,
            num_decode_tokens=ndtokens,
            num_prefill_tokens=nptokens,
            sas={},
            op_cache={},
        )

    def _prepare_step(
        self,
        common: AscendCommonAttentionMetadata,
        pcp_context,
        cache_group_idx: int,
        actual_reqs: int | None,
        full_graph_mode: bool,
    ) -> dict[str, Any]:
        owner = self.context_owner
        has_prefill = bool(pcp_context.global_batch.is_prefilling_np.any())
        needs_global = has_prefill or owner._is_decode_sharded

        global_scope = None
        if needs_global:
            pcp_context = owner._prepare_graph_pcp_context(
                pcp_context,
                full_graph_mode=full_graph_mode,
            )

            global_common = owner._build_global_common_attn_metadata(
                pcp_context,
                cache_group_idx,
                common,
            )
            global_actual_reqs = pcp_context.global_batch.num_reqs

            global_common = owner._build_graph_common_attn_metadata(
                global_common,
                global_actual_reqs,
            )

            global_scope = self._scope(
                global_common,
                global_actual_reqs,
            )

        if common.num_actual_tokens == 0:
            actual_reqs = 0

        local_common = owner._build_graph_common_attn_metadata(
            common,
            actual_reqs,
        )
        local_scope = self._scope(local_common, actual_reqs)
        return {
            "context": pcp_context,
            "has_prefill": has_prefill,
            "needs_global": needs_global,
            "global": global_scope,
            "local": local_scope,
            "full_graph_mode": full_graph_mode,
        }

    def _rope(
        self,
        scope: BatchScope,
        side: str,
        step: dict[str, Any],
    ):
        if scope.rope is not None:
            return scope.rope

        common = scope.common
        full_graph = step["full_graph_mode"]
        num_tokens = common.num_input_tokens if full_graph else common.num_actual_tokens

        positions = common.positions[:num_tokens].long()
        cos, sin = get_cos_and_sin_dsa(
            positions,
            use_cache=(not self.is_draft and side == "local" and scope.num_prefills == 0),
        )
        # Draft execution must not overwrite the target's runtime RoPE cache.
        if self.is_draft and self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs():
            cos, sin = cos.copy_to_buffers(self.draft_rope_buffers, self.restore_capacity)
        if (
            side == "global"
            and not step["has_prefill"]
            and self.context_owner._is_decode_sharded
            and self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            cos, sin = cos.copy_to_buffers(
                self.global_rope_buffers,
                self.restore_capacity,
            )
        scope.rope = (cos, sin)
        return scope.rope

    def _start_pos(
        self,
        scope: BatchScope,
        side: str,
    ) -> torch.Tensor:
        if scope.start_pos is not None:
            return scope.start_pos

        common = scope.common
        num_reqs = common.num_reqs
        buffer = self.start_buffers[side]

        if num_reqs > buffer.numel():
            raise ValueError(f"{side} request count exceeds metadata capacity")

        offsets = common.query_start_loc[: num_reqs + 1]
        query_lens = offsets[1:] - offsets[:-1]
        torch.sub(
            common.seq_lens[:num_reqs],
            query_lens,
            out=buffer[:num_reqs],
        )
        buffer[scope.actual_reqs :].zero_()
        scope.start_pos = buffer[:num_reqs]
        return scope.start_pos

    def _limits(self, scope: BatchScope) -> tuple[int, int]:
        if scope.limits is not None:
            return scope.limits

        common = scope.common
        num_reqs = common.num_reqs

        if num_reqs == 0:
            scope.limits = (0, 0)
            return scope.limits

        offsets = common.query_start_loc_cpu[: num_reqs + 1]
        seq_lens_cpu = common._seq_lens_cpu
        if seq_lens_cpu is None:
            seq_lens_cpu = common.seq_lens_cpu
        if seq_lens_cpu is None:
            # Match the ordinary builder for async draft batches. This copy
            # occurs once per scope, not once for every cache group.
            seq_lens_cpu = common.seq_lens.cpu()

        max_q = int((offsets[1:] - offsets[:-1]).max().item())
        max_kv = int(seq_lens_cpu[:num_reqs].max().item())

        speculative = self.vllm_config.speculative_config
        if (
            speculative is not None
            and speculative.method == "dspark"
            and getattr(
                speculative,
                "enable_adaptive_verification",
                False,
            )
            and scope.num_prefills == 0
        ):
            max_q = max(
                max_q,
                speculative.num_speculative_tokens + 1,
            )

        scope.limits = (max_q, max_kv)
        return scope.limits

    def _sas(
        self,
        scope: BatchScope,
        ratio: int,
        *,
        ori_win_left: int,
        ori_win_right: int,
        dspark_swa: bool,
    ):
        assert scope.sas is not None
        assert scope.op_cache is not None

        cache_key = (ratio, dspark_swa, ori_win_left, ori_win_right)
        cached = scope.sas.get(cache_key)
        if cached is not None:
            return cached

        owner = self.sas_variant_owners[cache_key]
        common = scope.common
        num_reqs = common.num_reqs
        offsets = common.query_start_loc[: num_reqs + 1]
        seq_lens = common.seq_lens[:num_reqs]
        max_q, max_kv = self._limits(scope)

        if scope.num_prefills > 0:
            cu_ori = offsets
            cu_cmp = None
        else:
            cu_ori = DeviceOperator.get_dsa_decode_cu_seqlens_ori_kv(
                scope.op_cache,
                "cu_seqlens_ori_kv",
                seq_lens,
                num_reqs,
                owner._zero_i32,
                owner.cu_seqlens_ori_kv,
            )
            cu_cmp = DeviceOperator.get_dsa_decode_cu_seqlens_cmp_kv(owner.cu_seqlens_cmp_kv)

        sas = owner._build_sas_metadata(
            # The underlying builder keys SAS by compression ratio only.
            # Separate its cache as well as the persistent output buffer.
            metadata_cache=scope.op_cache.setdefault(("sas", cache_key), {}),
            layer_name=f"c{ratio}",
            query_start_loc=offsets,
            seq_lens=seq_lens,
            max_seqlen_q=max_q,
            max_seqlen_kv=max_kv,
            cu_seqlens_ori_kv=cu_ori,
            cu_seqlens_cmp_kv=cu_cmp,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            dspark_swa=dspark_swa,
        )

        result = (sas, cu_cmp)
        scope.sas[cache_key] = result
        return result

    def _qli(self, scope: BatchScope):
        if scope.qli is not None:
            return scope.qli

        owner = self.qli_owner
        assert owner is not None
        assert scope.op_cache is not None

        common = scope.common
        num_reqs = common.num_reqs
        if num_reqs > owner.qli_seqused_k.numel():
            raise ValueError("QLI request count exceeds metadata capacity")

        offsets = common.query_start_loc[: num_reqs + 1]
        seq_lens = common.seq_lens[:num_reqs]
        max_q, max_kv = self._limits(scope)

        qli = owner._build_qli_metadata(
            metadata_cache=scope.op_cache,
            query_start_loc=offsets,
            seq_lens=seq_lens,
            max_seqlen_q=max_q,
            max_seqlen_kv=max_kv,
        )

        scope.qli = (
            qli,
            offsets,
            owner.qli_seqused_k[:num_reqs],
            owner.qli_cmp_residual_k[:num_reqs],
        )
        return scope.qli

    @staticmethod
    def _envelope(
        builder,
        scope: BatchScope,
        common: AscendCommonAttentionMetadata,
        req_metadata,
    ):
        return dsa_v1.AscendDSAMetadata(
            num_actual_tokens=common.num_actual_tokens,
            num_decodes=scope.num_decodes,
            num_decode_tokens=scope.num_decode_tokens,
            num_prefills=scope.num_prefills,
            head_dim=builder.model_config.get_head_size(),
            attn_state=common.attn_state,
            req_metadata=req_metadata,
            hadamard=builder.hadamard,
        )

    def _build_one(
        self,
        builder,
        use: MetadataUse,
        scope: BatchScope,
        common: AscendCommonAttentionMetadata,
        side: str,
        step: dict[str, Any],
        *,
        build_query: bool,
        update_cache: bool,
    ):
        num_reqs = common.num_reqs
        block_table = common.block_table_tensor[:num_reqs]

        if scope.actual_reqs < num_reqs:
            block_table[scope.actual_reqs : num_reqs].zero_()

        req = dsa_v1.AscendDSAReqMetadata(
            block_table=block_table,
            seq_lens=common.seq_lens[:num_reqs],
            slot_mapping=None,
            storage_block_size=builder.storage_block_size,
            query_start_loc=common.query_start_loc[: num_reqs + 1],
            num_actual_reqs=scope.actual_reqs,
            cache_group_key=builder.cache_group_key,
        )

        builder._device_metadata_tasks = ()

        if use == MetadataUse.STATE:
            return self._envelope(builder, scope, common, req)

        ratio = builder.compressor_ratio

        # Match AscendDSAMetadataBuilder.build_req_metadata: short
        # non-causal draft queries see the trailing window and the whole
        # draft block. Physical indices belong to this cache group and
        # must never be shared merely because the SAS plan is shared.
        dspark_swa_indices = None
        ori_win_left = builder.model_config.hf_config.sliding_window - 1
        ori_win_right = 0
        if (
            build_query
            and scope.num_prefills == 0
            and not common.causal
            and (ratio <= 1 or not is_turboquant(builder.vllm_config))
        ):
            assert builder.speculative_config is not None
            dspark_swa_indices, _ = build_dspark_swa_indices(
                block_table[: scope.num_decodes],
                builder.speculative_config.num_speculative_tokens,
                builder.model_config.hf_config.sliding_window,
                builder.storage_block_size,
                common.query_start_loc[: scope.num_decodes + 1],
                common.seq_lens[: scope.num_decodes],
                scope.num_decode_tokens,
                buffer=builder.dspark_swa_indices_buffer,
            )
            ori_win_left, ori_win_right = get_dspark_sparse_sas_window(builder.vllm_config)
        req.dspark_swa_indices = dspark_swa_indices
        req.ori_win_left = ori_win_left
        req.ori_win_right = ori_win_right

        needs_query_rope = build_query and bool(use & (MetadataUse.SWA | MetadataUse.ATTENTION | MetadataUse.INDEXER))
        needs_swa_write = update_cache and bool(use & MetadataUse.SWA)

        if needs_query_rope or needs_swa_write:
            req.cos, req.sin = self._rope(scope, side, step)

        if needs_swa_write:
            num_input_tokens = common.num_input_tokens
            slots = common.slot_mapping[:num_input_tokens]
            formatted = get_dsa_attn_kv_plan(builder.vllm_config).format_dsa_slot_mapping(
                slots,
                builder.storage_block_size,
            )
            builder.slot_mapping[:num_input_tokens].copy_(formatted)
            req.slot_mapping = builder.slot_mapping[: common.num_actual_tokens]
            req.num_compressed_tokens = common.num_actual_tokens

        needs_compressor = update_cache and ratio > 1 and bool(use & (MetadataUse.ATTENTION | MetadataUse.INDEXER))
        if needs_compressor:
            req.start_pos = self._start_pos(scope, side)
            req.full_compress_cos, req.full_compress_sin = get_full_cos_and_sin_dsa(f"c{ratio}")

            if step["full_graph_mode"]:
                num_tokens = common.num_input_tokens
                req.num_actual_reqs = num_reqs
            else:
                num_tokens = common.num_actual_tokens

            req.num_compressed_tokens = min(
                num_tokens,
                num_tokens // ratio + num_reqs,
            )

        needs_sas = build_query and bool(use & (MetadataUse.ATTENTION | MetadataUse.SWA))
        if needs_sas:
            req.sas_metadata, req.cu_cmp_seqlen_list = self._sas(
                scope,
                ratio,
                ori_win_left=ori_win_left,
                ori_win_right=ori_win_right,
                dspark_swa=dspark_swa_indices is not None,
            )

        needs_qli = build_query and bool(use & MetadataUse.INDEXER)
        if needs_qli:
            req.qli_metadata, req.qli_cu_seqlens_q, req.qli_seqused_k, req.qli_cmp_residual_k = self._qli(scope)

        return self._envelope(builder, scope, common, req)

    def build_draft(self, builder, common, actual_reqs, shared, full_graph_mode):
        builder.common_ratio_to_sas_metadata = shared
        key = ("draft_metadata_plan", id(self))
        scope = shared.get(key)
        if scope is None:
            scope = self._scope(common, actual_reqs)
            shared[key] = scope
        # Only request geometry is shared. Causality, physical blocks and
        # slots must come from the current group, including mixed SWA groups.
        return self._build_one(
            builder,
            self.uses[id(builder)],
            scope,
            common,
            "local",
            {"full_graph_mode": full_graph_mode},
            build_query=True,
            update_cache=True,
        )

    def build(
        self,
        builder: AscendDSAPCPMetadataBuilder,
        common: AscendCommonAttentionMetadata,
        pcp_context,
        cache_group_idx: int,
        actual_reqs: int | None,
        shared: dict,
        full_graph_mode: bool,
    ):
        builder.common_ratio_to_sas_metadata = shared
        builder._device_metadata_tasks = ()
        builder._global_metadata_builder._device_metadata_tasks = ()

        step_key = ("pcp_metadata_plan", id(self))
        step = shared.get(step_key)
        if step is None:
            step = self._prepare_step(
                common,
                pcp_context,
                cache_group_idx,
                actual_reqs,
                full_graph_mode,
            )
            shared[step_key] = step

        context = step["context"]
        use = self.uses[id(builder)]
        global_metadata = None

        if step["needs_global"]:
            global_scope = step["global"]
            global_common = global_scope.common.replace(
                block_table_tensor=(context.global_block_tables[cache_group_idx]),
                slot_mapping=(context.global_slot_mappings[cache_group_idx]),
            )

            global_metadata = self._build_one(
                builder._global_metadata_builder,
                use,
                global_scope,
                global_common,
                "global",
                step,
                build_query=False,
                update_cache=True,
            )
        local_scope = step["local"]

        local_common = common.replace(
            query_start_loc=local_scope.common.query_start_loc,
            query_start_loc_cpu=local_scope.common.query_start_loc_cpu,
        )
        local_common = builder._build_local_common_attn_metadata(
            context,
            local_common,
            cache_group_idx,
            step["has_prefill"],
        )

        if int(local_common.query_start_loc_cpu[-1]) == 0:
            local_metadata = dsa_v1.AscendDSAMetadata(
                num_actual_tokens=0,
                num_decodes=0,
                num_decode_tokens=0,
                num_prefills=0,
                head_dim=builder.model_config.get_head_size(),
                attn_state=local_common.attn_state,
                req_metadata=None,
                hadamard=builder.hadamard,
            )
        else:
            local_metadata = self._build_one(
                builder,
                use,
                local_scope,
                local_common,
                "local",
                step,
                build_query=True,
                update_cache=not step["needs_global"],
            )
        if not step["needs_global"]:
            return local_metadata

        assert global_metadata is not None
        return AscendDSAPCPMetadata.from_local_metadata(
            local_metadata, local_common.num_input_tokens, context.hidden_restore_idx, global_metadata
        )


def bind_pcp_metadata_plan(runner) -> None:
    from vllm_ascend.models.deepseek_v4.compressor import AscendCompressorStateCache
    from vllm_ascend.models.deepseek_v4.indexer import AscendDeepseekV4IndexerCache
    from vllm_ascend.models.deepseek_v4.model import AscendDeepseekV4SWACache
    from vllm_ascend.models.layer.attention.layer import DSAAttention

    config = runner.vllm_config
    hf = config.model_config.hf_config

    if runner.pcp_manager is None:
        return
    if hf.model_type != "deepseek_v4":
        return
    if getattr(hf, "vision_n_layers", 0):
        return

    builders = []
    uses: dict[int, MetadataUse] = {}
    seen = set()
    context = config.compilation_config.static_forward_context

    speculator = getattr(runner, "speculator", None)
    draft_builder_ids = {
        id(group.get_metadata_builder(0))
        for groups in (getattr(speculator, "attn_groups", None) or ())
        for group in groups
    }

    for groups in runner.attn_groups:
        for group in groups:
            builder = group.get_metadata_builder(0)

            if type(builder) is not AscendDSAPCPMetadataBuilder:
                continue
            if builder.vllm_config is not config:
                continue

            if id(builder) in draft_builder_ids:
                continue

            use = MetadataUse(0)

            if group.backend is dsa_v1.AscendDSASWABackend:
                if builder.compressor_ratio > 1:
                    continue
                use = MetadataUse.SWA
            else:
                supported = True
                for name in group.layer_names:
                    layer = context.get(name)
                    if isinstance(layer, AscendCompressorStateCache):
                        use |= MetadataUse.STATE
                    elif isinstance(layer, AscendDeepseekV4IndexerCache):
                        use |= MetadataUse.INDEXER
                    elif isinstance(layer, AscendDeepseekV4SWACache):
                        use |= MetadataUse.SWA
                    elif isinstance(layer, DSAAttention):
                        use |= MetadataUse.ATTENTION
                    else:
                        supported = False
                        break

                if not supported or not use:
                    continue

            uses[id(builder)] = uses.get(id(builder), MetadataUse(0)) | use
            if id(builder) not in seen:
                seen.add(id(builder))
                builders.append(builder)

    if not builders:
        return
    thresholds = {builder.decode_threshold for builder in builders}
    if len(thresholds) != 1:
        return

    for builder in builders:
        use = uses[id(builder)]
        if use & MetadataUse.INDEXER and builder.compressor_ratio != 4:
            return
    plan = PCPMetadataPlan(builders, uses)
    for builder in builders:
        builder._pcp_metadata_plan = plan


def bind_dspark_metadata_plan(speculator, target_attn_groups) -> None:
    """Optimize ordinary DSA SWA draft groups; other backends retain build()."""
    config = speculator.attn_vllm_config
    speculative = config.speculative_config
    if speculative is None or speculative.method != "dspark":
        return
    if config.parallel_config.decode_context_parallel_size != 1:
        return
    draft_names = set(speculator.draft_attn_layer_names or ())
    target_ids = {
        id(group.get_metadata_builder(0))
        for groups in target_attn_groups
        for group in groups
        if set(group.layer_names) - draft_names
    }
    builders = {}
    for groups in speculator.attn_groups:
        for group in groups:
            builder = group.get_metadata_builder(0)
            if (
                type(builder) is dsa_v1.AscendDSAMetadataBuilder
                and group.backend is dsa_v1.AscendDSASWABackend
                and builder.compressor_ratio <= 1
                and set(group.layer_names).issubset(draft_names)
                and id(builder) not in target_ids
            ):
                builders[id(builder)] = builder
    if not builders:
        return
    if len({(id(b.vllm_config), b.decode_threshold) for b in builders.values()}) != 1:
        return
    plan = PCPMetadataPlan(list(builders.values()), dict.fromkeys(builders, MetadataUse.SWA), draft=True)
    for builder in builders.values():
        builder._pcp_metadata_plan = plan
