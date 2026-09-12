# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 replicated-cache TP-token DSA CP adapter."""

from dataclasses import replace

import torch
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context

from vllm_ascend.attention.context_parallel.dsa_common import restore_tp_heads
from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPMetadataBuilder
from vllm_ascend.attention.dsa_v1 import dsv4_dsa_overlap_stream
from vllm_ascend.attention.dsa_v41 import (
    DeepseekV41EagerAttentionImpl,
    DeepseekV41MetadataBuilder,
    _config_value,
    scatter_cache_sk,
)
from vllm_ascend.attention.utils import enable_pcp
from vllm_ascend.utils import enable_dsa_cp, npu_stream_switch


def get_v41_cp_classes():
    if enable_pcp():
        raise NotImplementedError("V4.1 PCP is not supported")
    if enable_dsa_cp():
        return DeepseekV41CPMetadataBuilder, DeepseekV41CPImpl
    return DeepseekV41MetadataBuilder, DeepseekV41EagerAttentionImpl


class _ReplicatedCacheMetadataBuilder(DeepseekV41MetadataBuilder):
    """Keep global cache metadata independent from local query buffers."""

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(
            kv_cache_spec, layer_names, vllm_config, device, build_compressor_metadata=False
        )
        self._global_builder = DeepseekV41MetadataBuilder(
            kv_cache_spec, layer_names, vllm_config, device, build_query_metadata=False
        )

    def enable_device_metadata(self):
        super().enable_device_metadata()
        self._global_builder.enable_device_metadata()

    def take_device_metadata_tasks(self):
        return (
            *self._global_builder.take_device_metadata_tasks(),
            *super().take_device_metadata_tasks(),
        )

    def _build_global_metadata(self, common_prefix_len, common, fast_build, kwargs):
        global_kwargs = dict(kwargs)
        shared = kwargs.get("common_v41_metadata")
        if shared is not None:
            global_kwargs["common_v41_metadata"] = shared.setdefault("cp_global", {})
        batch_shared = kwargs.get("common_v41_batch_metadata")
        if batch_shared is not None:
            global_kwargs["common_v41_batch_metadata"] = batch_shared.setdefault("cp_global", {})
        return self._global_builder.build(common_prefix_len, common, fast_build, **global_kwargs)


class DeepseekV41CPMetadataBuilder(_ReplicatedCacheMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # SMLA consumes INT32 offsets at a fixed address during graph replay.
        self._cp_query_start_loc = self._seq_lens.new_zeros(self._seq_lens.numel() + 1)

    # Reuse Legacy DSACP's request intersection and causal-prefix calculation.
    _local_token_range = staticmethod(AscendDSACPMetadataBuilder._local_token_range)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False, **kwargs):
        common = common_attn_metadata
        global_metadata = self._build_global_metadata(common_prefix_len, common, fast_build, kwargs)
        seq_lens_cpu = (
            common._seq_lens_cpu if getattr(common, "_seq_lens_cpu", None) is not None else common.seq_lens_cpu
        )
        start, end, per_rank, padded, qsl, seq_lens = AscendDSACPMetadataBuilder._build_local_token_metadata(
            self,
            common.num_reqs,
            common.num_input_tokens,
            common.query_start_loc_cpu,
            seq_lens_cpu,
            is_noncausal=not bool(getattr(common, "causal", True)),
        )
        actual_end = min(end, common.num_actual_tokens)
        actual_start = min(start, actual_end)
        # Padding participates in the output exchange, not in cache reads.
        qsl = qsl.clamp_max(actual_end - actual_start).to(self._cp_query_start_loc.dtype)
        query_start_loc = self._cp_query_start_loc[: qsl.numel()]
        query_start_loc.copy_(qsl)
        # Device lengths are authoritative after speculative rejection; the
        # CPU mirror may still be an upper bound. Remove only the query suffix
        # beyond this rank's token interval from each request's device length.
        query_ends = common.query_start_loc_cpu[1 : common.num_reqs + 1]
        suffix = query_ends - query_ends.clamp(min=actual_start, max=actual_end)
        if not bool(getattr(common, "causal", True)):
            suffix = torch.zeros_like(suffix)
        local_seq_lens = (common.seq_lens[: common.num_reqs] - suffix.to(common.seq_lens.device)).clamp_min(0)
        local_seq_lens = torch.where(query_start_loc[1:] > query_start_loc[:-1], local_seq_lens, 0)
        local_common = common.replace(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=qsl,
            seq_lens=local_seq_lens,
            seq_lens_cpu=seq_lens,
            num_actual_tokens=actual_end - actual_start,
            num_input_tokens=actual_end - actual_start,
            positions=common.positions[actual_start:actual_end],
            slot_mapping=common.slot_mapping[actual_start:actual_end],
            max_query_len=int((qsl[1:] - qsl[:-1]).max()) if common.num_reqs else 0,
            max_seq_len=int(seq_lens.max()) if common.num_reqs else 0,
        )
        kwargs["num_query_heads"] = _config_value(self.vllm_config.model_config.hf_text_config, "num_attention_heads")
        if global_metadata.cos is not None and global_metadata.sin is not None:
            # Q owns a contiguous token slice of the global KV batch. Reuse
            # that slice: a second cached RoPE gather would overwrite the
            # process-wide buffer still referenced by global KV metadata.
            kwargs["rope_views"] = (
                global_metadata.cos[actual_start:actual_end],
                global_metadata.sin[actual_start:actual_end],
            )
        if global_metadata.ori_sparse_indices is not None:
            kwargs["ori_sparse_indices"] = global_metadata.ori_sparse_indices[actual_start:actual_end]
        local = super().build(common_prefix_len, local_common, fast_build, **kwargs)
        return replace(local, global_metadata=global_metadata, cp_token_range=(start, end, per_rank, padded))


class DeepseekV41CPImpl(DeepseekV41EagerAttentionImpl):
    def multistream_preprocess(self, attn, hidden_states, cos, sin, swa_metadata):
        """Slice local Q from full inputs and overlap replicated KV preprocessing."""
        global_metadata = self._global_layer_metadata(get_forward_context().attn_metadata)
        kv_hidden_states = hidden_states[: global_metadata.swa.num_actual_tokens]
        start, _, _, _ = swa_metadata.cp_token_range
        hidden_states = hidden_states[start : start + swa_metadata.num_actual_tokens]
        kv_cos, kv_sin = global_metadata.rope(attn.rotary_emb.layername, kv_hidden_states.shape[0])
        swa_metadata = global_metadata.swa
        main_stream = torch.npu.current_stream()
        aux_stream = dsv4_dsa_overlap_stream()
        v1_impl = attn.dsa_attn.dsa_attn.impl
        wq_a, wkv, wq_b = v1_impl.cv_wq_a, v1_impl.cv_wkv, v1_impl.cv_wq_b

        # Q and KV own different token ranges, even with identical quantizers.
        q_quant, q_scale = wq_a.quantize(hidden_states)
        q_quant_done = main_stream.record_event()
        with npu_stream_switch(aux_stream, enabled=True):
            aux_stream.wait_event(q_quant_done)
            kv_quant, kv_scale = wkv.quantize(kv_hidden_states)
            kv_quant_done = aux_stream.record_event()
        q_a = wq_a.matmul(q_quant, q_scale, bias=attn.wq_a.bias)

        # Serialize Cube matmuls while overlapping Q Vector work with KV Cube.
        part2_start = main_stream.record_event()
        main_stream.wait_event(kv_quant_done)
        with npu_stream_switch(aux_stream, enabled=True):
            aux_stream.wait_event(part2_start)
            kv = wkv.matmul(kv_quant, kv_scale, bias=attn.wkv.bias)
            kv_matmul_done = aux_stream.record_event()
        qr = attn.q_norm(q_a)
        q_b_quant, q_b_scale = wq_b.quantize(qr)

        # KV Vector work uses global RoPE and global cache slots.
        part3_start = main_stream.record_event()
        main_stream.wait_event(kv_matmul_done)
        with npu_stream_switch(aux_stream, enabled=True):
            aux_stream.wait_event(part3_start)
            kv = attn.kv_norm(kv).view(-1, 1, attn.head_dim)
            torch.ops._C_ascend.inplace_partial_rotary_mul(
                kv.unsqueeze(1),
                kv_cos,
                kv_sin,
                rotary_mode="interleave",
                partial_slice=[attn.nope_head_dim, attn.head_dim],
            )
            scatter_cache_sk(attn.dsa_attn.swa_cache_layer.kv_cache[0], swa_metadata.slot_mapping, kv.squeeze(1))
        q = wq_b.matmul(q_b_quant, q_b_scale, bias=attn.wq_b.bias).unflatten(-1, (attn.n_heads, attn.head_dim))
        main_stream.wait_stream(aux_stream)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[attn.nope_head_dim, attn.head_dim],
        )
        # Both streams have joined before compressor/indexer cache reads.
        if self.role.is_kv_source:
            self._write_compressed_source(
                attn, kv_hidden_states, global_metadata.positions[: kv_hidden_states.shape[0]],
                kv_cos, kv_sin, global_metadata,
            )
        return q.to(hidden_states.dtype), qr

    def _global_layer_metadata(self, metadata_by_prefix):
        global_by_prefix = {}
        # The runner also includes DSpark's native DSA metadata in this map.
        # Resolve only the cache planes consumed by this target layer.
        for prefix in (
            self.swa_prefix,
            self.long_kv_source_prefix,
            self.index_k_source_prefix,
            self.compressor_state_prefix,
        ):
            if prefix is None:
                continue
            metadata = metadata_by_prefix[prefix]
            if metadata.global_metadata is None:
                raise ValueError(f"V4.1 CP is missing global cache metadata for {prefix}")
            global_by_prefix[prefix] = metadata.global_metadata
        return self._get_layer_metadata(global_by_prefix)

    def _prepare_inputs_and_caches(self, attn, hidden_states, metadata, metadata_by_prefix):
        if not attn.dsa_attn.dsa_attn.impl.multistream_dsv4_dsa_overlap or metadata.swa.num_actual_tokens == 0:
            # Empty query ranks still update replicated caches before exchange.
            global_metadata = self._global_layer_metadata(metadata_by_prefix)
            self._update_caches(attn, hidden_states[: global_metadata.swa.num_actual_tokens], global_metadata)

    def _prepare_queries(self, attn, hidden_states, positions, cos, sin, metadata):
        if attn.dsa_attn.dsa_attn.impl.multistream_dsv4_dsa_overlap:
            return self.multistream_preprocess(attn, hidden_states, cos, sin, metadata.swa)
        # Replicated caches were updated before the TP token slice.
        start, _, _, _ = metadata.swa.cp_token_range
        hidden_states = hidden_states[start : start + metadata.swa.num_actual_tokens]
        return self._project_q(attn, hidden_states, cos, sin)

    def _select_sparse_indices(self, attn, hidden_states, qr, positions, cos, sin, metadata):
        if not self.role.has_long_context:
            return None
        if not self.role.is_index_source:
            shared = attn.shared_state
            if shared is None:
                raise RuntimeError("V4.1 shared attention state is not initialized")
            # ``hidden_states`` still owns the full pre-CP token batch here,
            # while ``qr`` was projected from this rank's local query slice.
            # SparseFlashMla requires cmp_sparse_indices.T to match q.T.
            return shared.topk_indices[: qr.shape[0]]
        start, _, _, _ = metadata.swa.cp_token_range
        hidden_states = hidden_states[start : start + metadata.swa.num_actual_tokens]
        return super()._select_sparse_indices(attn, hidden_states, qr, positions, cos, sin, metadata)

    def _project_output(self, attn, output, hidden_states, metadata, *, projected):
        _, _, per_rank, _ = metadata.swa.cp_token_range
        padded = output
        if output.shape[0] != per_rank:
            padded = output.new_zeros((per_rank, output.shape[1], output.shape[2]))
            padded[: output.shape[0]] = output
        exchanged = restore_tp_heads(padded, get_tp_group())
        # The inherited V4 module owns quantized weights and TP projection logic.
        local_output = attn.dsa_attn.dsa_attn.impl._forward_o_proj(exchanged)
        projected.copy_(local_output[: hidden_states.shape[0]])
        return projected
