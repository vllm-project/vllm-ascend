# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from dataclasses import fields, is_dataclass
from typing import Any, ClassVar, Generic, TypeVar

import torch
from vllm.logger import logger
from vllm.v1.attention.backend import AttentionImpl, AttentionLayer

from vllm_ascend.ascend_forward_context import _EXTRA_CTX


class AttentionMetadata:
    pass


T = TypeVar("T", bound=AttentionMetadata)


class DSAAttentionImpl(AttentionImpl[T], Generic[T]):
    _CUMULATIVE_LENGTH_FIELDS: ClassVar[set[str]] = {
        "actual_seq_lengths",
        "actual_seq_lengths_q",
        "cu_seq_lens",
        "cu_seqlens",
        "cu_seqlens_k",
        "cu_seqlens_kv",
        "cu_seqlens_q",
        "query_start_loc",
        "query_start_loc_cpu",
    }

    @staticmethod
    def _copy_tensor_inplace(dst: torch.Tensor | None, src: torch.Tensor | None, field_name: str | None = None) -> None:
        if not isinstance(dst, torch.Tensor) or not isinstance(src, torch.Tensor):
            return
        if dst.data_ptr() == src.data_ptr():
            return
        if dst.shape == src.shape:
            dst.copy_(src)
            return
        dst_flat = dst.reshape(-1)
        src_flat = src.reshape(-1)
        numel = min(dst_flat.numel(), src_flat.numel())
        if numel > 0:
            dst_flat[:numel].copy_(src_flat[:numel])
        if dst_flat.numel() > numel:
            if field_name in DSAAttentionImpl._CUMULATIVE_LENGTH_FIELDS and numel > 0:
                dst_flat[numel:].copy_(src_flat[numel - 1].expand_as(dst_flat[numel:]))
            else:
                dst_flat[numel:].zero_()

    @classmethod
    def _copy_metadata_inplace(
        cls,
        dst: Any,
        src: Any,
        visited: set[tuple[int, int]] | None = None,
    ) -> None:
        if dst is None or src is None or dst is src:
            return
        if isinstance(dst, torch.Tensor) or isinstance(src, torch.Tensor):
            cls._copy_tensor_inplace(dst, src)
            return

        if visited is None:
            visited = set()
        pair = (id(dst), id(src))
        if pair in visited:
            return
        visited.add(pair)

        if isinstance(dst, dict) and isinstance(src, dict):
            for key, dst_value in dst.items():
                if key not in src:
                    continue
                src_value = src[key]
                if isinstance(dst_value, torch.Tensor) or isinstance(src_value, torch.Tensor):
                    cls._copy_tensor_inplace(dst_value, src_value)
                elif isinstance(dst_value, (dict, list)) or is_dataclass(dst_value) or hasattr(dst_value, "__dict__"):
                    cls._copy_metadata_inplace(dst_value, src_value, visited)
                else:
                    dst[key] = src_value
            return

        if isinstance(dst, list) and isinstance(src, (list, tuple)):
            common_len = min(len(dst), len(src))
            for idx in range(common_len):
                dst_value = dst[idx]
                src_value = src[idx]
                if isinstance(dst_value, torch.Tensor) or isinstance(src_value, torch.Tensor):
                    cls._copy_tensor_inplace(dst_value, src_value)
                elif isinstance(dst_value, (dict, list)) or is_dataclass(dst_value) or hasattr(dst_value, "__dict__"):
                    cls._copy_metadata_inplace(dst_value, src_value, visited)
                else:
                    dst[idx] = src_value
            if len(src) > len(dst):
                dst.extend(src[common_len:])
            return

        if is_dataclass(dst):
            names = [field.name for field in fields(dst)]
        elif hasattr(dst, "__dict__"):
            names = list(vars(dst).keys())
        else:
            return

        for name in names:
            if not hasattr(src, name):
                continue
            dst_value = getattr(dst, name)
            src_value = getattr(src, name)
            if isinstance(dst_value, torch.Tensor) or isinstance(src_value, torch.Tensor):
                cls._copy_tensor_inplace(dst_value, src_value, name)
            elif isinstance(dst_value, (dict, list)) or is_dataclass(dst_value) or hasattr(dst_value, "__dict__"):
                cls._copy_metadata_inplace(dst_value, src_value, visited)

    @staticmethod
    def _as_metadata_sequence(metadata: Any) -> list[Any]:
        if isinstance(metadata, tuple):
            return list(metadata)
        if isinstance(metadata, list):
            return metadata
        return [metadata]

    @staticmethod
    def _pack_runtime_metadata(
        runtime_values: list[Any],
        captured_sizes: list[int],
    ) -> list[list[Any]]:
        if not runtime_values:
            return []
        if len(captured_sizes) == 1:
            captured_size = captured_sizes[0]
            if len(runtime_values) == 1 and captured_size > 1:
                return [runtime_values * captured_size]
            return [runtime_values[:captured_size]]
        if len(runtime_values) == len(captured_sizes):
            return [[metadata] * captured_size for metadata, captured_size in zip(runtime_values, captured_sizes)]
        if sum(captured_sizes) == len(runtime_values):
            groups = []
            offset = 0
            for size in captured_sizes:
                groups.append(runtime_values[offset : offset + size])
                offset += size
            return groups
        return [[metadata] for metadata in runtime_values[: len(captured_sizes)]]

    @classmethod
    def _group_draft_attn_metadata(
        cls,
        forward_context,
        draft_attn_metadatas,
        captured_entries,
    ) -> list[list[Any]]:
        captured_sizes = [len(entry) if isinstance(entry, tuple) else 1 for entry in captured_entries]
        if draft_attn_metadatas:
            runtime_values = []
            for per_step_metadata in draft_attn_metadatas:
                for key in sorted(per_step_metadata):
                    runtime_values.extend(cls._as_metadata_sequence(per_step_metadata[key]))
            return cls._pack_runtime_metadata(runtime_values, captured_sizes)

        attn_metadata = getattr(forward_context, "attn_metadata", None)
        if isinstance(attn_metadata, dict):
            runtime_values = []
            for key in sorted(attn_metadata):
                runtime_values.extend(cls._as_metadata_sequence(attn_metadata[key]))
            return cls._pack_runtime_metadata(runtime_values, captured_sizes)
        if isinstance(attn_metadata, list):
            return cls._pack_runtime_metadata(attn_metadata, captured_sizes)
        if attn_metadata is not None:
            return cls._pack_runtime_metadata([attn_metadata], captured_sizes)
        return []

    @staticmethod
    def _register_draft_graph_metadata(
        num_tokens: int,
        attn_metadata: list[Any],
    ) -> None:
        if not _EXTRA_CTX.capturing or not _EXTRA_CTX.is_draft_model:
            return
        from vllm_ascend.compilation.acl_graph import get_draft_graph_params

        graph_params = get_draft_graph_params()
        if graph_params is None:
            return
        stream = torch.npu.current_stream()
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        graph_params.events[num_tokens].append(event)
        graph_params.handles[num_tokens].append(None)
        graph_params.attn_params[num_tokens].append(tuple(attn_metadata))

    @classmethod
    def update_graph_params(
        cls,
        update_stream,
        forward_context,
        num_tokens,
        vllm_config=None,
        speculative_config=None,
        num_dcp_pcp_tokens=None,
        draft_attn_metadatas=None,
        draft_attn_layer_names=None,
    ):
        if not _EXTRA_CTX.is_draft_model:
            return
        from vllm_ascend.compilation.acl_graph import get_draft_graph_params

        graph_params = get_draft_graph_params()
        if graph_params is None:
            return
        captured_entries = graph_params.attn_params.get(num_tokens)
        events = graph_params.events.get(num_tokens)
        if not captured_entries or not events:
            return

        with torch.npu.stream(update_stream):
            runtime_updates = cls._group_draft_attn_metadata(forward_context, draft_attn_metadatas, captured_entries)
            if len(runtime_updates) < len(captured_entries):
                logger.warning_once(
                    "DSA draft graph metadata update count mismatch: captured=%s runtime=%s num_tokens=%s.",
                    len(captured_entries),
                    len(runtime_updates),
                    num_tokens,
                )

            for idx, (captured_entry, event) in enumerate(zip(captured_entries, events)):
                captured_metadatas = captured_entry if isinstance(captured_entry, tuple) else (captured_entry,)
                runtime_metadata = runtime_updates[idx] if idx < len(runtime_updates) else []
                for captured_metadata, new_metadata in zip(captured_metadatas, runtime_metadata):
                    cls._copy_metadata_inplace(captured_metadata, new_metadata)
                event.record(update_stream)

    @abstractmethod
    def __init__(
        self,
        dim: int,
        n_heads: int,
        scale: float,
        n_local_heads: int,
        q_lora_rank: int,
        o_lora_rank: int,
        head_dim: int,
        rope_head_dim: int | None,
        nope_head_dim: int,
        n_groups: int,
        n_local_groups: int,
        window_size: int,
        compress_ratio: int,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        layer: AttentionLayer,
        hidden_states_or_cq: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: T,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError
