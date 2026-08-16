# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch_npu
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    notify_kv_cache_written,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import record_attention_compute_start
from vllm_ascend.worker.v2.updatable_graph import (
    get_capture_resource,
    register_task,
)


_FIA_WORKSPACE_KEY = "npu_fused_infer_attention_score.workspace"


@dataclass
class AscendMetadata:
    attn_mask: torch.Tensor | None = None
    num_actual_tokens: int = 0
    seq_lens: list[int] = None
    query_start_loc: list[int] = None
    block_table: torch.Tensor = None


class AscendAttentionBackend(AttentionBackend):
    accept_output_buffer = True
    forward_includes_kv_cache_update = False

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["AscendAttentionBackendImpl"]:
        return AscendAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        return AscendAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128]


class AscendAttentionMetadataBuilder(AttentionMetadataBuilder[AscendMetadata]):
    reorder_batch_threshold = 1
    metadata_cls = AscendMetadata

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
        # FIA uses host-side sequence-length lists. Eager draft decode must
        # rebuild metadata for every step, while full graphs update each
        # captured FIA task through UpdatableGraph.
        self.supports_draft_decode_metadata_update = (
            cudagraph_mode is not None
            and cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
        )
        self.model_config = vllm_config.model_config
        self.max_num_blocks_per_req = cdiv(
            self.model_config.max_model_len,
            AscendAttentionBackend.get_supported_kernel_block_sizes()[0],
        )
        self.attn_mask_builder = AttentionMaskBuilder(device)

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.ALWAYS

    def reorder_batch(self, input_batch, scheduler_output: SchedulerOutput) -> bool:
        return False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendMetadata:
        return AscendMetadata(
            attn_mask=self.attn_mask_builder.get_attention_mask(
                common_attn_metadata.causal, self.model_config
            ),
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            seq_lens=common_attn_metadata.seq_lens_list,
            query_start_loc=common_attn_metadata.query_start_loc_list,
            block_table=common_attn_metadata.block_table_tensor,
        )

    def update_draft_decode_metadata(self, metadata: AscendMetadata) -> None:
        pass


@dataclass(frozen=True, slots=True)
class FIAParamProvider:
    layer_name: str

    def resolve(self, attn_metadata) -> dict[str, Any]:
        metadata = attn_metadata[self.layer_name]
        return {
            "actual_seq_lengths": metadata.query_start_loc,
            "actual_seq_lengths_kv": metadata.seq_lens,
        }


class AscendAttentionBackendImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        if alibi_slopes is not None:
            self.alibi_slopes = torch.tensor(
                alibi_slopes, dtype=torch.float32, device="npu"
            )
        else:
            self.alibi_slopes = None

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        DeviceOperator.reshape_and_cache(
            key=key,
            value=value,
            key_cache=kv_cache[0],
            value_cache=kv_cache[1],
            slot_mapping=slot_mapping,
        )
        notify_kv_cache_written()

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AscendMetadata | None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None
        assert output_scale is None and output_block_scale is None
        if attn_metadata is None:
            return output.fill_(0)

        record_attention_compute_start()
        query = query[: attn_metadata.num_actual_tokens]
        output_view = output[: attn_metadata.num_actual_tokens]
        key_cache, value_cache = kv_cache[0], kv_cache[1]
        num_blocks, block_size, _, _ = key_cache.shape
        key = key_cache.view(num_blocks, block_size, -1)
        value = value_cache.view(num_blocks, block_size, -1)

        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        workspace = get_capture_resource(
            _FIA_WORKSPACE_KEY,
            lambda: torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=attn_metadata.block_table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=attn_metadata.query_start_loc,
                actual_seq_lengths_kv=attn_metadata.seq_lens,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                sparse_mode=3,
            ),
        )
        register_task(
            torch_npu.npu_fused_infer_attention_score.out,
            {
                "query": query,
                "key": key,
                "value": value,
                "atten_mask": attn_metadata.attn_mask,
                "block_table": attn_metadata.block_table,
                "input_layout": "TND",
                "block_size": block_size,
                "actual_seq_lengths": attn_metadata.query_start_loc,
                "actual_seq_lengths_kv": attn_metadata.seq_lens,
                "num_key_value_heads": self.num_kv_heads,
                "num_heads": self.num_heads,
                "scale": self.scale,
                "sparse_mode": 3,
                "workspace": workspace,
                "out": [output_view, softmax_lse],
            },
            FIAParamProvider(layer.layer_name),
        )
        return output
