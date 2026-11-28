# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""A3 FlashMLA for Ling 3.1's component-major MLA cache."""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_ascend.ops.flash_mla_a3 import DecodePlan
    from vllm_ascend.ops.flash_mla_a3.balanced_decode import BalancedDecodePlan
    from vllm_ascend.ops.flash_mla_a3.graph_decode import GraphDecodePlan

BAILING_MLA_SUPPORTED_HEADS = frozenset((4, 8))
BAILING_MLA_LATENT_DIM = 512
BAILING_MLA_ROPE_DIM = 64
BAILING_MLA_PAGE_SIZE = 128
BAILING_MLA_MTP_WIDTH = 4
BAILING_MLA_BALANCED_MIN_BUCKET = 4
BAILING_MLA_BALANCED_MAX_BUCKET = 16

_BAILING_V3_ARCHITECTURES = frozenset(("BailingMoeV3ForCausalLM", "BailingMoeV3VLForConditionalGeneration"))


@lru_cache(maxsize=1)
def has_bailing_flashmla_ops() -> bool:
    """Check the operator libraries built by the A3 vLLM-Ascend installation."""
    build_dir = Path(__file__).resolve().parents[1] / "lib" / "flash_mla_a3"
    return all(
        (build_dir / library).is_file()
        for library in ("libflash_mla_a3_ops.so", "libflash_mla_a3_catlass_ops.so", "libflash_mla_a3_catlass.so")
    )


def supports_bailing_flashmla(config: "VllmConfig", *, enable_kv_nz: bool) -> bool:
    model = config.model_config
    if model is None or getattr(model, "architecture", None) not in _BAILING_V3_ARCHITECTURES:
        return False
    parallel = config.parallel_config
    text = model.hf_text_config
    return (
        not enable_kv_nz
        and parallel.prefill_context_parallel_size == parallel.decode_context_parallel_size == 1
        and model.dtype == torch.bfloat16
        and config.cache_config.cache_dtype in ("auto", "bfloat16")
        and model.get_num_attention_heads(parallel) in BAILING_MLA_SUPPORTED_HEADS
        and getattr(text, "kv_lora_rank", None) == BAILING_MLA_LATENT_DIM
        and getattr(text, "qk_rope_head_dim", None) == BAILING_MLA_ROPE_DIM
        and get_current_hardware_profile().supports(HardwareCapability.MLA_FLASH_A3_SPLIT_CACHE)
        and has_bailing_flashmla_ops()
    )


def supports_bailing_flashmla_full_graph(config: "VllmConfig") -> bool:
    from vllm.compilation import breakable_cudagraph
    from vllm.config import CUDAGraphMode

    # Separate FULL decode graphs provide one fixed-width segment per request.
    # Variable-width verification needs the existing FIA graph update path.
    if breakable_cudagraph.is_breakable_cudagraph_enabled() or config.compilation_config.cudagraph_mode not in (
        CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    ):
        return False
    speculative = config.speculative_config
    if speculative is not None and (
        speculative.num_speculative_tokens != BAILING_MLA_MTP_WIDTH - 1
        or getattr(speculative, "enable_adaptive_verification", False)
    ):
        return False
    from vllm_ascend.ops.flash_mla_a3.graph_decode import graph_decode_ops_available

    return graph_decode_ops_available()


def supports_bailing_flashmla_cache(cache: Sequence[torch.Tensor], device: torch.device) -> bool:
    """Check the component-major page layout used by the native A3 kernels."""
    import torch_npu

    return (
        device.type == "npu"
        and len(cache) == 2
        and all(
            component.device == device
            and component.dtype == torch.bfloat16
            and component.shape[1:] == (BAILING_MLA_PAGE_SIZE, 1, channels)
            and component.stride(1) == channels
            and component.stride(-1) == 1
            and BAILING_MLA_PAGE_SIZE * channels <= component.stride(0) <= 2**31 - 1
            and component.stride(0) % 16 == 0
            and component.storage_offset() % 16 == 0
            and torch_npu.get_npu_format(component) in (0, 2)
            for component, channels in zip(cache, (BAILING_MLA_LATENT_DIM, BAILING_MLA_ROPE_DIM))
        )
        and 0 < cache[0].size(0) == cache[1].size(0) <= 2**31 - 1
    )


@dataclass(frozen=True)
class BailingBalancedGraphPlan:
    """Use one schedule launch per graph step and MLA group."""

    decode: "BalancedDecodePlan"
    first_layer_name: str

    @property
    def block_table(self):
        return self.decode.block_table

    @property
    def query_length(self):
        return self.decode.schedule.query_length

    def update_for_layer(self, layer_name):
        if layer_name == self.first_layer_name:
            from vllm_ascend.ops.flash_mla_a3.balanced_decode import update_balanced_schedule

            update_balanced_schedule(self.decode.schedule)

    def attention(self, query, cache):
        from vllm_ascend.ops.flash_mla_a3.balanced_decode import flash_mla_decode_balanced

        return flash_mla_decode_balanced(query, cache, self.decode)


@dataclass
class BailingFlashMLAGraph:
    """Retain one graph plan per capture shape and one workspace per MLA group."""

    config: "VllmConfig"
    layer_names: tuple[str, ...]
    workspace: torch.Tensor | None = None
    plans: "dict[tuple[int, int, int, int, bool], GraphDecodePlan | BailingBalancedGraphPlan]" = field(
        default_factory=dict
    )
    balanced_abi_ready: bool | None = None

    def _prepare_balanced(self, table, lengths, width, causal, signature, max_kv, batch_sizes, guard_size):
        from vllm.model_executor.models.utils import extract_layer_index

        from vllm_ascend.ops.flash_mla_a3.balanced_decode import (
            balanced_decode_ops_available,
            flash_mla_decode_balanced,
            prepare_balanced_decode,
            prepare_balanced_schedule,
            update_balanced_schedule,
        )
        from vllm_ascend.ops.flash_mla_a3.prepared_decode import catlass_ops

        if self.balanced_abi_ready is False:
            return None
        if self.balanced_abi_ready is None and not balanced_decode_ops_available():
            self.balanced_abi_ready = False
            return None

        # Group declaration order can differ from model execution order.
        first = min(self.layer_names, key=extract_layer_index)
        page_count, page_strides, heads, scale = signature
        page_limit = (max_kv + BAILING_MLA_PAGE_SIZE - 1) // BAILING_MLA_PAGE_SIZE
        schedule = prepare_balanced_schedule(lengths, page_limit, query_length=width, causal=causal)
        capacity = max(count for count in batch_sizes if count <= BAILING_MLA_BALANCED_MAX_BUCKET)
        size = max(guard_size, catlass_ops().catlass_balanced_graph_workspace_bytes(table, capacity, heads * width))
        if self.workspace is None or self.workspace.numel() < size:
            self.workspace = torch.empty(size, dtype=torch.uint8, device=table.device)
        plan = prepare_balanced_decode(
            table,
            schedule,
            page_count,
            num_query_heads=heads,
            cache_page_strides=page_strides,
            softmax_scale=scale,
            workspace=self.workspace,
        )

        # Compile the scheduler specialization before capture. This warmup
        # leaves no Python flag that could suppress the node in capture.
        if self.balanced_abi_ready is None:
            cache = self.config.compilation_config.static_forward_context[first].kv_cache
            if isinstance(cache, list) and len(cache) == 1:
                cache = cache[0]
            blank = replace(schedule, lengths=torch.zeros_like(lengths))
            update_balanced_schedule(blank)
            query = tuple(
                torch.empty((table.size(0) * width, heads, channels), dtype=torch.bfloat16, device=table.device)
                for channels in (BAILING_MLA_LATENT_DIM, BAILING_MLA_ROPE_DIM)
            )
            try:
                flash_mla_decode_balanced(query, cache, replace(plan, schedule=blank))
            except (OSError, RuntimeError) as error:
                if not isinstance(error, OSError) and not any(
                    marker in str(error) for marker in ("Missing the A3 device-tiling launch entry", "Cannot load")
                ):
                    raise
                self.balanced_abi_ready = False
                logger.warning_once("Ling A3 FlashMLA balanced kernel unavailable. Using standard graph scheduling.")
                return None
            self.balanced_abi_ready = True
        else:
            update_balanced_schedule(schedule)

        return BailingBalancedGraphPlan(plan, first)

    def prepare(self, common, *, causal: bool):
        from vllm_ascend.ops.flash_mla_a3.graph_decode import _split_count, prepare_decode_graph
        from vllm_ascend.ops.flash_mla_a3.prepared_decode import catlass_ops

        table = common.block_table_tensor[: common.num_reqs]
        lengths = common.seq_lens[: common.num_reqs]
        width = common.max_query_len
        batches = table.size(0)
        tokens = batches * width
        if (
            width not in (1, BAILING_MLA_MTP_WIDTH)
            or not 0 < batches < 65536
            or table.device.type != "npu"
            or table.dtype != torch.int32
            or table.ndim != 2
            or table.stride(1) != 1
            or not 0 < table.size(1) <= table.stride(0) <= (2**31 - 1) // BAILING_MLA_PAGE_SIZE
            or lengths.shape != (batches,)
            or lengths.device != table.device
            or lengths.dtype != torch.int32
            or not lengths.is_contiguous()
            or common.num_actual_tokens != tokens
            or common.num_input_tokens != tokens
        ):
            return None
        # Capture uses the same physical partition as replay. The runner pads
        # with complete inactive requests and zeros their device KV lengths.
        ends = common.query_start_loc_cpu[: batches + 1]
        if len(ends) != batches + 1 or int(ends[0]) != 0 or not ((ends[1:] - ends[:-1]) == width).all():
            return None
        key = (table.data_ptr(), lengths.data_ptr(), batches, width, causal)
        if (plan := self.plans.get(key)) is not None:
            return plan

        contexts = self.config.compilation_config.static_forward_context
        signature = None
        for name in self.layer_names:
            layer = contexts[name]
            cache = layer.kv_cache
            if isinstance(cache, list) and len(cache) == 1:
                cache = cache[0]
            if not isinstance(cache, (list, tuple)) or not supports_bailing_flashmla_cache(cache, table.device):
                return None
            layer_signature = (
                cache[0].size(0),
                (cache[0].stride(0), cache[1].stride(0)),
                layer.impl.num_heads,
                layer.impl.scale,
            )
            if signature is not None and signature != layer_signature:
                return None
            signature = layer_signature
        if signature is None:
            return None

        page_count, page_strides, heads, scale = signature
        max_kv = min(self.config.model_config.max_model_len, table.size(1) * BAILING_MLA_PAGE_SIZE)
        ops = catlass_ops()
        cores = ops.catlass_cores(table)
        capture_sizes = self.config.compilation_config.cudagraph_capture_sizes or ()
        max_batches = max(batches, self.config.scheduler_config.max_num_seqs)
        batch_sizes = {
            size // width for size in capture_sizes if width <= size <= width * max_batches and size % width == 0
        }
        batch_sizes.add(batches)
        largest = max(batch_sizes, key=lambda count: count * _split_count(max_kv, count, cores))
        size = ops.catlass_graph_workspace_bytes(table, largest, heads * width, _split_count(max_kv, largest, cores))
        if (
            get_ascend_config().enable_balanced_flash_mla
            and BAILING_MLA_BALANCED_MIN_BUCKET <= batches <= BAILING_MLA_BALANCED_MAX_BUCKET
        ):
            balanced = self._prepare_balanced(table, lengths, width, causal, signature, max_kv, batch_sizes, size)
            if balanced is not None:
                self.plans[key] = balanced
                return balanced
        if self.workspace is None or self.workspace.numel() < size:
            # Model forwards and attention layers run in stream order, so
            # one allocation serves every captured capacity in this group.
            self.workspace = torch.empty(size, dtype=torch.uint8, device=table.device)
        plan = prepare_decode_graph(
            table,
            lengths,
            width,
            page_count,
            num_query_heads=heads,
            cache_page_strides=page_strides,
            softmax_scale=scale,
            causal=causal,
            max_kv=max_kv,
            workspace=self.workspace,
        )
        self.plans[key] = plan
        return plan


@dataclass
class BailingFlashMLABatch:
    """Share one decode plan across layers of the same MLA group and step."""

    query_width: int
    gather_queries: torch.Tensor | None = None
    gather_outputs: torch.Tensor | None = None
    device_lengths: torch.Tensor | None = None
    resolved_lengths: list[int] | None = None
    plans: "dict[tuple[int, int, int, int, float, bool], DecodePlan]" = field(default_factory=dict)

    @classmethod
    def from_query_ends(
        cls, query_ends: list[int], token_count: int, device: torch.device
    ) -> "BailingFlashMLABatch | None":
        if not query_ends or query_ends[-1] != token_count:
            return None
        lengths = [end - start for start, end in zip([0, *query_ends[:-1]], query_ends)]
        if min(lengths) < 1 or max(lengths) > BAILING_MLA_MTP_WIDTH:
            return None
        if all(length == 1 for length in lengths):
            return cls(1)
        if all(length == BAILING_MLA_MTP_WIDTH for length in lengths):
            return cls(BAILING_MLA_MTP_WIDTH)

        # Right-align each short request so the kernel's causal diagonal stays
        # at the same logical token. The artificial rows are discarded below.
        gather_queries = []
        gather_outputs = []
        for batch, (end, length) in enumerate(zip(query_ends, lengths)):
            start = end - length
            padding = BAILING_MLA_MTP_WIDTH - length
            gather_queries.extend([start] * padding)
            gather_queries.extend(range(start, end))
            offset = batch * BAILING_MLA_MTP_WIDTH
            gather_outputs.extend(range(offset + padding, offset + BAILING_MLA_MTP_WIDTH))
        pin_memory = device.type == "npu"
        query_indices = torch.tensor(gather_queries, dtype=torch.int32, pin_memory=pin_memory)
        output_indices = torch.tensor(gather_outputs, dtype=torch.int32, pin_memory=pin_memory)
        return cls(
            BAILING_MLA_MTP_WIDTH,
            query_indices.to(device, non_blocking=True),
            output_indices.to(device, non_blocking=True),
        )

    def attention(self, query, cache, block_table, lengths, *, scale, causal, attn_mask):
        from vllm_ascend.ops.flash_mla_a3 import flash_mla_decode, prepare_decode

        page_count = cache[0].size(0)
        page_strides = (cache[0].stride(0), cache[1].stride(0))
        num_query_heads = query[0].size(1)
        key = (page_count, *page_strides, num_query_heads, scale, causal)
        plan = self.plans.get(key)
        if plan is None:
            if self.device_lengths is not None:
                # Async MTP's CPU mirror includes optimistic draft tokens.
                # Read the exact lengths once per group, before building its plan.
                if self.resolved_lengths is None:
                    self.resolved_lengths = self.device_lengths.tolist()
                lengths = self.resolved_lengths
            plan = prepare_decode(
                block_table,
                lengths,
                self.query_width,
                page_count,
                causal=causal,
                softmax_scale=scale,
                attn_mask=attn_mask if causal else None,
                split_cache=True,
                validate_page_table=False,
                cache_page_strides=page_strides,
                num_query_heads=num_query_heads,
            )
            self.plans[key] = plan
        if self.gather_queries is not None:
            query = tuple(q.index_select(0, self.gather_queries) for q in query)
        output = flash_mla_decode(query, cache, plan, layout_out="NTD")
        if self.gather_outputs is not None:
            output = output.index_select(1, self.gather_outputs)
        return output
