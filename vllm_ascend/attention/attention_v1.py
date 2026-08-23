#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from dataclasses import dataclass
from enum import Enum
import os

import torch
import torch_npu
import vllm.envs as envs_vllm
from vllm.config import CUDAGraphMode, VllmConfig, get_current_vllm_config
from vllm.logger import logger
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (  # type: ignore
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
)
from vllm.v1.attention.backends.registry import (  # type: ignore
    AttentionBackendEnum,
    register_backend,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec, CrossAttentionSpec

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.context_parallel.common_cp import AscendMetadataForDecode, AscendMetadataForPrefill
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    PagedAttentionGraphParam,
    cache_graph_workspace,
    enable_cp,
    needs_layer_aware_fia_graph_replay,
    notify_kv_cache_written,
    split_decodes_and_prefills,
    update_paged_attention_graph_param,
    using_paged_attention,
)
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params,
    get_draft_graph_prefill_params,
    get_graph_params,
    update_draft_graph_params_workspaces,
    update_graph_params_workspaces,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.memcache_comm_fence import record_attention_compute_start
from vllm_ascend.utils import weak_ref_tensors

# default max value of sliding window size
SWA_INT_MAX = 2147483647
_ATTN_KEYS_BUFFER = None
# FA3 captured NPU tensors keyed by num_tokens, updated before each replay
# with the current batch data (cache_seqlens, cu_seqlens_q).
_FA3_GRAPH_TENSORS: dict = {}
# num_tokens values for which we already warned about a decode-replay key
# mismatch (S2 diagnostic); avoids spamming the log every decode step.
_FA3_S2_LOGGED: set = set()


def _no_fa3_graph_capture() -> bool:
    """Debug escape hatch: force decode to capture/replay via the CANN V1 graph
    path instead of FA3, for A/B comparison while isolating FA3-graph decode
    accuracy issues.  Set VLLM_ASCEND_DEBUG_FA3_NO_GRAPH=1.
    """
    return os.environ.get("VLLM_ASCEND_DEBUG_FA3_NO_GRAPH") == "1"


def _in_full_capture_stream() -> bool:
    """True only when attention runs INSIDE a FULL-mode NPUGraph capture.

    In PIECEWISE mode the attention op executes between piece graphs, yet
    ``_EXTRA_CTX.capturing`` stays True after the first piece capture of the
    step (it is only reset at the next init_forward_context).  Attention
    there must take the eager path so the capture-step output is also
    computed from the real batch instead of stale graph buffers.
    """
    try:
        from vllm.forward_context import get_forward_context

        return get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL
    except Exception:
        return False


def _fa3_prefill_graph_enabled() -> bool:
    """Prefill FA3 graph capture is opt-in (VLLM_ASCEND_FA3_PREFILL_GRAPH=1).

    Decode FA3 graph capture/replay is validated end-to-end; prefill FA3
    graph capture still has an open correctness issue under FULL mode
    (GPQA 4.55 vs baseline 72.12 with dummy-strip fixes applied), so it
    defaults OFF: prefill graph capture stays on the CANN V1 path and
    prefill FA3 runs eagerly where eager execution applies.
    """
    return os.environ.get("VLLM_ASCEND_FA3_PREFILL_GRAPH") == "1"


def _fa3_decode_graph_enabled() -> bool:
    """Decode FA3 graph capture is opt-in (VLLM_ASCEND_FA3_DECODE_GRAPH=1).

    Validated on a single-node TP4 FULL-mode deployment (no data parallelism)
    during the bring-up spike.  Under TP4 x DP4 with async scheduling the
    replayed decode graph still produces corrupted output (FULL-mode probe:
    gibberish answers with decode FA3 graph on, correct answers with
    VLLM_ASCEND_DEBUG_FA3_NO_GRAPH=1); root cause is under investigation.
    Defaults OFF: decode graph capture stays on the CANN V1 path.  Decode FA3
    still runs eagerly (FULL_DECODE_ONLY / PIECEWISE / graph-miss steps).
    """
    return os.environ.get("VLLM_ASCEND_FA3_DECODE_GRAPH") == "1"


@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND")
class AscendAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        # HACK(Ronald1995): vllm `initialize_kv_cache` method in model runner v2 make
        # attention name assertion, we just set name to FLASH_ATTN to avoid assertion error.
        # rectify this when vllm disable the assertion.
        return "CUSTOM" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AscendAttentionBackendImpl"]:
        if enable_cp():
            from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionCPImpl

            return AscendAttentionCPImpl
        return AscendAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        if enable_cp():
            from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionCPMetadataBuilder

            return AscendAttentionCPMetadataBuilder
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
    def swap_blocks(
        src_kv_cache: list[torch.Tensor],
        dst_kv_cache: list[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache, src_value_cache = src_kv_cache[0], src_kv_cache[1]
        dst_key_cache, dst_value_cache = dst_kv_cache[0], dst_kv_cache[1]
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]

        dst_key_cache[dst_indices] = src_key_cache[src_indices].to(dst_key_cache.device)
        dst_value_cache[dst_indices] = src_value_cache[src_indices].to(dst_key_cache.device)

    @staticmethod
    def copy_blocks(
        kv_caches: list[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        src_indices = src_to_dists[:, 0]
        dst_indices = src_to_dists[:, 1]

        for kv_cache in kv_caches:
            key_caches = kv_cache[0]
            value_caches = kv_cache[1]
            key_caches[dst_indices] = key_caches[src_indices]
            value_caches[dst_indices] = value_caches[src_indices]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128]


class AscendAttentionState(Enum):
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3
    SpecDecoding = 4


@dataclass
class AscendMetadata:
    """
    Per-layer attention metadata for Ascend FlashAttention backend.

    Contains attention masks, token counts, sequence lengths and KV cache
    related properties for attention computation.
    """

    # **************************** Basic Properties ************************** #
    attn_mask: torch.Tensor | None = None
    # Current state of this attention run.
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    # Number of tokens excluding padding.
    num_actual_tokens_pcp_padded: int = 0
    num_actual_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0

    # The sequence length per sequence. Sequence length means the computed
    # tokens + new tokens (is None if it is a decoding).
    # (batch_size,)
    # TODO(Angazenn): The following parameters are quite redundant and
    # contains similar information (such as seq_lens seq_lens_list). We
    # should simplified these parameters once attention schema in vLLM-Ascend
    # is unified.
    seq_lens: torch.Tensor = None
    seq_lens_cpu: torch.Tensor = None
    seq_lens_list: list[int] = None  # type: ignore
    actual_seq_lengths_q: list[int] = None  # type: ignore

    query_start_loc: torch.Tensor = None
    # Maximum query length in the batch (None for decoding).
    max_query_len: int | None = None

    # ********************** KV Cache Related Properties ********************* #
    # Block addresses per sequence (Seq id -> list of physical block).
    # (batch_size, max_blocks_per_seq)
    block_tables: torch.Tensor = None

    # The indices of the token slots that input tokens will be stored into.
    # E.g., if `slot_mapping` is [35, 2, 17] and the block size is 16, the
    # three tokens are stored in the 3rd slot in block 2, 2nd slot in block 0,
    # and 1st slot in block 1, respectively.
    # (num_tokens,)
    slot_mapping: torch.Tensor = None
    # pcp
    prefill: AscendMetadataForPrefill | None = None
    # dcp
    decode_meta: AscendMetadataForDecode | None = None

    causal: bool = True
    # runner_type in model_config.
    model_runner_type: str = ""
    # prefill reshape_and_cache event
    reshape_cache_event: torch.npu.Event = None


class AscendAttentionMetadataBuilder(AttentionMetadataBuilder[AscendMetadata]):
    """
    Builder for constructing AscendMetadata from CommonAttentionMetadata.

    Handles attention mask generation and metadata preparation for
    Ascend FlashAttention backend.
    """

    # Does this backend/builder reorder the batch?
    # If not, set this to None. Otherwise set it to the query
    # length that will be pulled into the front of the batch.
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.compilation_config = vllm_config.compilation_config
        self.device = device
        self.max_num_blocks_per_req = cdiv(
            self.model_config.max_model_len, AscendAttentionBackend.get_supported_kernel_block_sizes()[0]
        )

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, (
                f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"
            )

        self.reorder_batch_threshold = self.decode_threshold

        scheduler_config = vllm_config.scheduler_config
        self.chunked_prefill_enabled = scheduler_config.enable_chunked_prefill
        self.attn_mask_builder = AttentionMaskBuilder(self.device)

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendAttentionMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Explicit override in case the underlying builder specialized this getter.
        # @override omitted only because of mypy limitation due to type variable.
        return AttentionCGSupport.ALWAYS

    def reorder_batch(self, input_batch, scheduler_output: "SchedulerOutput") -> bool:
        return False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
            common_attn_metadata, decode_threshold=self.decode_threshold
        )

        block_table = common_attn_metadata.block_table_tensor
        # Prefer _seq_lens_cpu (always available, updated during draft
        # iterations) over seq_lens_cpu (None in async spec decode mode).
        if common_attn_metadata._seq_lens_cpu is not None:
            seq_lens = common_attn_metadata._seq_lens_cpu[:num_reqs]
        elif common_attn_metadata.seq_lens_cpu is not None:
            seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]
        else:
            seq_lens = common_attn_metadata.seq_lens[:num_reqs].to("cpu")

        slot_mapping = common_attn_metadata.slot_mapping[:num_actual_tokens]
        # this slot_mapping override doesn't work since vllm will override it again. We should fix it vllm.
        # see: https://github.com/vllm-project/vllm/blob/ce88756b967c2c5006746a424c15dd59a284ed8c/vllm/model_executor/layers/attention/cross_attention.py#L117
        if isinstance(self.kv_cache_spec, CrossAttentionSpec):
            seq_lens = common_attn_metadata.seq_lens
            slot_mapping = common_attn_metadata.slot_mapping.to(torch.int32)
        elif self.speculative_config and self.speculative_config.parallel_drafting:
            seq_lens = common_attn_metadata.seq_lens

        attn_state = common_attn_metadata.attn_state

        # Get attn_mask from singleton AttentionMaskBuilder
        attn_mask = self.attn_mask_builder.get_attention_mask(common_attn_metadata.causal, self.model_config)

        # TODO: Yet another unnecessary H2D while we already have a query_start_loc on device
        query_start_loc = query_start_loc_cpu.pin_memory().to(self.device, non_blocking=True)

        actual_seq_lengths_q = query_start_loc_cpu[1:].tolist()
        seq_lens_list = seq_lens.tolist()
        # flashcomm1/SP (or cudagraph) padding makes the model runner insert a
        # dummy padding request into query_start_loc to satisfy the FIA TND-layout
        # constraint (sum of q lengths == hidden_states.shape[0]), bumping the
        # q-derived batchSize by one. The query_start_loc buffer is sized
        # `max_num_reqs + 2` to hold it, but the seq_lens and block_table buffers
        # are only `max_num_reqs`, so when the batch is full the padded request
        # overflows and `[:num_reqs_padded]` silently truncates them. FIA then
        # fails (error 561002) checking, in order, the `actualSeqLengthsKv` length
        # and then the block_table row count against batchSize. Pad them to match:
        # the dummy request points at block 0, and its output is harmless because:
        #   (1) read side: the attention output for padding tokens is trimmed by
        #       `hidden_states = hidden_states[:-pad_size, :]` downstream;
        #   (2) write side: reshape_and_cache slices key/value/slot_mapping to
        #       `[:num_actual_tokens]` (unpadded count), so the dummy request
        #       never writes to KV cache.
        # So any valid positive KV length / zero block row is fine. Pad both
        # seq_lens_list and the seq_lens tensor: full_graph_fia_v2 passes the
        # seq_lens tensor (not seq_lens_list) as actual_seq_kvlen during graph
        # capture, and _get_fia_params derives the PrefillCacheHit batch size from
        # seq_lens.shape[0], so the tensor has to carry the dummy request too.
        num_reqs_fia = len(actual_seq_lengths_q)
        if len(seq_lens_list) < num_reqs_fia:
            padding_len = num_reqs_fia - len(seq_lens_list)
            seq_lens_list = seq_lens_list + [1] * padding_len
            seq_lens = torch.cat([seq_lens, seq_lens.new_ones(padding_len)])
        if block_table is not None and block_table.shape[0] < num_reqs_fia:
            block_table = torch.cat(
                [
                    block_table,
                    block_table.new_zeros((num_reqs_fia - block_table.shape[0], block_table.shape[1])),
                ],
                dim=0,
            )

        attn_metadata = AscendMetadata(
            num_actual_tokens=num_actual_tokens,
            num_decode_tokens=num_decode_tokens,
            block_tables=block_table,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            seq_lens_list=seq_lens_list,
            max_query_len=common_attn_metadata.max_query_len,
            actual_seq_lengths_q=actual_seq_lengths_q,
            slot_mapping=slot_mapping,
            attn_mask=attn_mask,
            attn_state=attn_state,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            causal=common_attn_metadata.causal,
            model_runner_type=self.model_config.runner_type,
        )
        return attn_metadata

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
    ):
        if attn_state in (
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.ChunkedPrefill,
            AscendAttentionState.SpecDecoding,
        ):
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
        else:
            raise NotImplementedError(
                "Currently we only support building dummy metadata for DecodeOnly and ChunkedPrefill state"
            )

        attn_metadata.attn_state = attn_state
        return attn_metadata


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
        sinks: torch.Tensor = None,
        **kwargs,
    ) -> None:
        self.vllm_config = get_current_vllm_config()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32, device="npu")
        self.alibi_slopes = alibi_slopes
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.key_cache = None
        self.value_cache = None
        self.is_kv_producer = (
            self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.is_kv_producer
        )
        self.enable_c8_quant = self.vllm_config.quant_config is not None and getattr(
            self.vllm_config.quant_config, "enable_c8_quant", False
        )
        self._use_layer_aware_fia_graph_replay = needs_layer_aware_fia_graph_replay()
        self._use_max_workspace_for_fia_graph = self._use_layer_aware_fia_graph_replay
        self.sinks = sinks
        self.layerIndex = 0
        # Some mixed-attention models cannot rely on the iteration order of
        # attn_metadata during graph replay. Record the captured layer name only
        # for that path.
        self._layer_name: str | None = None
        # flash-attention-npu (FA3) replacement for CANN V1 FIA in eager mode.
        self._fa3_enabled = self._check_fa3_available()
        # FA3 scheduler metadata cache: populated during eager warmup, reused
        # during graph capture so no H2D copy is needed inside torch.npu.graph().
        self._fa3_scheduler_metadata: dict = {}

    @staticmethod
    def _check_fa3_available():
        """Return True if flash-attention-npu is installed and importable."""
        try:
            from vllm_ascend.attention.fa3_adapter import HAS_FLASH_ATTN_NPU
            return HAS_FLASH_ATTN_NPU
        except (ImportError, AttributeError):
            return False

    def _fa3_eligible(self, attn_metadata: "AscendMetadata") -> bool:
        """Whether this attention call can be served by FA3 instead of CANN FIA.

        FA3 covers GQA with template masks only (causal / sliding-window /
        full).  Calls with learnable sinks, ENCODER_DECODER cross-attention
        (KV lengths decoupled from query lengths) or unsupported states stay
        on the CANN path.  head_dim <= 256 is validated inside the adapter
        (raises ValueError -> the caller trips _fa3_enabled and falls back).

        Under FULL cudagraph mode eager prefill is excluded: each eager FA3
        prefill allocates per-batch kernel workspaces sized by max_seqlen_q
        (multi-hundred MiB), and the FULL graph pool leaves too little
        headroom — observed as a sampler OOM that crashed the engine
        mid-benchmark.  Prefill FA3 there stays on the validated CANN graph
        path; eager prefill FA3 runs in FULL_DECODE_ONLY / PIECEWISE / eager
        modes where the headroom is sufficient (GPQA 69.70 verified).
        """
        if not self._fa3_enabled:
            return False
        if self.sinks is not None:
            return False
        if self.attn_type == AttentionType.ENCODER_DECODER:
            return False
        if (
            attn_metadata.attn_state != AscendAttentionState.DecodeOnly
            and self.vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL
        ):
            return False
        return attn_metadata.attn_state in (
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.PrefillNoCache,
            AscendAttentionState.PrefillCacheHit,
            AscendAttentionState.ChunkedPrefill,
        )

    def _build_fa3_scheduler_metadata(
        self,
        attn_metadata: "AscendMetadata",
        block_size: int,
        query: torch.Tensor,
    ):
        """Build FA3 scheduler metadata for the CURRENT batch (eager path).

        Called fresh each iteration — the eager batch changes every forward.
        Returns ``None`` for non-cache attention states (PrefillNoCache).
        """
        from vllm_ascend.attention.fa3_adapter import get_scheduler_metadata

        is_cache = attn_metadata.attn_state != AscendAttentionState.PrefillNoCache
        if not is_cache:
            return None

        # Seed the graph-cache during eager warmup (decode AND cache-mode
        # prefill states) so graph capture never calls get_scheduler_metadata
        # inside torch.npu.graph() — FA3's get_scheduler_metadata performs an
        # aclrtSynchronizeStream which is illegal on the captured stream.
        #
        # IMPORTANT: skip during the memory-profile run.  get_scheduler_metadata
        # allocates buffers sized by max_model_len; if allocated during the
        # profile it shrinks the measured free memory and thus the KV cache,
        # corrupting prefill output.  Defer to the graph-capture warmup which
        # runs AFTER the KV cache has been sized.
        if (
            (
                (
                    attn_metadata.attn_state == AscendAttentionState.DecodeOnly
                    and _fa3_decode_graph_enabled()
                )
                or (
                    _fa3_prefill_graph_enabled()
                    and attn_metadata.attn_state
                    in (
                        AscendAttentionState.PrefillCacheHit,
                        AscendAttentionState.ChunkedPrefill,
                    )
                )
            )
            and not _EXTRA_CTX.in_profile_run
        ):
            num_tokens = attn_metadata.actual_seq_lengths_q[-1]
            self._get_fa3_graph_params(num_tokens, attn_metadata, block_size, query)

        cache_seqlens = attn_metadata.seq_lens
        if cache_seqlens.device != query.device:
            cache_seqlens = cache_seqlens.to(device=query.device)
        # Strip the padding dummy segment (KV len 0, query spanning all
        # padding tokens) so the metadata fingerprint matches the call side,
        # which applies the same stripping in fa3_forward.  Without this the
        # dummy's huge q span explodes max_seqlen_q (fingerprint mismatch +
        # multi-GiB kernel workspace -> OOM).
        from vllm_ascend.attention.fa3_adapter import strip_padding_dummy

        real_cu, real_kv, max_seqlen_q = strip_padding_dummy(
            attn_metadata.actual_seq_lengths_q, attn_metadata.seq_lens_list,
        )
        cu_seqlens_q = torch.tensor(real_cu, dtype=torch.int32, device=query.device)
        cache_seqlens = torch.tensor(real_kv, dtype=torch.int32, device=query.device)

        # get_scheduler_metadata bakes the block-table ROW STRIDE
        # (maxNumBlocksPerBatch) as ceil(max_seqlen_k / block_size).  The kernel
        # walks the paged block table as
        #   blockTable[BIdx * maxNumBlocksPerBatch + col]
        # so the stride MUST equal the block-table width (page_table.shape[1]).
        # Passing the batch's current max KV length makes the stride smaller than
        # the width -> the kernel reads across rows into a previous request's
        # unallocated (-1) slots -> invalid K/V address -> MTE fault (507011).
        if attn_metadata.block_tables is not None:
            max_blocks_per_seq = attn_metadata.block_tables.shape[1]
        else:
            max_blocks_per_seq = (
                max(attn_metadata.seq_lens_list) + block_size - 1
            ) // block_size

        return get_scheduler_metadata(
            batch_size=len(real_kv),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_blocks_per_seq * block_size,
            num_heads_q=self.num_heads,
            num_heads_kv=self.num_kv_heads,
            headdim=self.head_size,
            cache_seqlens=cache_seqlens,
            qkv_dtype=query.dtype,
            cu_seqlens_q=cu_seqlens_q,
            page_size=block_size,
            causal=attn_metadata.causal,
        )

    def _get_fa3_graph_params(
        self,
        num_tokens: int,
        attn_metadata: "AscendMetadata",
        block_size: int,
        query: torch.Tensor,
    ):
        """Return cached FA3 graph params for a capture bucket (decode + prefill).

        Pre-allocates fixed-size NPU buffers so that:
          1. ``scheduler_metadata`` is baked with the BUCKET-BOUND config:
             decode buckets use ``max_seqlen_q=1`` (every decode request has
             exactly one query token); prefill buckets use the bucket upper
             bound ``num_tokens`` (any packed batch composition within the
             bucket has max q_len <= num_tokens; the kernel derives the real
             per-seq boundaries from cu_seqlens_q, validated by
             test_fa3_prefill_graph_contracts.py).  ``max_seqlen_k`` is baked
             with the paged-cache capacity so the block-table row stride
             equals the table width (a smaller stride reads across rows into
             -1 slots -> MTE fault 507011).
          2. The buffers' data is refreshed before each replay by
             ``refresh_fa3_graph_params`` (no H2D inside torch.npu.graph()).
          3. The ``block_table`` buffer is zero-padded so padding requests
             point to block 0 (valid memory), never stale freed blocks.

        The metadata cache is keyed by ``(num_tokens, baked_max_seqlen_q)``
        because the same num_tokens bucket holds a decode graph (max_q=1)
        and a prefill graph (max_q=num_tokens) with different tilings; the
        batch-level data buffers stay keyed by num_tokens alone (shared
        across both, fully overwritten on refresh).

        Returns ``None`` when the bucket has not been seeded by an eager
        warmup step yet (caller falls back to the CANN V1 capture).
        """
        # E3 diagnostic: do not seed FA3 graph buffers / _FA3_GRAPH_TENSORS when
        # FA3 graph capture is switched back to CANN V1.  The eager FA3 path
        # (which builds its own fresh scheduler_metadata) is unaffected.
        if _no_fa3_graph_capture():
            return None
        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            # Dense varlen has no fixed-address paged KV and its metadata
            # cannot be pre-baked for capture; its graph capture stays on the
            # CANN V1 path (the eager path still runs FA3).
            return None

        is_decode = attn_metadata.attn_state == AscendAttentionState.DecodeOnly
        baked_max_seqlen_q = 1 if is_decode else num_tokens
        cache_key = (num_tokens, baked_max_seqlen_q)
        if cache_key in self._fa3_scheduler_metadata:
            return self._fa3_scheduler_metadata[cache_key]

        from vllm_ascend.attention.fa3_adapter import get_scheduler_metadata

        device = query.device
        max_batch_size = num_tokens
        max_seqlen_k = max(
            attn_metadata.seq_lens_list if attn_metadata.seq_lens_list else [num_tokens]
        )
        # Match the buffer's block-table width to the actual block table so
        # the pre-replay .copy_() shapes align.
        if attn_metadata.block_tables is not None:
            max_blocks_per_seq = attn_metadata.block_tables.shape[1]
        else:
            max_blocks_per_seq = (max_seqlen_k + block_size - 1) // block_size

        # Fixed-size NPU buffers whose addresses are captured by NPUGraph.
        #
        # These hold BATCH-LEVEL data (cache_seqlens / cu_seqlens_q /
        # block_table) that is identical for every layer.  They MUST be shared
        # across layers: `refresh_fa3_graph_params` refreshes only the single
        # tuple stored in the global `_FA3_GRAPH_TENSORS[num_tokens]`, so if
        # each layer allocated its own buffers, every layer except the last
        # one would keep reading its own stale (capture-time,
        # block_table=zeros) buffers and produce wrong output.
        global _FA3_GRAPH_TENSORS
        if num_tokens in _FA3_GRAPH_TENSORS:
            cache_seqlens_buf, cu_seqlens_q_buf, block_table_buf = _FA3_GRAPH_TENSORS[num_tokens]
        else:
            cache_seqlens_buf = torch.zeros(max_batch_size, dtype=torch.int32, device=device)
            cu_seqlens_q_buf = torch.zeros(max_batch_size + 1, dtype=torch.int32, device=device)
            block_table_buf = torch.zeros(
                max_batch_size, max_blocks_per_seq, dtype=torch.int32, device=device
            )

            if is_decode:
                # Decode: cu_seqlens_q is always [0, 1, ..., num_tokens] (each
                # request has exactly 1 query token).  Fixed at allocation time.
                cu_seqlens_q_buf.copy_(
                    torch.arange(max_batch_size + 1, dtype=torch.int32, device=device)
                )
            else:
                # Prefill: cu_seqlens_q is batch-dependent; initialize with the
                # STRIPPED seeding batch (padding dummy removed, matching the
                # refresh layout) — refresh fully overwrites before replay.
                from vllm_ascend.attention.fa3_adapter import strip_padding_dummy

                seed_cu, seed_kv, _ = strip_padding_dummy(
                    attn_metadata.actual_seq_lengths_q,
                    attn_metadata.seq_lens_list,
                )
                cu_seqlens_q_buf[: len(seed_cu)] = torch.as_tensor(
                    seed_cu, dtype=torch.int32, device=device,
                )
            # cache_seqlens: warmup batch's real KV lengths (dummy KV=0 entries
            # replaced by 1 so padding rows read block 0, valid memory).
            n = min(attn_metadata.seq_lens.numel(), max_batch_size)
            cache_seqlens_buf[:n].copy_(attn_metadata.seq_lens[:n].to(device=device))
            cache_seqlens_buf[cache_seqlens_buf == 0] = 1
            _FA3_GRAPH_TENSORS[num_tokens] = (
                cache_seqlens_buf, cu_seqlens_q_buf, block_table_buf,
            )

        # Scheduler metadata for the bucket max config — valid for any batch
        # padded to num_tokens requests with max_seqlen_q <= baked bound.
        meta = get_scheduler_metadata(
            batch_size=max_batch_size,
            max_seqlen_q=baked_max_seqlen_q,
            max_seqlen_k=max_blocks_per_seq * block_size,
            num_heads_q=self.num_heads,
            num_heads_kv=self.num_kv_heads,
            headdim=self.head_size,
            cache_seqlens=cache_seqlens_buf,
            qkv_dtype=query.dtype,
            cu_seqlens_q=cu_seqlens_q_buf,
            page_size=block_size,
            causal=attn_metadata.causal,
        )

        fa3_graph = (
            meta, cache_seqlens_buf, cu_seqlens_q_buf, block_table_buf,
            baked_max_seqlen_q,
        )
        self._fa3_scheduler_metadata[cache_key] = fa3_graph
        logger.info(
            "FA3 graph capture: cached tensors for num_tokens=%s max_seqlen_q=%s "
            "(max_seqlen_k=%s, max_batch_size=%s, block_table_cols=%s).",
            num_tokens, baked_max_seqlen_q, max_seqlen_k, max_batch_size,
            max_blocks_per_seq,
        )
        return fa3_graph

    def _graph_metadata_layer_name(self, layer: AttentionLayer | None = None) -> str | None:
        layer_name = layer.layer_name if layer is not None else self._layer_name
        # KV-sharing layers replay with the target layer's metadata instead of
        # their own module name, matching vLLM's shared KV-cache ownership.
        return self.kv_sharing_target_layer_name or layer_name

    @staticmethod
    def refresh_fa3_graph_params(update_stream, forward_context, num_tokens):
        """Refresh FA3 graph buffers BEFORE the aclgraph replay (decode + prefill).

        FA3 is a plain torch op captured inside the aclgraph: it reads its
        ``cache_seqlens``/``cu_seqlens_q``/``block_table`` buffers directly at
        replay time and, unlike the CANN V1 path (task group + event), has no
        replay-side sync.  The buffers therefore must be refreshed BEFORE the
        replay — otherwise the first decode step reads the capture-time
        (block_table=zeros) buffers and produces wrong output.

        The copies are issued on the *current* stream, NOT on ``update_stream``
        followed by ``wait_stream``.  ``current_stream().wait_stream(update_stream)``
        emits a CANN ``notify wait`` task that fails under the FULL aclgraph
        replay (the device logs ``sqe_type=7(notify wait)`` then an MTE fault in
        the FA3 split kernel reading a half-written ``block_table``).  Copying on
        the current stream keeps the copies strictly ordered before the replay
        with no cross-stream dependency; the buffers total a few KB, so nothing
        meaningful is lost by not overlapping them.

        Returns True if an FA3 refresh was performed.
        """
        global _FA3_GRAPH_TENSORS
        first_meta = next(iter(forward_context.attn_metadata.values()), None)
        is_fa3_replay = first_meta is not None and (
            first_meta.attn_state
            in (
                AscendAttentionState.DecodeOnly,
                AscendAttentionState.PrefillCacheHit,
                AscendAttentionState.ChunkedPrefill,
            )
        )
        fa3_tensors = _FA3_GRAPH_TENSORS.get(num_tokens) if is_fa3_replay else None
        if is_fa3_replay and fa3_tensors is None and _FA3_GRAPH_TENSORS:
            # S2 diagnostic: FA3 replay could not find the captured FA3
            # tensors for this num_tokens.  Capture keys by
            # actual_seq_lengths_q[-1] (== num_tokens_padded at capture); replay
            # keys by num_tokens_padded.  If these diverge, no refresh happens
            # and the graph replays with STALE cache_seqlens/block_table ->
            # decode precision bug.  Log the mismatch once per key.
            global _FA3_S2_LOGGED
            if num_tokens not in _FA3_S2_LOGGED:
                _FA3_S2_LOGGED.add(num_tokens)
                logger.warning(
                    "FA3 replay: no captured graph tensors for "
                    "num_tokens=%s (captured keys=%s). cache_seqlens/block_table "
                    "will NOT be refreshed -> stale graph replay.",
                    num_tokens, sorted(_FA3_GRAPH_TENSORS),
                )
        if fa3_tensors is None:
            return False
        cache_seqlens, cu_seqlens_q, block_table_buf = fa3_tensors
        from vllm_ascend.attention.fa3_adapter import strip_padding_dummy

        for meta in forward_context.attn_metadata.values():
            if meta.seq_lens is not None:
                # Strip the padding dummy (KV 0, query spanning all padding
                # tokens) exactly like the eager path (fa3_forward) and the
                # metadata builder: mixing a dummy cu segment with real-only
                # cache_seqlens makes the kernel read cache_seqlens out of
                # bounds and corrupts the replayed output.
                real_cu, real_kv, _ = strip_padding_dummy(
                    meta.actual_seq_lengths_q, meta.seq_lens_list,
                )
                n_batch = cache_seqlens.numel()
                n_actual = len(real_kv)
                n_pad = n_batch - n_actual

                # cache_seqlens: real lengths first, padding requests
                # get KV length 1 (dummy, reads block 0 via zero rows).
                cache_seqlens[:n_actual].copy_(
                    torch.tensor(real_kv, dtype=cache_seqlens.dtype,
                                 device=cache_seqlens.device)
                )
                if n_pad > 0:
                    cache_seqlens[n_actual:].fill_(1)

                # cu_seqlens_q: real cumulative first, then one query
                # token per padding request.
                n_cu = cu_seqlens_q.numel()
                cu_seqlens_q[: len(real_cu)].copy_(
                    torch.tensor(real_cu, dtype=cu_seqlens_q.dtype,
                                 device=cu_seqlens_q.device)
                )
                if len(real_cu) < n_cu:
                    last = real_cu[-1]
                    for i in range(len(real_cu), n_cu):
                        last += 1
                        cu_seqlens_q[i] = last

                # block_table: real rows first, padding rows zeroed so
                # padding requests point to block 0 (valid memory).
                if meta.block_tables is not None:
                    n_bt_r = min(meta.block_tables.shape[0], block_table_buf.shape[0])
                    n_bt_c = min(meta.block_tables.shape[1], block_table_buf.shape[1])
                    block_table_buf[:n_bt_r, :n_bt_c].copy_(
                        meta.block_tables[:n_bt_r, :n_bt_c]
                    )
                    if n_bt_r < block_table_buf.shape[0]:
                        block_table_buf[n_bt_r:].zero_()
                break
        return True

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config=None,
        num_dcp_pcp_tokens=None,
        draft_attn_metadatas=None,
    ):
        use_layer_aware_replay = needs_layer_aware_fia_graph_replay()

        # FA3 graphs are invisible to the CANN task-group mechanism, so they
        # have NO entries in graph_params.  graph_params[num_tokens] may hold
        # entries from a CANN V1 graph captured with the SAME num_tokens
        # (e.g. a prefill CANN graph while decode runs FA3); running the CANN
        # task-group update below would use the current batch's data to update
        # those unrelated handles, corrupting the other graph.  The FA3
        # buffers themselves are refreshed by refresh_fa3_graph_params BEFORE
        # the replay.
        #
        # Skip ONLY for decode replays: decode buckets are FA3-captured, and
        # a prefill CANN graph sharing the same num_tokens bucket still needs
        # its task-group update.  (Prefill FA3 capture is opt-in and shares
        # the bucket's data buffers, never the CANN handles.)
        global _FA3_GRAPH_TENSORS
        first_meta = next(iter(forward_context.attn_metadata.values()), None)
        is_decode_replay = first_meta is not None and (
            first_meta.attn_state == AscendAttentionState.DecodeOnly
        )
        if is_decode_replay and num_tokens in _FA3_GRAPH_TENSORS:
            return

        if using_paged_attention(num_tokens, vllm_config):
            # Paged Attention update logic
            if _EXTRA_CTX.is_draft_model:
                if _EXTRA_CTX.is_draft_model_prefill:
                    graph_params = get_draft_graph_prefill_params()
                else:
                    graph_params = get_draft_graph_params()
            else:
                graph_params = get_graph_params()
            with torch.npu.stream(update_stream):
                for key, param, handle, event in zip(
                    forward_context.attn_metadata,
                    graph_params.attn_params[num_tokens],
                    graph_params.handles[num_tokens],
                    graph_params.events[num_tokens],
                ):
                    (
                        query,
                        key_cache,
                        value_cache,
                        num_kv_heads,
                        num_heads,
                        scale,
                        block_table,
                        seq_lens,
                        output,
                    ) = param
                    seq_lens = forward_context.attn_metadata[key].seq_lens

                    workspace = torch_npu._npu_paged_attention_get_workspace(
                        query=query,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        num_kv_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale_value=scale,
                        block_table=block_table,
                        context_lens=seq_lens,
                        out=output,
                    )
                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch_npu._npu_paged_attention(
                        query=query,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        num_kv_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale_value=scale,
                        block_table=block_table,
                        context_lens=seq_lens,
                        out=output,
                        workspace=workspace,
                    )
                    torch.npu.graph_task_update_end(update_stream)
                    event.record(update_stream)
        elif _EXTRA_CTX.sinks:
            # FIA update logic
            if _EXTRA_CTX.is_draft_model:
                graph_params = get_draft_graph_params()
                attn_metadata = draft_attn_metadatas
                draft_attn_key_steps = [
                    (draft_step, key)
                    for draft_step, per_step_metadata in enumerate(attn_metadata)
                    for key in per_step_metadata
                ]
                attn_keys = [key for _, key in draft_attn_key_steps]
            else:
                graph_params = get_graph_params()
                attn_metadata = forward_context.attn_metadata
                attn_keys = list(attn_metadata.keys())
            # For Qwen3-next, since the kv_cache_config has already categorized
            # linear_attn and self_attn, the attn_metadata is first arranged with
            # self_attn followed by linear_attn. Therefore, using zip directly
            # filters out the update operations for linear_attn.
            # TODO: We use a new variable `attn_keys` to ensure the loop count is
            # correct after get by `zip` because of the new structure of the attn_metadata
            # when running with the merged full eagle-graph. Should check it with Qwen3-next.
            num_layers = len(attn_keys)
            if num_layers == 0:
                return
            captured_attn_params = graph_params.attn_params[num_tokens]
            handles = graph_params.handles[num_tokens]
            events = graph_params.events[num_tokens]
            graph_param_count = len(captured_attn_params)
            workspace = graph_params.workspaces.get(num_tokens)
            if _EXTRA_CTX.is_draft_model:
                if graph_param_count > len(draft_attn_key_steps):
                    repeat_count = cdiv(graph_param_count, len(draft_attn_key_steps))
                    draft_attn_key_steps = (draft_attn_key_steps * repeat_count)[:graph_param_count]
                else:
                    draft_attn_key_steps = draft_attn_key_steps[:graph_param_count]
                attn_keys = [key for _, key in draft_attn_key_steps]
            elif use_layer_aware_replay:
                # One graph size can contain captured FIA ops from all layers.
                # Repeat attn keys to match the captured op count, then use the
                # stored layer name in each op param to resolve the exact
                # metadata entry during replay.
                attn_keys = [attn_keys[index % num_layers] for index in range(graph_param_count)]
            attn_count = 0
            with torch.npu.stream(update_stream):
                for key, param, handle, event in zip(
                    attn_keys,
                    captured_attn_params,
                    handles,
                    events,
                ):
                    (
                        query,
                        key_cache,
                        value,
                        block_tables,
                        attn_mask,
                        block_size,
                        seq_lens,
                        num_kv_heads,
                        num_heads,
                        scale,
                        sliding_window,
                        sinks,
                        attn_output,
                        softmax_lse,
                        layer_name,
                    ) = param

                    if _EXTRA_CTX.is_draft_model:
                        draft_step, key = draft_attn_key_steps[attn_count]
                        seq_lens = attn_metadata[draft_step][key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[draft_step][key].actual_seq_lengths_q
                        attn_count = attn_count + 1
                    else:
                        metadata_key = layer_name if layer_name is not None and layer_name in attn_metadata else key
                        seq_lens = attn_metadata[metadata_key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[metadata_key].actual_seq_lengths_q

                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch_npu.npu_fused_infer_attention_score_v2.out(
                        query=query,
                        key=key_cache,
                        value=value,
                        block_table=block_tables,
                        atten_mask=attn_mask,
                        input_layout="TND",
                        block_size=block_size,
                        actual_seq_qlen=actual_seq_lengths_q,
                        actual_seq_kvlen=seq_lens,
                        num_key_value_heads=num_kv_heads,
                        num_query_heads=num_heads,
                        sparse_mode=4 if sliding_window is not None else 3,
                        pre_tokens=sliding_window if sliding_window is not None else SWA_INT_MAX,
                        next_tokens=0,
                        softmax_scale=scale,
                        learnable_sink=sinks,
                        workspace=workspace,
                        out=[attn_output, softmax_lse],
                    )
                    torch.npu.graph_task_update_end(update_stream)
                    event.record(update_stream)
        else:
            # FIA update logic
            if _EXTRA_CTX.is_draft_model:
                if _EXTRA_CTX.is_draft_model_prefill:
                    graph_params = get_draft_graph_prefill_params()
                else:
                    graph_params = get_draft_graph_params()
                attn_metadata = draft_attn_metadatas
                draft_attn_key_steps = [
                    (draft_step, key)
                    for draft_step, per_step_metadata in enumerate(attn_metadata)
                    for key in per_step_metadata
                ]
                attn_keys = [key for _, key in draft_attn_key_steps]
            else:
                graph_params = get_graph_params()
                attn_metadata = forward_context.attn_metadata
                attn_keys = list(attn_metadata.keys())
                if not use_layer_aware_replay:
                    # In some speculative methods (such as DFlash), the order of
                    # attn_keys in the Target model will be disrupted instead of
                    # increasing by layer index, so need regular expressions to
                    # reorder the attn_keys and store the results in
                    # _ATTN_KEYS_BUFFER.
                    attn_keys_length = len(graph_params.attn_params[num_tokens])
                    global _ATTN_KEYS_BUFFER
                    if attn_keys_length == 0:
                        return
                    if not _ATTN_KEYS_BUFFER or len(_ATTN_KEYS_BUFFER) != attn_keys_length:
                        import regex as re

                        def extract_layer_index(key: str) -> int:
                            match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", key)
                            return int(match.group(1)) if match else 0

                        def is_direct_target_attn_key(key: str) -> bool:
                            return (
                                re.search(
                                    r"(?:^|\.)layers\.(\d+)\.self_attn\.attn$",
                                    key,
                                )
                                is not None
                            )

                        attn_keys_to_order = attn_keys[:attn_keys_length]
                        if getattr(speculative_config, "method", None) == "mtp":
                            # Step3.5 MTP can expose draft KV-cache groups in the
                            # target runtime metadata.  The target FULL graph only
                            # captures direct base-model self-attention handles, so
                            # select that target key domain instead of depending on
                            # the current draft module name.
                            direct_target_attn_keys = [key for key in attn_keys if is_direct_target_attn_key(key)]
                            if len(direct_target_attn_keys) >= attn_keys_length:
                                attn_keys_to_order = direct_target_attn_keys

                        attn_keys_tmp = attn_keys_to_order
                        attn_keys_tmp.sort(key=extract_layer_index)
                        _ATTN_KEYS_BUFFER = attn_keys_tmp[:attn_keys_length]
                    attn_keys[:attn_keys_length] = _ATTN_KEYS_BUFFER
            # For Qwen3-next, since the kv_cache_config has already categorized
            # linear_attn and self_attn, the attn_metadata is first arranged with
            # self_attn followed by linear_attn. Therefore, using zip directly
            # filters out the update operations for linear_attn.
            # TODO: We use a new variable `attn_keys` to ensure the loop count is
            # correct after get by `zip` because of the new structure of the attn_metadata
            # when running with the merged full eagle-graph. Should check it with Qwen3-next.
            num_layers = len(attn_keys)
            if num_layers == 0:
                return
            captured_attn_params = graph_params.attn_params[num_tokens]
            handles = graph_params.handles[num_tokens]
            events = graph_params.events[num_tokens]
            graph_param_count = len(captured_attn_params)
            workspace = graph_params.workspaces.get(num_tokens)
            if _EXTRA_CTX.is_draft_model:
                if graph_param_count > len(draft_attn_key_steps):
                    repeat_count = cdiv(graph_param_count, len(draft_attn_key_steps))
                    draft_attn_key_steps = (draft_attn_key_steps * repeat_count)[:graph_param_count]
                else:
                    draft_attn_key_steps = draft_attn_key_steps[:graph_param_count]
                attn_keys = [key for _, key in draft_attn_key_steps]
            elif use_layer_aware_replay:
                # Keep the replay loop length aligned with captured FIA ops;
                # layer-specific metadata lookup below prevents global/sliding
                # window layers from accidentally sharing the same metadata.
                attn_keys = [attn_keys[index % num_layers] for index in range(graph_param_count)]
            attn_count = 0
            layer_count = 0
            with torch.npu.stream(update_stream):
                for key, param, handle, event in zip(
                    attn_keys,
                    captured_attn_params,
                    handles,
                    events,
                ):
                    if isinstance(param, PagedAttentionGraphParam):
                        if _EXTRA_CTX.is_draft_model:
                            draft_step, key = draft_attn_key_steps[attn_count]
                            block_table = attn_metadata[draft_step][key].block_tables
                            seq_lens = attn_metadata[draft_step][key].seq_lens
                            attn_count = attn_count + 1
                        else:
                            layer_name = param.layer_name
                            metadata_key = layer_name if layer_name is not None and layer_name in attn_metadata else key
                            block_table = attn_metadata[metadata_key].block_tables
                            seq_lens = attn_metadata[metadata_key].seq_lens
                        update_paged_attention_graph_param(
                            update_stream,
                            handle,
                            event,
                            param,
                            block_table,
                            seq_lens,
                        )
                        continue
                    (
                        query,
                        key_cache,
                        value,
                        block_tables,
                        attn_mask,
                        block_size,
                        seq_lens,
                        query_start_loc,
                        num_kv_heads,
                        num_heads,
                        scale,
                        attn_output,
                        softmax_lse,
                        sparse_mode,
                        pre_tokens,
                        next_tokens,
                        c8_k_aq_scale,
                        c8_k_aq_offset,
                        c8_v_aq_scale,
                        c8_v_aq_offset,
                        layer_name,
                    ) = param

                    if _EXTRA_CTX.is_draft_model:
                        draft_step, key = draft_attn_key_steps[attn_count]
                        metadata = attn_metadata[draft_step][key]
                        seq_lens = metadata.seq_lens_list
                        actual_seq_lengths_q = metadata.actual_seq_lengths_q
                        block_tables = metadata.block_tables
                        attn_count = attn_count + 1
                        if not metadata.causal:
                            sparse_mode = 0
                    else:
                        metadata_key = layer_name if layer_name is not None and layer_name in attn_metadata else key
                        seq_lens = attn_metadata[metadata_key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[metadata_key].actual_seq_lengths_q
                        # NOTE:
                        # For models with sliding-window attention on the FIA full-graph replay path,
                        # rebinding `block_tables` to the latest metadata tensor causes corrupted /
                        # repeated outputs in our repro on Ascend NPU.
                        #
                        # Keep the captured block_tables tensor on this affected path.
                        # Non-SWA models preserve the original behavior and continue to refresh
                        # block_tables from attn_metadata.
                        if not hasattr(vllm_config.model_config.hf_text_config, "sliding_window"):
                            block_tables = attn_metadata[metadata_key].block_tables
                    layer_count += 1

                    torch.npu.graph_task_update_begin(update_stream, handle)
                    input_layout = "TND"
                    extra_args = {}
                    if c8_k_aq_scale is not None:
                        extra_args = {
                            "key_antiquant_scale": c8_k_aq_scale,
                            "value_antiquant_scale": c8_v_aq_scale,
                            "key_antiquant_mode": 0,
                            "value_antiquant_mode": 0,
                            "inner_precise": 1,
                        }
                        input_layout = "BNSD"
                        sparse_mode = 0
                    torch_npu.npu_fused_infer_attention_score.out(
                        query=query,
                        key=key_cache,
                        value=value,
                        block_table=block_tables,
                        atten_mask=attn_mask,
                        input_layout=input_layout,
                        block_size=block_size,
                        actual_seq_lengths=actual_seq_lengths_q,
                        actual_seq_lengths_kv=seq_lens,
                        num_key_value_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale=scale,
                        sparse_mode=sparse_mode,
                        pre_tokens=pre_tokens,
                        next_tokens=next_tokens,
                        **extra_args,
                        workspace=workspace,
                        out=[attn_output, softmax_lse],
                    )
                    torch.npu.graph_task_update_end(update_stream)

                    event.record(update_stream)

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        super().process_weights_after_loading(act_dtype)

    def full_graph_fia(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer=None,
    ) -> torch.Tensor:
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)

        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        if _EXTRA_CTX.is_draft_model:
            if _EXTRA_CTX.is_draft_model_prefill:
                graph_params = get_draft_graph_prefill_params()
            else:
                graph_params = get_draft_graph_params()
        else:
            graph_params = get_graph_params()
        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q
        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        input_layout = "TND"
        attn_mask = attn_metadata.attn_mask
        sparse_mode = 4 if self.sliding_window else 3 if attn_metadata.causal else 0
        pre_tokens = self.sliding_window or SWA_INT_MAX
        next_tokens = 0 if self.sliding_window else SWA_INT_MAX

        extra_args = {}
        if self.enable_c8_quant and layer is not None:
            extra_args = {
                "key_antiquant_scale": layer._c8_k_aq_scale_nz_bnsd,
                "value_antiquant_scale": layer._c8_v_aq_scale_nz_bnsd,
                "key_antiquant_mode": 0,
                "value_antiquant_mode": 0,
                "inner_precise": 1,
            }

            # change key/value shape
            _, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self._nz_5d_view(self.key_cache, block_size)
            value = self._nz_5d_view(self.value_cache, block_size)

            # TODO: change layerout from BNSD to TND.
            input_layout = "BNSD"
            query = query.unsqueeze(2)
            output = output.unsqueeze(2)
            attn_mask = None
            sparse_mode = 0
        use_max_workspace = self._use_max_workspace_for_fia_graph
        workspace = graph_params.workspaces.get(num_tokens)
        should_update_workspace_cache = False
        if use_max_workspace:
            # Some models mix attention layer shapes under the same graph size.
            # During capture, keep the largest required workspace for that size.
            candidate_workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_mask,
                block_table=block_table,
                input_layout=input_layout,
                block_size=block_size,
                actual_seq_lengths=actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                sparse_mode=sparse_mode,
                pre_tokens=pre_tokens,
                next_tokens=next_tokens,
                scale=self.scale,
                **extra_args,
            )
            workspace = cache_graph_workspace(
                graph_params,
                num_tokens,
                candidate_workspace,
                use_max_workspace=use_max_workspace,
            )
            should_update_workspace_cache = True
        elif workspace is None:
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_mask,
                block_table=block_table,
                input_layout=input_layout,
                block_size=block_size,
                actual_seq_lengths=actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                sparse_mode=sparse_mode,
                pre_tokens=pre_tokens,
                next_tokens=next_tokens,
                scale=self.scale,
                **extra_args,
            )
            should_update_workspace_cache = True
        if should_update_workspace_cache:
            if _EXTRA_CTX.is_draft_model:
                update_draft_graph_params_workspaces(num_tokens, workspace)
            else:
                update_graph_params_workspaces(num_tokens, workspace)

        # Handle graph capturing mode
        stream = torch_npu.npu.current_stream()

        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        graph_params.events[num_tokens].append(event)
        attn_params = (
            weak_ref_tensors(query),
            weak_ref_tensors(key),
            weak_ref_tensors(value),
            weak_ref_tensors(block_table),
            weak_ref_tensors(attn_mask) if attn_mask is not None else None,
            block_size,
            actual_seq_lengths_kv,
            actual_seq_lengths_q,
            self.num_kv_heads,
            self.num_heads,
            self.scale,
            weak_ref_tensors(output),
            weak_ref_tensors(softmax_lse),
            sparse_mode,
            pre_tokens,
            next_tokens,
        )
        if self.enable_c8_quant and layer is not None:
            attn_params = attn_params + (
                weak_ref_tensors(layer._c8_k_aq_scale_nz_bnsd),
                None,
                weak_ref_tensors(layer._c8_v_aq_scale_nz_bnsd),
                None,
            )  # type: ignore
        else:
            attn_params = attn_params + (None, None, None, None)  # type: ignore
        layer_name = self._graph_metadata_layer_name(layer) if self._use_layer_aware_fia_graph_replay else None
        attn_params = attn_params + (layer_name,)  # type: ignore
        graph_params.attn_params[num_tokens].append(attn_params)

        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score.out(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_mask,
            block_table=block_table,
            input_layout=input_layout,
            block_size=block_size,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=sparse_mode,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            workspace=workspace,
            out=[output, softmax_lse],
            **extra_args,
        )

        output = output.view(num_tokens, self.num_heads, self.head_size)

        handle = torch.npu.graph_task_group_end(stream)
        graph_params.handles[num_tokens].append(handle)
        return output, num_tokens

    def full_graph_fia_v2(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)
        actual_seq_lengths_kv = attn_metadata.seq_lens
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        if _EXTRA_CTX.is_draft_model:
            graph_params = get_draft_graph_params()
        else:
            graph_params = get_graph_params()

        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q
        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        use_max_workspace = self._use_max_workspace_for_fia_graph
        workspace = graph_params.workspaces.get(num_tokens)
        should_update_workspace_cache = False
        if use_max_workspace:
            # See full_graph_fia: this path needs the max workspace across layer
            # variants sharing the same graph size.
            candidate_workspace = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=block_table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_qlen=actual_seq_lengths_q,
                actual_seq_kvlen=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                softmax_scale=self.scale,
                num_query_heads=self.num_heads,
                sparse_mode=4 if self.sliding_window is not None else 3,
                pre_tokens=self.sliding_window if self.sliding_window is not None else SWA_INT_MAX,
                next_tokens=0,
                learnable_sink=self.sinks,
            )
            workspace = cache_graph_workspace(
                graph_params,
                num_tokens,
                candidate_workspace,
                use_max_workspace=use_max_workspace,
            )
            should_update_workspace_cache = True
        elif workspace is None:
            workspace = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=block_table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_qlen=actual_seq_lengths_q,
                actual_seq_kvlen=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                softmax_scale=self.scale,
                num_query_heads=self.num_heads,
                sparse_mode=4 if self.sliding_window is not None else 3,
                pre_tokens=self.sliding_window if self.sliding_window is not None else SWA_INT_MAX,
                next_tokens=0,
                learnable_sink=self.sinks,
            )
            should_update_workspace_cache = True
        if should_update_workspace_cache:
            if _EXTRA_CTX.is_draft_model:
                update_draft_graph_params_workspaces(num_tokens, workspace)
            else:
                update_graph_params_workspaces(num_tokens, workspace)

        # Handle graph capturing mode
        stream = torch_npu.npu.current_stream()

        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        graph_params.events[num_tokens].append(event)
        graph_params.attn_params[num_tokens].append(
            (
                weak_ref_tensors(query),
                weak_ref_tensors(key),
                weak_ref_tensors(value),
                weak_ref_tensors(block_table),
                weak_ref_tensors(attn_metadata.attn_mask),
                block_size,
                actual_seq_lengths_kv,
                self.num_kv_heads,
                self.num_heads,
                self.scale,
                self.sliding_window,
                self.sinks,
                weak_ref_tensors(output),
                weak_ref_tensors(softmax_lse),
                self._graph_metadata_layer_name() if self._use_layer_aware_fia_graph_replay else None,
            )
        )
        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score_v2.out(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_qlen=actual_seq_lengths_q,
            actual_seq_kvlen=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_query_heads=self.num_heads,
            sparse_mode=4 if self.sliding_window is not None else 3,
            pre_tokens=self.sliding_window if self.sliding_window is not None else SWA_INT_MAX,
            next_tokens=0,
            softmax_scale=self.scale,
            learnable_sink=self.sinks,
            workspace=workspace,
            out=[output, softmax_lse],
        )
        handle = torch.npu.graph_task_group_end(stream)
        graph_params.handles[num_tokens].append(handle)
        return output, num_tokens

    def full_graph_pa(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ):
        graph_params = get_graph_params()
        num_tokens = query.shape[0]
        if _EXTRA_CTX.capturing:
            # Get workspace from cache or calculate it if not present.
            workspace = graph_params.workspaces.get(num_tokens)
            if workspace is None:
                workspace = torch_npu._npu_paged_attention_get_workspace(
                    query=query,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    num_kv_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale_value=self.scale,
                    block_table=attn_metadata.block_tables,
                    context_lens=attn_metadata.seq_lens,
                    out=output,
                )
                update_graph_params_workspaces(num_tokens, workspace)

            # Handle graph capturing mode
            stream = torch_npu.npu.current_stream()

            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            graph_params.events[num_tokens].append(event)
            graph_params.attn_params[num_tokens].append(
                PagedAttentionGraphParam(
                    (
                        weak_ref_tensors(query),
                        weak_ref_tensors(self.key_cache),
                        weak_ref_tensors(self.value_cache),
                        self.num_kv_heads,
                        self.num_heads,
                        self.scale,
                        attn_metadata.block_tables,
                        attn_metadata.seq_lens,
                        weak_ref_tensors(output),
                    ),
                    self._graph_metadata_layer_name() if self._use_layer_aware_fia_graph_replay else None,
                )
            )

            torch.npu.graph_task_group_begin(stream)
            torch_npu._npu_paged_attention(
                query=query,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale_value=self.scale,
                block_table=attn_metadata.block_tables,
                context_lens=attn_metadata.seq_lens,
                out=output,
                workspace=workspace,
            )
            handle = torch.npu.graph_task_group_end(stream)
            graph_params.handles[num_tokens].append(handle)
            return output

    def full_graph_fa3(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: "AscendMetadata",
        output: torch.Tensor,
        block_size: int,
        block_table: torch.Tensor | None,
        actual_seq_lengths_kv: list[int] | torch.Tensor,
        fa3_graph_params: tuple | None = None,
    ):
        """FA3 graph capture — NPUGraph driver-level recording (decode + prefill).

        FA3 (PyTorch CustomOp) is invisible to the CANN task-group mechanism,
        so no ``graph_task_group_begin/End`` wrappers are used — an empty task
        group (FA3 not captured by CANN) corrupts CANN runtime state for
        subsequent graph captures (e.g. the prefill CANN V1 graph), breaking
        prefill accuracy.

        ``fa3_graph_params`` is the cached (scheduler_metadata, cache_seqlens,
        cu_seqlens_q, block_table, baked_max_seqlen_q) tuple from
        ``_get_fa3_graph_params``.  All buffers are fixed-size NPU tensors
        whose addresses are captured and whose data is refreshed before each
        replay by ``refresh_fa3_graph_params``.  ``max_seqlen_q`` passed to
        the kernel is the BAKED bucket bound (required to match the metadata
        fingerprint); the kernel derives real per-seq boundaries from
        cu_seqlens_q.
        """
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]

        from flash_attn_npu_3 import flash_attn_with_kvcache as fa3_kvcache

        num_blocks, bs = key.shape[0], key.shape[1]
        k_fa = key.view(num_blocks, bs, self.num_kv_heads, self.head_size)
        v_fa = value.view(num_blocks, bs, self.num_kv_heads, self.head_size)

        scheduler_metadata, cache_seqlens, cu_seqlens_q, block_table_buf, baked_max_seqlen_q = (
            fa3_graph_params
        )

        causal = attn_metadata.causal
        window_size = (
            (self.sliding_window, 0)
            if causal and self.sliding_window is not None
            else (-1, -1)
        )

        attn_output = fa3_kvcache(
            query,
            k_fa,
            v_fa,
            cache_seqlens=cache_seqlens,
            page_table=block_table_buf,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=baked_max_seqlen_q,
            softmax_scale=self.scale,
            causal=causal,
            window_size=window_size,
            scheduler_metadata=scheduler_metadata,
        )

        attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output[:num_tokens]
        return output, num_tokens

    def _get_fia_params(self, key: torch.Tensor, value: torch.Tensor, attn_metadata: AscendMetadata, kv_cache=None):
        # PrefillNoCache doesn't need key_cache, but other modes do
        # Only initialize/require cache for modes that actually use it
        if attn_metadata.attn_state != AscendAttentionState.PrefillNoCache:
            # Initialize cache from kv_cache if not already set (for DecodeOnly mode)
            if self.key_cache is None and kv_cache is not None:
                if (
                    isinstance(kv_cache, torch.Tensor)
                    and kv_cache.dim() > 0
                    and kv_cache.shape[0] == 2
                    or isinstance(kv_cache, (list, tuple))
                    and len(kv_cache) >= 2
                ):
                    self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

            if self.key_cache is None:
                raise RuntimeError(
                    f"key_cache is None in _get_fia_params for mode {attn_metadata.attn_state}. kv_cache={kv_cache}"
                )

        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            block_size = 128
            block_table = None
            actual_seq_lengths_kv = attn_metadata.actual_seq_lengths_q
            if self.attn_type == AttentionType.ENCODER_DECODER:
                actual_seq_lengths_kv = torch.cumsum(attn_metadata.seq_lens, dim=0).tolist()
        elif attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
            batch_size = attn_metadata.seq_lens.shape[0]
            block_table = attn_metadata.block_tables[:batch_size, :]
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        # chunked prefill.
        else:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        return key, value, block_size, block_table, actual_seq_lengths_kv

    def forward_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        kv_cache=None,
    ):
        # we inherit ForwardContext in model runner v2, when enable model
        # runner v2, there is not capturing attribute in forward_context,
        # just use getattr to avoid attribute error.
        if _EXTRA_CTX.capturing:
            if (
                self._fa3_eligible(attn_metadata)
                and _in_full_capture_stream()
                and not _no_fa3_graph_capture()
                and (
                    (
                        attn_metadata.attn_state == AscendAttentionState.DecodeOnly
                        and _fa3_decode_graph_enabled()
                    )
                    or (
                        attn_metadata.attn_state != AscendAttentionState.DecodeOnly
                        and _fa3_prefill_graph_enabled()
                    )
                )
            ):
                # FA3 graph capture (opt-in per state, see the two enable
                # helpers above for the current validation status).  Only
                # reachable inside a FULL-mode NPUGraph capture where the FA3
                # call and its fixed-size buffers are address-captured;
                # PIECEWISE residue capturing falls through to the eager path
                # below so the capture-step output uses the real batch.
                # Unseeded buckets return None and fall back to CANN V1.
                key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(
                    key, value, attn_metadata, kv_cache,
                )
                num_tokens = attn_metadata.actual_seq_lengths_q[-1]
                fa3_graph_params = self._get_fa3_graph_params(
                    num_tokens, attn_metadata, block_size, query,
                )
                if fa3_graph_params is not None:
                    attn_output, num_tokens = self.full_graph_fa3(
                        query, key, value, attn_metadata, output,
                        block_size=block_size,
                        block_table=block_table,
                        actual_seq_lengths_kv=actual_seq_lengths_kv,
                        fa3_graph_params=fa3_graph_params,
                    )
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
            if self.sinks is not None:
                attn_output, num_tokens = self.full_graph_fia_v2(query, key, value, attn_metadata, output)
                output[:num_tokens] = attn_output[:num_tokens]
                return output
            else:
                attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output)
                output[:num_tokens] = attn_output[:num_tokens]
                return output
        passed_value = value
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(
            key, value, attn_metadata, kv_cache
        )
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        query = query[:num_tokens]
        if (
            attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
            and self.attn_type != AttentionType.ENCODER_DECODER
        ):
            key = key[:num_tokens]
            value = value[:num_tokens]
        # Get workspace from cache or calculate it if not present.
        if self.sinks is not None:
            actual_seq_qlen = attn_metadata.actual_seq_lengths_q
            if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                actual_seq_qlen = torch.tensor([1] * len(attn_metadata.seq_lens_list), dtype=torch.int32).cumsum(dim=0)
            if self.sliding_window is not None:
                sparse_mode = 4
            else:
                sparse_mode = 3
            attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
                query,
                key,
                value,
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="TND",
                pre_tokens=self.sliding_window if self.sliding_window is not None else SWA_INT_MAX,
                next_tokens=0,
                atten_mask=attn_metadata.attn_mask,
                sparse_mode=sparse_mode,
                softmax_scale=self.scale,
                block_table=block_table,
                block_size=block_size,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_lengths_kv,
                learnable_sink=self.sinks,
            )
        else:
            if self._fa3_eligible(attn_metadata):
                from vllm_ascend.attention.fa3_adapter import fa3_forward

                is_cache = attn_metadata.attn_state != AscendAttentionState.PrefillNoCache
                try:
                    # Build FA3 scheduler metadata for the current batch
                    # (eager path — fresh each iteration).  Skip during the
                    # memory-profile run: get_scheduler_metadata allocates
                    # buffers sized by max_model_len, shrinking the measured
                    # free memory and thus the KV cache, which corrupts prefill.
                    if _EXTRA_CTX.in_profile_run:
                        scheduler_metadata = None
                    else:
                        scheduler_metadata = self._build_fa3_scheduler_metadata(
                            attn_metadata, block_size, query,
                        )
                    attn_output = fa3_forward(
                        query, key, value,
                        attn_metadata=attn_metadata,
                        scale=self.scale,
                        num_heads=self.num_heads,
                        num_kv_heads=self.num_kv_heads,
                        head_size=self.head_size,
                        sliding_window=self.sliding_window,
                        causal=attn_metadata.causal,
                        cache_mode=is_cache,
                        block_table=block_table if is_cache else None,
                        seq_lens_list=actual_seq_lengths_kv if is_cache else None,
                        scheduler_metadata=scheduler_metadata,
                    )
                except (ImportError, ValueError, RuntimeError, TypeError) as exc:
                    # FA3 unavailable for this invocation (e.g. head_dim too
                    # large, or FA3 package not importable) → fall back to
                    # the CANN path below.
                    logger.warning(
                        "FA3 forward failed for %s (q_lens=%s seq_lens_list=%s "
                        "max_query_len=%s; %s); falling back to CANN FIA.",
                        attn_metadata.attn_state,
                        attn_metadata.actual_seq_lengths_q[:8],
                        list(attn_metadata.seq_lens_list)[:8],
                        getattr(attn_metadata, "max_query_len", None), exc,
                    )
                    self._fa3_enabled = False
                else:
                    attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output

            # CANN V1 path: reached when FA3 is disabled / tripped / not
            # eligible for this state (e.g. ENCODER_DECODER, sinks).
            if not attn_metadata.causal:
                    attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                        query=query,
                        key=key,
                        value=value,
                        block_table=block_table,
                        input_layout="TND",
                        block_size=block_size,
                        actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                        actual_seq_lengths_kv=actual_seq_lengths_kv,
                        num_key_value_heads=self.num_kv_heads,
                        num_heads=self.num_heads,
                        scale=self.scale,
                        sparse_mode=0,
                    )
            elif self.sliding_window is not None:
                attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                    query=query,
                    key=key,
                    value=value,
                    atten_mask=attn_metadata.attn_mask,
                    block_table=block_table,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    num_key_value_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale=self.scale,
                    pre_tokens=self.sliding_window,
                    next_tokens=0,
                    sparse_mode=4,
                )
            else:
                attn_output, _ = DeviceOperator.npu_fused_infer_attention_score(
                    query=query,
                    key=key,
                    value=value,
                    atten_mask=attn_metadata.attn_mask,
                    block_table=block_table,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    num_key_value_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    head_size=self.head_size,
                    scale=self.scale,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    current_key=key,
                    current_value=passed_value,
                    attn_metadata=attn_metadata,
                    is_prefill_no_cache=attn_metadata.attn_state == AscendAttentionState.PrefillNoCache,
                    sparse_mode=3,
                )

            attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output[:num_tokens]
        return output

    def forward_paged_attention(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if _EXTRA_CTX.capturing:
            return self.full_graph_pa(query, attn_metadata, output)
        torch_npu._npu_paged_attention(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            block_table=attn_metadata.block_tables,
            context_lens=attn_metadata.seq_lens,
            out=output,
        )
        return output

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        _: torch.Tensor,
    ) -> torch.Tensor:
        # use default sparse_mode 0 in normal scenario, which means no mask works on it
        # Pad actual_seq_len with 0 when num_tokens > actual_seq_len in TND layout
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        if query.shape[0] > actual_seq_qlen[-1]:
            actual_seq_qlen = actual_seq_qlen + [0]
        return torch_npu.npu_fusion_attention(
            query=query,
            key=key,
            value=value,
            head_num=self.num_heads,
            input_layout="TND",
            scale=self.scale,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_qlen,
        )[0]

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: list[torch.Tensor],
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY):
            return

        if self.key_cache is None:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

        DeviceOperator.reshape_and_cache(
            key=key,
            value=value,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            slot_mapping=slot_mapping,
        )

    def reshape_and_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        if len(kv_cache) > 1:
            if self.key_cache is None:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
            if self.kv_sharing_target_layer_name is not None:
                # KV-sharing target layers consume another layer's cache.
                # Writing their dummy/current K/V would overwrite shared slots.
                if self.is_kv_producer:
                    attn_metadata.reshape_cache_event.record()
                return query, key, value, output
            slots = attn_metadata.slot_mapping
            encoder_decoder = self.attn_type == AttentionType.ENCODER_DECODER
            DeviceOperator.reshape_and_cache(
                key=key[: attn_metadata.num_actual_tokens] if not encoder_decoder else key,
                value=value[: attn_metadata.num_actual_tokens] if not encoder_decoder else value,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                # quick fix to make sure slots is int32 for cross attention case.
                # see: https://github.com/vllm-project/vllm/blob/ce88756b967c2c5006746a424c15dd59a284ed8c/vllm/model_executor/layers/attention/cross_attention.py#L117
                slot_mapping=slots[: attn_metadata.num_actual_tokens] if not encoder_decoder else slots.to(torch.int32),
            )
            notify_kv_cache_written()
        return query, key, value, output

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        record_attention_compute_start()
        num_tokens = query.shape[0]

        if (
            attn_metadata.attn_state == AscendAttentionState.DecodeOnly
            and self.sliding_window is None
            and using_paged_attention(num_tokens, self.vllm_config, self.head_size)
        ):
            output = self.forward_paged_attention(query, attn_metadata, output)
        else:
            output = self.forward_fused_infer_attention(query, key, value, attn_metadata, output, kv_cache)

        return output

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.
        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."
        if self._use_layer_aware_fia_graph_replay:
            self._layer_name = layer.layer_name

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported for AscendAttentionBackendImpl")

        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)

        # Initialize key_cache and value_cache from kv_cache if not already set.
        # This is needed for DecodeOnly mode where key/value are None but we still
        # need access to the cache for attention computation.
        if self.key_cache is None and kv_cache is not None:
            if (
                isinstance(kv_cache, torch.Tensor)
                and kv_cache.dim() > 0
                and kv_cache.shape[0] == 2
                or isinstance(kv_cache, (list, tuple))
                and len(kv_cache) >= 2
            ):
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

        output_padded = None
        if key is not None and value is not None:
            output_padded = output
            query, key, value, output_padded = self.reshape_and_cache(
                query, key, value, kv_cache, attn_metadata, output
            )
        # pooling model branch
        if attn_metadata.model_runner_type == "pooling" and not attn_metadata.causal:
            attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
            output[:num_tokens] = attn_output[:num_tokens]
            return output
        if output_padded is not None:
            attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output_padded)
        else:
            attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output)
        output[:num_tokens] = attn_output[:num_tokens]
        return output


class AscendC8AttentionBackendImpl(AscendAttentionBackendImpl):
    """Attention backend implementation for INT8 KV cache (C8/QuaRot) models.

    This subclass handles static per-channel INT8 KV cache quantization.
    It is activated via class surgery in AscendC8KVCacheAttentionMethod.create_weights
    (vllm_ascend/quantization/methods/kv_c8.py)
    so that C8 attention layers automatically use this forward path.
    """

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if self._use_layer_aware_fia_graph_replay:
            self._layer_name = layer.layer_name

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported for AscendC8AttentionBackendImpl")

        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)

        self._prepare_c8_scales(layer, query.device)
        float_key, float_value = None, None
        if self.vllm_config.kv_transfer_config is None:
            if key is not None and value is not None:
                if attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
                    float_key, float_value = key, value
                key, value = self._quantize_kv_to_int8(key, value, layer, attn_metadata.num_actual_tokens)
                query, key, value, _ = self._reshape_and_cache(query, key, value, kv_cache, attn_metadata, output)
            # pooling model branch
            if attn_metadata.model_runner_type == "pooling":
                attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
                output[:num_tokens] = attn_output[:num_tokens]
                return output
            if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                if _EXTRA_CTX.capturing:
                    attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output, layer)
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
                return self._forward_c8_decode(query, attn_metadata, output, layer)
            elif attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill:
                return self._forward_c8_chunked_prefill(query, float_key, float_value, attn_metadata, output, layer)
            else:
                return self._forward_c8_fused_infer_attention(
                    query,
                    float_key if float_key is not None else key,
                    float_value if float_value is not None else value,
                    attn_metadata,
                    output,
                    layer,
                )
        else:
            if attn_metadata.attn_state != AscendAttentionState.DecodeOnly and self.is_kv_producer:
                output_padded = None
                if key is not None and value is not None:
                    output_padded = output
                    query, key, value, output_padded = self.reshape_and_cache(
                        query, key, value, kv_cache, attn_metadata, output
                    )
                # pooling model branch
                if attn_metadata.model_runner_type == "pooling":
                    attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
                if output_padded is not None:
                    attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output_padded)
                else:
                    attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output)
                output[:num_tokens] = attn_output[:num_tokens]
                return output
            elif not self.is_kv_producer:
                if key is not None and value is not None:
                    key, value = self._quantize_kv_to_int8(key, value, layer, attn_metadata.num_actual_tokens)
                    query, key, value, _ = self._reshape_and_cache(query, key, value, kv_cache, attn_metadata, output)
                # pooling model branch
                if attn_metadata.model_runner_type == "pooling":
                    attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
                if _EXTRA_CTX.capturing:
                    attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output, layer)
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
                elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                    return self._forward_c8_decode(query, attn_metadata, output, layer)

    def _nz_5d_view(self, cache: torch.Tensor, block_size: int) -> torch.Tensor:
        """View a KV cache tensor in NZ 5D layout: (num_blocks, num_kv_heads, head_size//nz, block_size, nz)."""
        NZ_FMT_LAST_DIM = 32
        return cache.view(-1, self.num_kv_heads, self.head_size // NZ_FMT_LAST_DIM, block_size, NZ_FMT_LAST_DIM)

    def _prepare_c8_scales(self, layer: AttentionLayer, device: torch.device) -> None:
        """Shard per-channel C8 scales/offsets to this TP rank and pre-compute
        BF16 BNSD antiquant tensors for FIA V1 decode fast path.
        """
        if hasattr(layer, "_c8_scales_prepared"):
            return

        def _shard_and_reshape(raw: torch.Tensor) -> torch.Tensor:
            if raw.numel() == 1:
                return raw.to(device=device)
            expected = self.num_kv_heads * self.head_size
            if raw.numel() != expected:
                total_kv_heads = raw.numel() // self.head_size
                tp_rank = get_tensor_model_parallel_rank()
                tp_size = get_tensor_model_parallel_world_size()
                kv_head_start = tp_rank * total_kv_heads // tp_size
                raw = raw.view(total_kv_heads, self.head_size)[
                    kv_head_start : kv_head_start + self.num_kv_heads
                ].contiguous()
            return raw.view(1, self.num_kv_heads, self.head_size).to(device=device)

        layer._c8_k_scale = _shard_and_reshape(layer.k_cache_scale.data)
        layer._c8_k_offset = _shard_and_reshape(layer.k_cache_offset.data)
        layer._c8_v_scale = _shard_and_reshape(layer.v_cache_scale.data)
        layer._c8_v_offset = _shard_and_reshape(layer.v_cache_offset.data)

        layer._c8_k_inv_scale = 1.0 / layer._c8_k_scale
        layer._c8_v_inv_scale = 1.0 / layer._c8_v_scale

        nz_bnsd = (self.num_kv_heads, 1, self.head_size)
        layer._c8_k_aq_scale_nz_bnsd = layer._c8_k_scale.view(nz_bnsd).contiguous()
        layer._c8_v_aq_scale_nz_bnsd = layer._c8_v_scale.view(nz_bnsd).contiguous()

        layer._c8_scales_prepared = True

    def _dequant_paged_kv_to_dense(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: list,
        target_dtype: torch.dtype,
        layer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather paged INT8 KV blocks and dequantize."""
        batch_size = block_table.shape[0]
        max_blocks_per_seq = block_table.shape[1]

        # NZ 5D view: (num_blocks, num_kv_heads, head_size//nz, block_size, nz)
        block_size = self.key_cache.shape[1]  # type: ignore[attr-defined]
        max_tokens_padded = max_blocks_per_seq * block_size

        flat_ids = block_table.reshape(-1)
        key_nz = self._nz_5d_view(key, block_size)
        value_nz = self._nz_5d_view(value, block_size)

        # Gather: (batch*max_blocks, H, D//nz, S, nz)
        gathered_k = key_nz[flat_ids]
        gathered_v = value_nz[flat_ids]
        # NZ→ND conversion: permute (S, H, D//nz, nz) → reshape (S, H, D)
        gathered_k = (
            gathered_k.permute(0, 3, 1, 2, 4)
            .contiguous()
            .view(batch_size, max_tokens_padded, self.num_kv_heads, self.head_size)
        )
        gathered_v = (
            gathered_v.permute(0, 3, 1, 2, 4)
            .contiguous()
            .view(batch_size, max_tokens_padded, self.num_kv_heads, self.head_size)
        )

        seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=key.device)
        positions = torch.arange(max_tokens_padded, dtype=torch.long, device=key.device)
        valid_mask = (positions.unsqueeze(0) < seq_lens_t.unsqueeze(1)).view(-1)

        dense_k = gathered_k.view(-1, self.num_kv_heads, self.head_size)[valid_mask]
        dense_v = gathered_v.view(-1, self.num_kv_heads, self.head_size)[valid_mask]

        # Scale-only dequant for NZ (symmetric)
        dense_k = dense_k.to(target_dtype) * layer._c8_k_scale
        dense_v = dense_v.to(target_dtype) * layer._c8_v_scale
        return dense_k, dense_v

    def _quantize_kv_to_int8(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: AttentionLayer,
        num_actual_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize K/V from float to INT8 using static per-channel C8 scales."""
        actual_key = key[:num_actual_tokens]
        actual_value = value[:num_actual_tokens]

        k_int8 = torch.clamp(
            torch.round(actual_key * layer._c8_k_inv_scale + layer._c8_k_offset),
            -128,
            127,
        ).to(torch.int8)
        v_int8 = torch.clamp(
            torch.round(actual_value * layer._c8_v_inv_scale + layer._c8_v_offset),
            -128,
            127,
        ).to(torch.int8)
        return k_int8, v_int8

    def _forward_c8_decode(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ) -> torch.Tensor:
        """C8 decode via FIA V1 BNSD with native paged INT8 KV + perchannel antiquant."""
        num_block, block_size, _, _ = self.key_cache.shape  # type: ignore[attr-defined]
        assert block_size % 32 == 0, f"C8 INT8 KV cache requires block_size to be a multiple of 32, got {block_size}"
        batch_size = len(attn_metadata.seq_lens_list)

        key = self._nz_5d_view(self.key_cache, block_size)
        value = self._nz_5d_view(self.value_cache, block_size)

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query[:batch_size].unsqueeze(2),
            key,
            value,
            key_antiquant_scale=layer._c8_k_aq_scale_nz_bnsd,
            value_antiquant_scale=layer._c8_v_aq_scale_nz_bnsd,
            block_table=attn_metadata.block_tables,
            actual_seq_lengths_kv=attn_metadata.seq_lens_list,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BNSD",
            scale=self.scale,
            block_size=block_size,
            antiquant_mode=0,
            key_antiquant_mode=0,
            value_antiquant_mode=0,
            inner_precise=1,
            sparse_mode=0,
        )
        attn_output = attn_output.squeeze(2)
        output[:batch_size] = attn_output
        return output

    def _forward_c8_chunked_prefill(
        self,
        query: torch.Tensor,
        float_key: torch.Tensor | None,
        float_value: torch.Tensor | None,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ) -> torch.Tensor:
        """C8 ChunkedPrefill: decode via FIA V1 BNSD paged INT8 (zero gather),
        prefill via FIA V1 TND with float KV (new) or gather+dequant (continuing).
        """
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_decodes = attn_metadata.num_decodes
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])  # type: ignore[index]

        if num_decode_tokens > 0:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore[attr-defined]
            assert block_size % 32 == 0, (
                f"C8 INT8 KV cache requires block_size to be a multiple of 32, got {block_size}"
            )
            kv_k = self._nz_5d_view(self.key_cache, block_size)
            kv_v = self._nz_5d_view(self.value_cache, block_size)

            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                query[:num_decode_tokens].unsqueeze(2),
                kv_k,
                kv_v,
                key_antiquant_scale=layer._c8_k_aq_scale_nz_bnsd,
                value_antiquant_scale=layer._c8_v_aq_scale_nz_bnsd,
                block_table=attn_metadata.block_tables[:num_decodes],
                actual_seq_lengths_kv=attn_metadata.seq_lens_list[:num_decodes],
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="BNSD",
                scale=self.scale,
                block_size=block_size,
                antiquant_mode=0,
                key_antiquant_mode=0,
                value_antiquant_mode=0,
                inner_precise=1,
                sparse_mode=0,
            )
            output[:num_decode_tokens] = attn_out.squeeze(2)

        if attn_metadata.num_prefills > 0:
            prefill_q = query[num_decode_tokens:num_tokens]

            prefill_seq_qlen = [
                actual_seq_qlen[i] - num_decode_tokens for i in range(num_decodes, len(actual_seq_qlen))
            ]

            all_new_prefill = True
            for i in range(num_decodes, len(attn_metadata.seq_lens_list)):
                q_start = actual_seq_qlen[i - 1] if i > 0 else 0
                qlen_i = actual_seq_qlen[i] - q_start
                if attn_metadata.seq_lens_list[i] > qlen_i:
                    all_new_prefill = False
                    break

            if all_new_prefill and float_key is not None and float_value is not None:
                prefill_k = float_key[num_decode_tokens:num_tokens]
                prefill_v = float_value[num_decode_tokens:num_tokens]
                prefill_seq_kvlen = prefill_seq_qlen
            else:
                num_block, blk_size, _, _ = self.key_cache.shape  # type: ignore[attr-defined]
                paged_k = self._nz_5d_view(self.key_cache, blk_size)
                paged_v = self._nz_5d_view(self.value_cache, blk_size)

                prefill_bt = attn_metadata.block_tables[num_decodes:]
                prefill_sl = attn_metadata.seq_lens_list[num_decodes:]
                prefill_k, prefill_v = self._dequant_paged_kv_to_dense(
                    paged_k, paged_v, prefill_bt, prefill_sl, query.dtype, layer
                )
                prefill_seq_kvlen = torch.tensor(prefill_sl, dtype=torch.int32).cumsum(dim=0)

            # block_table is None for prefill; FIA ignores block_size in this case.
            # Use cache block_size for consistency rather than a magic number.
            cache_block_size = self.key_cache.shape[1]  # type: ignore[attr-defined]
            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                query=prefill_q,
                key=prefill_k,
                value=prefill_v,
                atten_mask=attn_metadata.attn_mask,
                block_table=None,
                input_layout="TND",
                block_size=cache_block_size,
                actual_seq_lengths=prefill_seq_qlen,
                actual_seq_lengths_kv=prefill_seq_kvlen,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                sparse_mode=3,
            )
            n_prefill = num_tokens - num_decode_tokens
            attn_out = attn_out.view(n_prefill, self.num_heads, self.head_size)
            output[num_decode_tokens:num_tokens] = attn_out[:n_prefill]

        return output

    def _forward_c8_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ):
        """C8 FIA V1 TND for prefill states (PrefillNoCache uses float KV directly,
        PrefillCacheHit gathers + dequants paged INT8 KV).
        """
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)

        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])  # type: ignore[index]
        query = query[:num_tokens]

        if (
            attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
            and self.attn_type != AttentionType.ENCODER_DECODER
        ):
            key = key[:num_tokens]
            value = value[:num_tokens]

        if key.dtype == torch.int8:
            if block_table is not None:
                seq_lens = (
                    actual_seq_lengths_kv if isinstance(actual_seq_lengths_kv, list) else actual_seq_lengths_kv.tolist()
                )
                key, value = self._dequant_paged_kv_to_dense(key, value, block_table, seq_lens, query.dtype, layer)
                block_table = None
                # block_table is None after dequant; FIA ignores block_size.
                # Use cache block_size for consistency rather than a magic number.
                block_size = self.key_cache.shape[1]  # type: ignore[attr-defined]
                actual_seq_lengths_kv = torch.tensor(seq_lens, dtype=torch.int32).cumsum(dim=0)
            else:
                key = (key.to(query.dtype) - layer._c8_k_offset) * layer._c8_k_scale
                value = (value.to(query.dtype) - layer._c8_v_offset) * layer._c8_v_scale

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=actual_seq_qlen,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )
        attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output
        return output

    def _reshape_and_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        if len(kv_cache) > 1:
            if self.key_cache is None:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
            if self.kv_sharing_target_layer_name is not None:
                # C8/NZ cache writes follow the same KV-sharing rule.
                if self.is_kv_producer:
                    attn_metadata.reshape_cache_event.record()
                return query, key, value, output
            slots = attn_metadata.slot_mapping

            encoder_decoder = self.attn_type == AttentionType.ENCODER_DECODER

            # NZ write path: 5D view + npu_scatter_pa_kv_cache
            block_size = self.vllm_config.cache_config.block_size
            k_cache_layer = self._nz_5d_view(self.key_cache, block_size)
            v_cache_layer = self._nz_5d_view(self.value_cache, block_size)

            torch_npu.npu_scatter_pa_kv_cache(
                key=key[: attn_metadata.num_actual_tokens] if not encoder_decoder else key,
                value=value[: attn_metadata.num_actual_tokens] if not encoder_decoder else value,
                key_cache=k_cache_layer,
                value_cache=v_cache_layer,
                slot_mapping=slots[: attn_metadata.num_actual_tokens] if not encoder_decoder else slots,
            )

            notify_kv_cache_written()
        return query, key, value, output
