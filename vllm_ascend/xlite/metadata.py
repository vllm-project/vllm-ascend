#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
#
"""V2 attention-metadata bridge for the xlite hybrid (MHA+GDN) forward path.

The xlite native runtime exposes a device-tensor metadata path
(:class:`xlite._C.AttnMetaV2` consumed by ``Model.forward_v2`` /
``Model.forward_with_inputs_embeds_v2``) that aliases pre-built device tensors
instead of recomputing slot mappings on the host and copying them to the
device every step (the V1 :class:`xlite._C.AttnMeta` behavior).

Everything the V2 path needs is already maintained per step by the
vLLM-Ascend attention metadata builder — this module only derives the two
vectors vLLM does not materialize directly (query lengths and cached lengths
as device tensors) and slices the rest:

======================  =====================================================
``AttnMetaV2`` field    Source on ``AscendMetadata``
======================  =====================================================
``lens``                ``diff(query_start_loc_gpu)`` (derived here, on-device)
``cached_lens``         ``seq_lens_gpu - lens`` (derived here, on-device)
``query_start_loc``     ``query_start_loc_gpu[:num_reqs]`` (exclusive prefix sum)
``slot_mapping``        ``slot_mapping[:num_actual_tokens]``, cast to int32
``block_tables``        ``block_tables[:num_reqs]``
``positions``           forward ``positions[:num_actual_tokens]``
``lens_cpu``            ``actual_seq_lengths_q`` differences (host list)
``cached_lens_cpu``     ``seq_lens_list - lens_cpu`` (host list)
======================  =====================================================

The derivations are tiny on-device kernels; no host<->device synchronization
is introduced. Note that vLLM already expands hybrid block tables and slot
mappings to *kernel* block ids (see ``vllm.v1.worker.block_table``), so both
line up with the kernel-block-sized xlite KV-cache views.
"""

from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import logger

# Older xlite releases do not ship AttnMetaV2; resolve it dynamically so the
# import itself never breaks those installs — the builder falls back to the
# native runnable with a clear error instead (see build_hybrid_step_metadata).
from xlite import _C as _xlite_C

from vllm_ascend.xlite.utils import AttnMetadataRouter

AttnMetaV2: Any = getattr(_xlite_C, "AttnMetaV2", None)


@dataclass
class HybridStepMetadata:
    """Per-step metadata for one hybrid xlite forward.

    Attributes:
        attn_meta (AttnMetaV2): Ready-to-use V2 metadata; device tensors alias
            the vLLM-Ascend buffers (plus the two derived length vectors).
        num_reqs (int): Number of real (non-padding) requests in the batch.
        num_actual_tokens (int): Total number of real tokens in the batch.
        query_lens (list[int]): Per-request query lengths (host).
        cached_lens (list[int]): Per-request cached (context) lengths (host).
    """

    attn_meta: Any
    num_reqs: int
    num_actual_tokens: int
    query_lens: list[int]
    cached_lens: list[int]


def _derive_host_lens(
    attn_metadata: Any,
    num_reqs: int,
) -> tuple[list[int], list[int]] | None:
    """Derive per-request query and sequence lengths on the host.

    Prefers the builder-maintained host lists (``actual_seq_lengths_q`` /
    ``seq_lens_list``); falls back to the metadata router, which may cost one
    device-to-host copy. Returns ``(query_lens, seq_lens)`` or ``None`` when
    the metadata is incomplete.
    """
    seq_lens_list = list(getattr(attn_metadata, "seq_lens_list", None) or [])[:num_reqs]
    actual_q = list(getattr(attn_metadata, "actual_seq_lengths_q", None) or [])
    if len(actual_q) >= num_reqs > 0:
        cum = [0] + [int(x) for x in actual_q[:num_reqs]]
        query_lens = [cum[i + 1] - cum[i] for i in range(num_reqs)]
    else:
        router = AttnMetadataRouter(attn_metadata=attn_metadata, device="cpu")
        seq = router.seq_lens[:num_reqs]
        cum_t = router.cu_query_lens[-seq.size(0) :][: num_reqs + 1]
        if cum_t.device.type != "cpu":
            cum_t = cum_t.to("cpu")
        query_lens = torch.diff(cum_t, prepend=cum_t.new_zeros(1)).tolist()
        if not seq_lens_list:
            seq_lens_list = seq.tolist()
    if len(seq_lens_list) < num_reqs or len(query_lens) < num_reqs:
        return None
    return [int(x) for x in query_lens], [int(x) for x in seq_lens_list]


def build_hybrid_step_metadata(
    attn_metadata: Any,
    positions: torch.Tensor,
    *,
    num_reqs_hint: int = 0,
) -> HybridStepMetadata | None:
    """Build the V2 xlite metadata for one hybrid forward step.

    Args:
        attn_metadata (Any): The full-attention :class:`AscendMetadata` picked
            from the forward context (see :meth:`XliteWrapper._pick_attn_metadata`).
        positions (torch.Tensor): Token positions of the current step,
            shape ``[num_tokens]`` or ``[1, num_tokens]``, int64 device.
        num_reqs_hint (int): Fallback request count (e.g. from the batch
            descriptor) used when the metadata carries no decode/prefill counts.

    Returns:
        HybridStepMetadata | None: The prepared metadata, or ``None`` when the
        input metadata is incomplete and the caller must fall back to the
        native runnable. Device tensors are zero-copy views of the vLLM-Ascend
        buffers; the two derived length vectors are computed on-device.
    """
    num_reqs = int(getattr(attn_metadata, "num_decodes", 0) or 0) + int(getattr(attn_metadata, "num_prefills", 0) or 0)
    if num_reqs <= 0:
        num_reqs = int(num_reqs_hint)
    if num_reqs <= 0:
        logger.error("xlite hybrid: metadata carries no requests; falling back to native")
        return None
    if AttnMetaV2 is None:
        logger.error(
            "xlite hybrid: the installed xlite lacks AttnMetaV2 (forward_v2); "
            "upgrade xlite to run hybrid models — falling back to native"
        )
        return None

    host_lens = _derive_host_lens(attn_metadata, num_reqs)
    if host_lens is None:
        # Falling back here means the native GDN layers would resume from
        # state this request never produced on xlite (the recurrent state is
        # not transferred across backends in full-hybrid mode) — the response
        # quality for this request is not guaranteed.
        logger.error(
            "xlite hybrid: metadata lens incomplete (reqs=%s); falling back to "
            "native with unbridged GDN state — output quality for in-flight "
            "requests is not guaranteed",
            num_reqs,
        )
        return None
    query_lens, seq_lens = host_lens

    num_actual_tokens = int(attn_metadata.num_actual_tokens)
    if sum(query_lens) != num_actual_tokens:
        logger.error(
            "xlite hybrid: query_lens sum %s != num_actual_tokens %s; falling back to native",
            sum(query_lens),
            num_actual_tokens,
        )
        return None
    cached_lens = [max(seq_lens[i] - query_lens[i], 0) for i in range(num_reqs)]

    # Device tensors maintained by the vLLM-Ascend metadata builder. All are
    # required: without them the V2 zero-copy path is impossible, and silently
    # rebuilding them on the host would reintroduce the per-step H2D copies.
    qsl_gpu = getattr(attn_metadata, "query_start_loc_gpu", None)
    seq_lens_gpu = getattr(attn_metadata, "seq_lens_gpu", None)
    slot_mapping = getattr(attn_metadata, "slot_mapping", None)
    block_tables = getattr(attn_metadata, "block_tables", None)
    if (
        not isinstance(qsl_gpu, torch.Tensor)
        or not isinstance(seq_lens_gpu, torch.Tensor)
        or not isinstance(slot_mapping, torch.Tensor)
        or not isinstance(block_tables, torch.Tensor)
        or qsl_gpu.numel() < num_reqs + 1
        or seq_lens_gpu.numel() < num_reqs
        or slot_mapping.numel() < num_actual_tokens
        or block_tables.shape[0] < num_reqs
    ):
        logger.error(
            "xlite hybrid: device metadata incomplete (qsl_gpu=%s seq_lens_gpu=%s "
            "slot_mapping=%s block_tables=%s, reqs=%s tokens=%s); falling back to native",
            None if not isinstance(qsl_gpu, torch.Tensor) else tuple(qsl_gpu.shape),
            None if not isinstance(seq_lens_gpu, torch.Tensor) else tuple(seq_lens_gpu.shape),
            None if not isinstance(slot_mapping, torch.Tensor) else tuple(slot_mapping.shape),
            None if not isinstance(block_tables, torch.Tensor) else tuple(block_tables.shape),
            num_reqs,
            num_actual_tokens,
        )
        return None

    # Refuse capture/dummy all-zero block tables on requests with context.
    # Dummy/profile/capture steps are routed away by the caller, so this is a
    # last-line check and only runs when the runner maintains a host copy of
    # the block tables (reading the device table here would sync the stream).
    block_tables_host = getattr(attn_metadata, "block_tables_cpu_np", None)
    if block_tables_host is not None:
        for i, cached in enumerate(cached_lens):
            if cached > 0 and (len(block_tables_host[i]) == 0 or int(block_tables_host[i][0]) == 0):
                logger.error("xlite hybrid: dummy block table on request %s; falling back to native", i)
                return None

    # lens / cached_lens as int32 device vectors; query_start_loc is the
    # exclusive prefix sum (xlite convention, matches the C++ host algorithm).
    lens_dev = qsl_gpu[1 : num_reqs + 1] - qsl_gpu[:num_reqs]
    cached_lens_dev = seq_lens_gpu[:num_reqs] - lens_dev

    meta = AttnMetaV2()
    meta.lens = lens_dev
    meta.cached_lens = cached_lens_dev
    meta.query_start_loc = qsl_gpu[:num_reqs]
    # vLLM keeps slot_mapping in int64; the xlite kernels read it as int32.
    meta.slot_mapping = [slot_mapping[:num_actual_tokens].to(torch.int32)]
    meta.block_tables = [block_tables[:num_reqs]]
    pos = positions[0] if positions.ndim == 2 else positions
    meta.positions = pos[:num_actual_tokens].contiguous()
    meta.lens_cpu = query_lens
    meta.cached_lens_cpu = cached_lens

    return HybridStepMetadata(
        num_reqs=num_reqs,
        num_actual_tokens=num_actual_tokens,
        query_lens=query_lens,
        cached_lens=cached_lens,
        attn_meta=meta,
    )
