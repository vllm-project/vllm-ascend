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

from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor
from vllm.logger import logger

from vllm_ascend._310p.dflash_full_decode_only import (
    is_310p_dflash_full_decode_only,
)


class DFlashFullDecodeManifestError(RuntimeError):
    """Raised when startup did not produce every required FULL graph."""


@dataclass(frozen=True)
class DFlashFullDecodeManifestKey:
    component: str
    local_rank: int
    graph_mode: CUDAGraphMode
    descriptor: BatchDescriptor


@dataclass(frozen=True)
class DFlashFullDecodeManifestRecord:
    key: DFlashFullDecodeManifestKey
    capture_count: int
    warmup_replay_count: int
    output_bound: bool
    contract_validated: bool


_full_decode_capture_manifest: dict[
    DFlashFullDecodeManifestKey,
    DFlashFullDecodeManifestRecord,
] = {}


def get_full_decode_local_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def reset_full_decode_capture_manifest() -> None:
    _full_decode_capture_manifest.clear()


def get_full_decode_capture_manifest() -> dict[
    DFlashFullDecodeManifestKey,
    DFlashFullDecodeManifestRecord,
]:
    return dict(_full_decode_capture_manifest)


def remove_full_decode_capture_for_test(
    key: DFlashFullDecodeManifestKey,
) -> None:
    _full_decode_capture_manifest.pop(key)


def record_full_decode_capture(
    *,
    component: str,
    local_rank: int,
    runtime_mode: CUDAGraphMode,
    descriptor: BatchDescriptor,
    capture_count: int,
    warmup_replay_count: int,
    output_bound: bool,
    contract_validated: bool,
) -> DFlashFullDecodeManifestRecord:
    if component not in {"target", "draft"}:
        raise DFlashFullDecodeManifestError(f"component must be target or draft, got {component!r}")
    if runtime_mode is not CUDAGraphMode.FULL:
        raise DFlashFullDecodeManifestError(f"graph_mode must be FULL, got {runtime_mode.name}")
    required = {
        "capture_count": capture_count == 1,
        "warmup_replay_count": warmup_replay_count >= 1,
        "output_bound": output_bound,
        "contract_validated": contract_validated,
    }
    for field, valid in required.items():
        if not valid:
            raise DFlashFullDecodeManifestError(
                f"incomplete FULL capture: component={component}, "
                f"rank={local_rank}, descriptor={descriptor}, {field} invalid"
            )
    if not descriptor.uniform or descriptor.num_reqs is None:
        raise DFlashFullDecodeManifestError(f"FULL decode descriptor is not uniform: {descriptor}")

    key = DFlashFullDecodeManifestKey(
        component=component,
        local_rank=local_rank,
        graph_mode=runtime_mode,
        descriptor=descriptor,
    )
    record = DFlashFullDecodeManifestRecord(
        key=key,
        capture_count=capture_count,
        warmup_replay_count=warmup_replay_count,
        output_bound=output_bound,
        contract_validated=contract_validated,
    )
    if key in _full_decode_capture_manifest:
        raise DFlashFullDecodeManifestError(f"duplicate FULL capture manifest entry: {key}")
    _full_decode_capture_manifest[key] = record
    logger.debug(
        "[310p-dflash-full-decode-only/manifest] event=record component=%s "
        "rank=%d mode=%s descriptor=%s capture_count=%d warmup_replay_count=%d",
        component,
        local_rank,
        runtime_mode.name,
        descriptor,
        capture_count,
        warmup_replay_count,
    )
    return record


def _expected_manifest_keys(
    vllm_config: VllmConfig,
    local_rank: int,
) -> set[DFlashFullDecodeManifestKey]:
    query_len = 1 + vllm_config.speculative_config.num_speculative_tokens
    max_tokens = query_len * vllm_config.scheduler_config.max_num_seqs
    capture_sizes = sorted(
        {
            int(size)
            for size in vllm_config.compilation_config.cudagraph_capture_sizes
            if query_len <= int(size) <= max_tokens and int(size) % query_len == 0
        }
    )
    return {
        DFlashFullDecodeManifestKey(
            component=component,
            local_rank=local_rank,
            graph_mode=CUDAGraphMode.FULL,
            descriptor=BatchDescriptor(
                num_tokens=size,
                num_reqs=size // query_len,
                uniform=True,
            ),
        )
        for component in ("target", "draft")
        for size in capture_sizes
    }


def validate_full_decode_capture_manifest(
    vllm_config: VllmConfig,
    *,
    local_rank: int,
) -> dict[DFlashFullDecodeManifestKey, DFlashFullDecodeManifestRecord]:
    if not is_310p_dflash_full_decode_only(vllm_config):
        return {}

    expected = _expected_manifest_keys(vllm_config, local_rank)
    actual = {key: record for key, record in _full_decode_capture_manifest.items() if key.local_rank == local_rank}
    missing = expected - actual.keys()
    if missing:
        details = ", ".join(
            f"component={key.component}/rank={key.local_rank}/"
            f"mode={key.graph_mode.name}/tokens={key.descriptor.num_tokens}"
            for key in sorted(
                missing,
                key=lambda item: (
                    item.component,
                    item.descriptor.num_tokens,
                ),
            )
        )
        raise DFlashFullDecodeManifestError(f"missing FULL capture manifest entries: {details}")

    validated = {key: actual[key] for key in expected}
    logger.info(
        "[310p-dflash-full-decode-only/manifest] event=complete rank=%d "
        "entries=%d components=target,draft descriptors=%s",
        local_rank,
        len(validated),
        sorted({key.descriptor.num_tokens for key in validated}),
    )
    return validated
