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

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from functools import wraps
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None


MRV2_DECODE_QUERY_LIMITS_KEY = "_mrv2_padding_decode_query_limits"
MRV2_DRAFT_DECODE_REQUEST_SIZES_KEY = "_mrv2_padding_draft_decode_request_sizes"
MRV2_VARLEN_DECODE_TOKEN_SIZES_KEY = "_mrv2_padding_varlen_decode_token_sizes"


def use_v2_model_runner(vllm_config: VllmConfig) -> bool:
    """Select the Ascend model runner.

    Model Runner V1 has been removed, so Model Runner V2 is the only
    implementation and this always returns ``True``.
    """
    return True


def _v1_runner_requested() -> bool:
    """Whether the environment explicitly selects Model Runner V1."""
    from vllm import envs

    value = getattr(envs, "VLLM_USE_V2_MODEL_RUNNER", None)
    if value is not None:
        return value is False
    # Upstream drops the entry together with Model Runner V1. Read the raw
    # variable so a stale ``=0`` still fails instead of being ignored.
    raw = os.getenv("VLLM_USE_V2_MODEL_RUNNER")
    if raw is None:
        return False
    try:
        return int(raw) == 0
    except ValueError:
        return False


def _validate_v2_model_runner(self) -> None:
    """Reject the configurations Model Runner V2 cannot serve.

    The feature list comes from upstream so it keeps growing with it. Only the
    Triton precondition is dropped: an Ascend deployment always ships Triton,
    while the CPU test lanes and config-only hosts import neither, and there
    is no V1 runner left to fall back to.
    """
    unsupported = self._get_v2_model_runner_unsupported_features()
    if unsupported:
        raise ValueError(f"Model Runner V2 does not yet support: {', '.join(unsupported)}")


def resolve_mrv2_decode_query_lens(
    vllm_config: VllmConfig,
    decode_query_len: int,
    upstream: Callable[[VllmConfig, int], dict[int, int]],
) -> dict[int, int]:
    """Reuse Ascend's aligned target tiers across config re-validation.

    EngineCore calls ``VllmConfig.__post_init__`` again after its startup
    handshake. Dynamic speculative decoding normally derives its tier mapping
    from ``max_num_seqs``; that field is the physical runner capacity after
    Ascend padding, while the schedule was derived from the smaller logical
    request limit. Re-deriving from the physical value would introduce tiers
    that the scheduler can never produce and invalidate the already-resolved
    graph sizes.
    """
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    aligned_limits = additional_config.get(MRV2_DECODE_QUERY_LIMITS_KEY)
    if (
        decode_query_len == getattr(vllm_config, "uniform_decode_query_len", None)
        and isinstance(aligned_limits, Mapping)
        and aligned_limits
    ):
        return {int(query_len): int(max_reqs) for query_len, max_reqs in aligned_limits.items()}
    return upstream(vllm_config, decode_query_len)


def resolve_mrv2_draft_decode_request_sizes(
    vllm_config: VllmConfig,
    upstream: Callable[[VllmConfig], list[int]],
) -> list[int]:
    """Reuse aligned one-token draft gears across config re-validation."""
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    if MRV2_DRAFT_DECODE_REQUEST_SIZES_KEY in additional_config:
        return [int(size) for size in additional_config[MRV2_DRAFT_DECODE_REQUEST_SIZES_KEY]]
    return upstream(vllm_config)


def resolve_mrv2_varlen_decode_token_sizes(
    vllm_config: VllmConfig,
    decode_query_len: int,
    upstream: Callable[[VllmConfig, int], list[int]],
) -> list[int]:
    """Reuse aligned adaptive-verification token gears after re-validation."""
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    aligned_sizes = additional_config.get(MRV2_VARLEN_DECODE_TOKEN_SIZES_KEY)
    if isinstance(aligned_sizes, Mapping):
        sizes = aligned_sizes.get(decode_query_len)
        if sizes is None:
            sizes = aligned_sizes.get(str(decode_query_len))
        if sizes is not None:
            return [int(size) for size in sizes]
    return upstream(vllm_config, decode_query_len)


def _patch_config_resolver(cls, name: str, marker: str, resolve: Callable) -> None:
    """Wrap one c410 graph resolver once, preserving its upstream callable."""
    upstream = getattr(cls, name)
    if getattr(upstream, marker, False):
        return

    @wraps(upstream)
    def patched(self, *args):
        return resolve(self, *args, upstream)

    setattr(patched, marker, True)
    cast(Any, patched)._vllm_ascend_upstream = upstream
    setattr(cls, name, patched)


def apply_v2_model_runner_config_patch() -> None:
    """Install Ascend runner selection on ``VllmConfig``.

    Re-apply in every process that reads ``use_v2_model_runner`` (frontend,
    workers, engine-core). Pickled configs do not carry this class patch.

    An explicit ``VLLM_USE_V2_MODEL_RUNNER=0`` raises: there is no V1 runner to
    select any more, and ignoring the setting would hide a stale deployment
    script.
    """
    if _v1_runner_requested():
        raise ValueError(
            "Model Runner V1 has been removed from vllm-ascend, so VLLM_USE_V2_MODEL_RUNNER=0 "
            "cannot be honored. Unset it to run Model Runner V2."
        )
    from vllm.config.vllm import VllmConfig

    VllmConfig.use_v2_model_runner = property(use_v2_model_runner)
    # The guard keeps this working once upstream renames the hook.
    if hasattr(VllmConfig, "_validate_v2_model_runner"):
        VllmConfig._validate_v2_model_runner = _validate_v2_model_runner
    # Main re-runs config validation after the EngineCore handshake. Keep the
    # per-config aligned target mapping stable across that re-entry. This class
    # patch is stateless: unlike a runner-capturing closure, it reads only the
    # current VllmConfig's serialized ``additional_config``.
    for name, marker, resolve in (
        ("get_decode_cudagraph_query_lens", "_mrv2_limits", resolve_mrv2_decode_query_lens),
        (
            "get_draft_decode_cudagraph_request_sizes",
            "_mrv2_draft",
            resolve_mrv2_draft_decode_request_sizes,
        ),
        (
            "get_varlen_decode_cudagraph_token_sizes",
            "_mrv2_varlen",
            resolve_mrv2_varlen_decode_token_sizes,
        ),
    ):
        _patch_config_resolver(VllmConfig, name, marker, resolve)
