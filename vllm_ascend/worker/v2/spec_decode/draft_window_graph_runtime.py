# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Parallel Draft Graph installation and persistent read-metadata binding.

One provider owns the captured read banks. Runtime preparation binds the
current proposal to those banks while preserving canonical Full write slots,
absolute query positions, and independent Target metadata.
"""

from __future__ import annotations

from typing import Any

from vllm.config.compilation import CUDAGraphMode

import vllm_ascend.envs as envs


def install_parallel_graph(owner: Any) -> None:
    """Install one provider and register every real startup capture bucket."""
    provider = getattr(owner, "_draft_window_graph_binding_provider", None)
    if provider is None:
        return
    if not envs.VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH:
        raise RuntimeError("Draft Graph configuration changed during initialization")
    if provider._carriers:
        raise RuntimeError("Draft Window Graph capture buckets are already installed")
    capture_descs = owner.query_cudagraph_manager._capture_descs
    descriptors = tuple(
        descriptor
        for mode_descriptors in capture_descs.values()
        for descriptor in mode_descriptors
        if descriptor.cg_mode is CUDAGraphMode.FULL
    )
    if not descriptors:
        raise RuntimeError("Draft Window Graph manager has no capture descriptors")
    for descriptor in descriptors:
        provider.register_capture_identity(
            descriptor=descriptor,
        )
