# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import functools
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version

from packaging.version import InvalidVersion, Version

import vllm_ascend.envs as envs_ascend


@functools.cache
def vllm_version_is(target_vllm_version: str) -> bool:
    if envs_ascend.VLLM_VERSION is not None:
        vllm_version = envs_ascend.VLLM_VERSION
    elif (vllm_module := sys.modules.get("vllm")) is not None:
        vllm_version = vllm_module.__version__
    else:
        # This helper is also needed before importing vLLM. Reading package
        # metadata avoids triggering vLLM's import-time platform setup.
        try:
            vllm_version = distribution_version("vllm")
        except PackageNotFoundError:
            # Source-only developer environments may not have distribution
            # metadata. Importing the lightweight package root preserves the
            # existing vllm_version_is() behavior for that setup.
            import vllm

            vllm_version = vllm.__version__

    try:
        installed_version = Version(vllm_version)
        target_version = Version(target_vllm_version)
        # Source and device-specific wheels may append a PEP 440 local version
        # (for example, ``0.26.0+empty``). The local suffix does not change the
        # upstream vLLM release that compatibility gates target.
        return installed_version.public == target_version.public
    except InvalidVersion:
        raise ValueError(
            f"Invalid vllm version {vllm_version} found. A dev version of vllm "
            "is installed probably. Set the environment variable VLLM_VERSION "
            "to control it by hand. And please make sure the value follows the "
            "format of x.y.z."
        )
