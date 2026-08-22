# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Feature capability boundary for the Ascend 310P Model Runner V2."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MRv2FeatureSupport:
    """Capabilities implemented by one 310P MRv2 release.

    Later releases opt in only after their runtime implementation and hardware
    tests land. Keeping this object on the runner provides an extension point
    without exposing an environment variable that can enable incomplete code.
    """

    prefix_caching: bool = False
    qwen3_5_mtp: bool = False


FIRST_RELEASE_FEATURE_SUPPORT = MRv2FeatureSupport()
