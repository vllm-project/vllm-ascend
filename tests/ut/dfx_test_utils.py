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

"""Shared test helpers for the DFX module.

Small factories that replace the repeated ``DfxRuntimeConfig(...)``
construction boilerplate in the DFX unit tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vllm_ascend.dfx.runtime_config import DfxRuntimeConfig


def make_dfx_config(
    tmp_path: Path,
    *,
    name: str = "dfx_config.json",
    report_dir: Path | None = None,
    **kwargs: Any,
) -> DfxRuntimeConfig:
    """Build a ``DfxRuntimeConfig`` pointed at a real (auto-created) JSON file.

    Defaults to a non-hot-reloading, file-synced config rooted at ``tmp_path``:
    ``ensure_file=True`` so the backing JSON exists for reads, and
    ``reload_interval_seconds=0`` so ``refresh_config`` never triggers a hot
    reload during a test. Pass extra kwargs (e.g. ``custom="..."``) through to
    the constructor for bespoke setups.
    """
    return DfxRuntimeConfig(
        tmp_path / name,
        report_dir=report_dir if report_dir is not None else tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=0,
        **kwargs,
    )
