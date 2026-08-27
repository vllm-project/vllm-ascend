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

import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from vllm_ascend.dfx.runtime_config import DfxRuntimeConfig

_REQUEST_STATE_LOGGER = "vllm_ascend.dfx.request_state"


@contextmanager
def capture_vllm_ascend_logs(
    caplog,
    level: int,
    *,
    logger_name: str = _REQUEST_STATE_LOGGER,
) -> Generator[None, None, None]:
    """Capture vllm_ascend logs via caplog after apply_ascend_log_level.

    ``apply_ascend_log_level`` sets ``vllm_ascend.propagate = False``, so pytest's
    root-level caplog handler never sees child records. Temporarily re-enable
    propagation and reset the target logger level (prior tests may pin dfx to
    ERROR via ``ascend_log.modules``).
    """
    ascend = logging.getLogger("vllm_ascend")
    target = logging.getLogger(logger_name)
    old_ascend_propagate = ascend.propagate
    old_target_level = target.level
    ascend.propagate = True
    target.setLevel(level)
    try:
        with caplog.at_level(level, logger=logger_name):
            yield
    finally:
        ascend.propagate = old_ascend_propagate
        target.setLevel(old_target_level)


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
