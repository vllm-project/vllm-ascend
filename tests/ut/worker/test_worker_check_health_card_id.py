#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""``check_health`` must query the card NPU ID, not the rank or chip Phy-ID."""

import logging
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import vllm_ascend.worker.worker as worker_module
from vllm_ascend.cpu_binding import DeviceInfo

# Mirrors the multi-chip host in #23272: chips Phy-ID 2 and 3 share card 1,
# and card 4 owns chips 8 and 9.
NPU_SMI_MAP = (
    "NPU ID  Chip ID  Chip Logic ID  Chip Phy-ID  Chip Name\n"
    "1 0 0 2 Ascend910\n"
    "1 1 1 3 Ascend910\n"
    "1 2 - - Mcu\n"
    "4 0 2 8 Ascend910\n"
    "4 1 3 9 Ascend910\n"
    "4 2 - - Mcu"
)


def _parse_map(npu_smi_map):
    """Parse the fixture the same way the production helper does."""
    lines = [ln.strip() for ln in npu_smi_map.splitlines() if ln.strip()]
    header = [c.strip() for c in lines[0].split("  ") if c.strip()]
    npu_idx, phy_idx = header.index("NPU ID"), header.index("Chip Phy-ID")
    out = {}
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) <= max(npu_idx, phy_idx):
            continue
        if parts[phy_idx].isdigit():
            out[parts[phy_idx]] = parts[npu_idx]
    return out


def _platform(visible, phy, *, visible_error=None):
    """Fake platform exposing the mapping entry points the worker uses."""

    def to_visible(_rank):
        if visible_error is not None:
            raise visible_error
        return visible

    return SimpleNamespace(
        logical_device_id_to_visible_device_id=to_visible,
        visible_device_id_to_physical_device_id=lambda _dev: phy,
        device_id_to_physical_device_id=lambda _rank: phy,
    )


def _probe(local_rank, platform, *, mapping=NPU_SMI_MAP, returncode=0, stdout="Health: OK"):
    """Drive ``NPUWorker.check_health``; return the argv handed to npu-smi."""
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="boom")

    worker = worker_module.NPUWorker.__new__(worker_module.NPUWorker)
    worker.local_rank = local_rank
    with (
        patch.object(worker_module, "current_platform", platform),
        patch.object(DeviceInfo, "get_npu_phy_id_to_card_id", return_value=_parse_map(mapping)),
        patch.object(subprocess, "run", fake_run),
    ):
        worker.check_health()
    return seen


@pytest.mark.parametrize(
    "local_rank,visible,chip_phy,expected_card",
    [
        # Two chips on card 1: the case a rank->Phy-ID mapping gets wrong.
        (0, 0, 2, 1),
        (1, 1, 3, 1),
        # ASCEND_RT_VISIBLE_DEVICES reordering: rank 0 -> visible 1 -> chip 8 -> card 4.
        (0, 1, 8, 4),
        (3, 3, 9, 4),
    ],
)
def test_check_health_queries_card_id(local_rank, visible, chip_phy, expected_card):
    """``-i`` receives the containing card's NPU ID for every mapping."""
    seen = _probe(local_rank, _platform(visible, chip_phy))

    assert seen["cmd"] == ["npu-smi", "info", "-i", str(expected_card), "-t", "health"]


def test_both_chips_of_one_card_query_the_same_card():
    """Chips 2 and 3 both live on card 1, so both ranks must ask for card 1.

    This is the assertion #23272 asked for: a rank -> Phy-ID mapping alone sends
    ``-i 2`` and ``-i 3``, and neither is a card on that host.
    """
    first = _probe(0, _platform(visible=0, phy=2))
    second = _probe(1, _platform(visible=1, phy=3))

    assert first["cmd"][3] == "1"
    assert second["cmd"][3] == "1"
    # Guard the regression directly: chip Phy-IDs must not leak through.
    assert first["cmd"][3] != "2"
    assert second["cmd"][3] != "3"


def test_visible_id_is_cast_from_string():
    """``logical_device_id_to_visible_device_id`` may return a string."""
    seen = _probe(0, _platform("1", 8))

    assert seen["cmd"] == ["npu-smi", "info", "-i", "4", "-t", "health"]


def test_resolution_failure_never_crashes_the_probe():
    """A raising first hop degrades to the fallback instead of propagating."""
    seen = _probe(0, _platform(0, 8, visible_error=RuntimeError("no CANN context")))

    assert seen["cmd"][:4] == ["npu-smi", "info", "-i", "8"]


def test_missing_chip_phy_column_falls_back():
    """Hosts without ``Chip Phy-ID`` keep the previous best-effort path.

    With no chip->card map the resolver falls through to
    ``device_id_to_physical_device_id``, which is correct on single-chip cards.
    """
    legacy = "NPU ID  Chip ID  Chip Logic ID  Chip Name\n0 0 0 Ascend910"
    seen = _probe(0, _platform(0, 0), mapping=legacy)

    assert seen["cmd"][:4] == ["npu-smi", "info", "-i", "0"]


def test_timeout_and_capture_flags_preserved():
    """Resolving the id must not change the rest of the probe contract."""
    seen = _probe(0, _platform(0, 2))

    assert seen["kwargs"]["capture_output"] is True
    assert seen["kwargs"]["text"] is True
    # The subprocess gets the remaining budget, never the full 10 s RPC window.
    assert 0 < seen["kwargs"]["timeout"] <= worker_module._HEALTH_CHECK_TIMEOUT_SECONDS


def test_nonzero_exit_stays_a_warning():
    """A non-zero npu-smi exit is still swallowed and logged, as before."""
    seen = _probe(0, _platform(0, 2), returncode=1)

    assert seen["cmd"][:4] == ["npu-smi", "info", "-i", "1"]


def test_unhealthy_card_still_detected(caplog):
    """The health verdict itself is unchanged by the id fix."""
    with caplog.at_level(logging.ERROR):
        _probe(0, _platform(0, 2), stdout="Health: ALARM")

    assert "health status is not OK" in caplog.text


def test_get_npu_phy_id_to_card_id_parses_multi_chip_map():
    """The chip -> card mapping itself, straight from ``npu-smi info -m``."""
    with patch("vllm_ascend.cpu_binding.execute_command", return_value=(NPU_SMI_MAP, 0)) as run:
        mapping = DeviceInfo.get_npu_phy_id_to_card_id()

    assert mapping == {"2": "1", "3": "1", "8": "4", "9": "4"}
    run.assert_called_once_with(["npu-smi", "info", "-m"], timeout=10)


def test_get_npu_phy_id_to_card_id_returns_empty_without_column():
    """No ``Chip Phy-ID`` column -> empty mapping so the caller can fall back."""
    legacy = "NPU ID  Chip ID  Chip Logic ID  Chip Name\n0 0 0 Ascend910"
    with patch("vllm_ascend.cpu_binding.execute_command", return_value=(legacy, 0)):
        assert DeviceInfo.get_npu_phy_id_to_card_id() == {}


def test_get_npu_phy_id_to_card_id_raises_on_failed_command():
    with patch("vllm_ascend.cpu_binding.execute_command", return_value=("", 1)):
        with pytest.raises(RuntimeError, match="failed with exit code"):
            DeviceInfo.get_npu_phy_id_to_card_id()
