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

import importlib.util
import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "ai_qos.py"
MODULE_NAME = "vllm_ascend_tools_ai_qos"

MASTER_IDS = (11, 12, 13, 7)


def _load_ai_qos_tool(mock_ai: MagicMock | None = None):
    if mock_ai is None:
        mock_ai = MagicMock()

        def get_qos_fn(device_id, master_id):
            return (0, master_id, 42, 0, 0, 0)

        mock_ai.get_qos.side_effect = get_qos_fn
        mock_ai.get_bw.return_value = (0, 1, 2, 0)
        mock_ai.get_fuse_mode.return_value = (0, 1, 1, 0)
        mock_ai.set_bw.return_value = 0
        mock_ai.set_qos.return_value = 0
        mock_ai.set_fuse_gbl_config.return_value = 0

    ascend = MagicMock()
    ascend.ai_qos = mock_ai

    sys.modules.pop(MODULE_NAME, None)
    with patch.dict(
        sys.modules,
        {
            "vllm_ascend": ascend,
        },
    ):
        spec = importlib.util.spec_from_file_location(MODULE_NAME, TOOL_PATH)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod, mock_ai


def test_device_list_uses_all_visible_devices_when_env_unset(monkeypatch):
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    mod, _ = _load_ai_qos_tool()
    mock_torch = MagicMock()
    mock_torch.npu.device_count.return_value = 4
    with patch.dict(sys.modules, {"torch": mock_torch}):
        assert mod._device_list() == [0, 1, 2, 3]


def test_device_list_exits_when_env_unset_and_torch_query_fails(monkeypatch):
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    mod, _ = _load_ai_qos_tool()
    mock_torch = MagicMock()
    mock_torch.npu.device_count.side_effect = RuntimeError("query failed")
    with patch.dict(sys.modules, {"torch": mock_torch}), pytest.raises(SystemExit) as e:
        mod._device_list()
    assert e.value.code == 1


def test_device_list_parses_visible_devices(monkeypatch):
    mod, _ = _load_ai_qos_tool()
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,2")
    assert mod._device_list() == [0, 2]


def test_device_list_parses_single_id(monkeypatch):
    mod, _ = _load_ai_qos_tool()
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "3")
    assert mod._device_list() == [3]


def test_print_config_block(capsys):
    mod, _ = _load_ai_qos_tool()
    mod._print_config_block(["line a", "line b"])
    out = capsys.readouterr().out
    assert "system-view" in out and "line a" in out and "line b" in out
    assert out.strip().endswith("commit")


def test_load_first_apply_baseline_no_file(tmp_path):
    mod, _ = _load_ai_qos_tool()
    p = tmp_path / "missing.json"
    assert mod._load_first_apply_baseline(p) is None


def test_load_first_apply_baseline_malformed_json(tmp_path):
    mod, _ = _load_ai_qos_tool()
    p = tmp_path / "x.json"
    p.write_text("{", encoding="utf-8")
    assert mod._load_first_apply_baseline(p) is None


def test_load_first_apply_baseline_invalid_original_qos(tmp_path):
    mod, _ = _load_ai_qos_tool()
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"original_qos": "bad"}), encoding="utf-8")
    assert mod._load_first_apply_baseline(p) is None


def test_load_first_apply_baseline_success(tmp_path):
    mod, _ = _load_ai_qos_tool()
    p = tmp_path / "x.json"
    body = {
        "original_qos": {"0": {"7": [7, 0, 0, 0, 0]}},
        "original_sdma_mata": {"0": [0, 1, 2, 0]},
        "original_fuse": {"0": [1, 1, 0]},
    }
    p.write_text(json.dumps(body), encoding="utf-8")
    b = mod._load_first_apply_baseline(p)
    assert b is not None
    oq, osm, ofu = b
    assert oq == body["original_qos"]
    assert osm == body["original_sdma_mata"]
    assert ofu == body["original_fuse"]


def test_run_unset_exits_without_state_file(capsys):
    mod, _ = _load_ai_qos_tool()
    with pytest.raises(SystemExit) as e:
        mod.run_unset(Path("/nonexistent/ai_qos_state.json"))
    assert e.value.code == 1
    err = capsys.readouterr().err
    assert "No state file" in err


def test_run_unset_parse_failed_bad_json_deletes_file(tmp_path, capsys):
    mod, _ = _load_ai_qos_tool()
    state = tmp_path / "ai_qos_state.json"
    state.write_text("{", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        mod.run_unset(state)
    assert e.value.code == 1
    assert not state.is_file()
    err = capsys.readouterr().err
    assert "Failed to parse the state file." in err


def test_run_unset_parse_failed_invalid_structure_deletes_file(tmp_path, capsys):
    mod, _ = _load_ai_qos_tool()
    state = tmp_path / "ai_qos_state.json"
    state.write_text(
        json.dumps(
            {
                "original_qos": {},
                "printed_commands": [123],
                "original_sdma_mata": {},
                "original_fuse": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as e:
        mod.run_unset(state)
    assert e.value.code == 1
    assert not state.is_file()
    assert "Failed to parse the state file." in capsys.readouterr().err


def test_run_unset_restores_and_deletes_file(tmp_path):
    mod, mock_ai = _load_ai_qos_tool()
    state = tmp_path / "ai_qos_state.json"
    data = {
        "original_qos": {
            "0": {
                str(MASTER_IDS[0]): [11, 1, 0, 0, 0],
            }
        },
        "original_sdma_mata": {"0": [0, 1, 2, 0]},
        "original_fuse": {"0": [1, 0, 0]},
        "printed_commands": ["hccs qos remap 1 0 0"],
    }
    state.write_text(json.dumps(data), encoding="utf-8")
    out_buf = StringIO()
    with patch("sys.stdout", out_buf):
        mod.run_unset(state)
    assert not state.is_file()
    assert mock_ai.set_bw.called
    assert mock_ai.set_qos.called
    assert mock_ai.set_fuse_gbl_config.called
    uo = out_buf.getvalue()
    assert "system-view" in uo
    assert "undo hccs qos remap 1 0 0" in uo
    assert "commit" in uo


def test_AiqosConfig_set_qos_captures_and_writes_state(tmp_path, monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0")
    mod, mock_ai = _load_ai_qos_tool()
    cfg = {
        "mode": "auto",
        "aiqos_priority": {
            "AIV_D2D": "high",
            "AIV_H2D": "high",
            "SDMA_D2D": "high",
            "SDMA_H2D": "low",
            "PCIEDMA_H2D": "high",
        },
    }
    out_buf = StringIO()
    with patch("sys.stdout", out_buf):
        mod.AiqosConfig(cfg).set_qos(tmp_path / "state.json")
    state = tmp_path / "state.json"
    assert state.is_file()
    j = json.loads(state.read_text(encoding="utf-8"))
    assert "original_qos" in j and "printed_commands" in j
    assert mock_ai.get_qos.call_count == 9
    assert mock_ai.set_fuse_gbl_config.called
    assert mock_ai.get_fuse_mode.called
    assert mock_ai.get_bw.called
    assert mock_ai.set_bw.called


def test_AiqosConfig_second_apply_reuses_baseline_fewer_capture_qos(tmp_path, monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0")
    mod, mock_ai = _load_ai_qos_tool()
    body = {
        "original_qos": {
            "0": {str(m): [m, 1, 0, 0, 0] for m in MASTER_IDS},
        },
        "original_sdma_mata": {"0": [0, 1, 1, 0]},
        "original_fuse": {"0": [1, 1, 0]},
        "printed_commands": [],
    }
    p = tmp_path / "s.json"
    p.write_text(json.dumps(body), encoding="utf-8")
    mock_ai.reset_mock()
    cfg = {
        "mode": "auto",
        "aiqos_priority": {
            "AIV_D2D": "high",
            "AIV_H2D": "high",
            "SDMA_D2D": "high",
            "SDMA_H2D": "low",
            "PCIEDMA_H2D": "high",
        },
    }
    with patch("sys.stdout", StringIO()):
        mod.AiqosConfig(cfg).set_qos(p)
    n_with_baseline = mock_ai.get_qos.call_count
    assert n_with_baseline == 4, "apply loop only: 4 masters, no capture get_qos"
    p.unlink()
    mock_ai.reset_mock()
    with patch("sys.stdout", StringIO()):
        mod.AiqosConfig(cfg).set_qos(p)
    n_cold = mock_ai.get_qos.call_count
    assert n_cold == 9, "1 dev cold: capture 4 + SDMA 1 + apply 4 = 9"


def test_AiqosConfig_merges_baseline_when_device_list_grows(tmp_path, monkeypatch):
    """Second apply with more NPU ids than the first-apply state must save baseline for new ids."""
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1")
    mod, _ = _load_ai_qos_tool()
    body = {
        "original_qos": {
            "0": {str(m): [m, 1, 0, 0, 0] for m in MASTER_IDS},
        },
        "original_sdma_mata": {"0": [0, 1, 1, 0]},
        "original_fuse": {"0": [1, 1, 0]},
        "printed_commands": [],
    }
    p = tmp_path / "state.json"
    p.write_text(json.dumps(body), encoding="utf-8")
    cfg = {
        "mode": "auto",
        "aiqos_priority": {
            "AIV_D2D": "high",
            "AIV_H2D": "high",
            "SDMA_D2D": "high",
            "SDMA_H2D": "low",
            "PCIEDMA_H2D": "high",
        },
    }
    with patch("sys.stdout", StringIO()):
        mod.AiqosConfig(cfg).set_qos(p)
    j = json.loads(p.read_text(encoding="utf-8"))
    assert "0" in j["original_qos"] and "1" in j["original_qos"]
    assert "0" in j["original_sdma_mata"] and "1" in j["original_sdma_mata"]
    assert "0" in j["original_fuse"] and "1" in j["original_fuse"]
