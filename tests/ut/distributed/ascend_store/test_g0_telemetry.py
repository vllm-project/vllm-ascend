import json

import pytest

from vllm_ascend import g0_telemetry


def test_emit_is_disabled_without_an_output_path(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("VLLM_G0_TELEMETRY_PATH", raising=False)
    g0_telemetry.emit("disabled", value=1)
    assert list(tmp_path.iterdir()) == []


def test_emit_and_bytes_in_flight_accounting(tmp_path, monkeypatch) -> None:
    output = tmp_path / "telemetry.jsonl"
    monkeypatch.setenv("VLLM_G0_TELEMETRY_PATH", str(output))
    monkeypatch.setattr(g0_telemetry, "_BYTES_IN_FLIGHT", 0)

    assert g0_telemetry.adjust_bytes_in_flight(1024) == 1024
    g0_telemetry.emit("transfer_started", request_id="r0", bytes_in_flight=1024)
    assert g0_telemetry.adjust_bytes_in_flight(-1024) == 0

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["event"] == "transfer_started"
    assert row["request_id"] == "r0"
    assert row["bytes_in_flight"] == 1024
    assert row["ts_ns"] > 0
    assert row["wall_ts_ns"] > 0


def test_negative_bytes_in_flight_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(g0_telemetry, "_BYTES_IN_FLIGHT", 0)
    with pytest.raises(RuntimeError, match="became negative"):
        g0_telemetry.adjust_bytes_in_flight(-1)
