import csv
from pathlib import Path

import pytest

from tests.e2e.nightly.scripts.update_good_table import (
    HEADER,
    build_parser,
    load_rows,
    resolve_test_path,
    update_table,
)


def _row(*, name: str, path: str, soc: str, scene: str, commit: str) -> dict[str, str]:
    return {
        "name": name,
        "yaml/path": path,
        "link": "https://example.invalid/run",
        "status": "success",
        "vLLM Git information": "vllm",
        "vLLM-Ascend Git information": commit,
        "soc": soc,
        "scene": scene,
        "time": "2026-07-29 10:00:00 +08:00",
    }


def _base_args(*, soc: str = "a2", scene: str = "single_node") -> list[str]:
    return [
        "--cache-csv",
        "good_table.csv",
        "--test-name",
        "m",
        "--test-path",
        "cases/model.yaml",
        "--config-base-path",
        "cases",
        "--scene",
        scene,
        "--soc",
        soc,
        "--run-link",
        "https://example.invalid/run",
    ]


def test_parser_requires_soc_and_rejects_placeholders():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([arg for arg in _base_args() if arg != "a2" and arg != "--soc"])
    for bad_soc in ("", "unknown", "UNKNOWN", "none", "null"):
        with pytest.raises(SystemExit):
            parser.parse_args(_base_args(soc=bad_soc))


def test_parser_requires_scene_and_rejects_invalid_values():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([arg for arg in _base_args() if arg != "single_node" and arg != "--scene"])
    for bad_scene in ("", "double_node", "multi-card"):
        with pytest.raises(SystemExit):
            parser.parse_args(_base_args(scene=bad_scene))


def test_parser_accepts_valid_dimension_combinations():
    parser = build_parser()
    a2 = parser.parse_args(_base_args(soc="a2", scene="single_node"))
    assert a2.soc == "a2"
    assert a2.scene == "single_node"
    a3 = parser.parse_args(_base_args(soc="a3", scene="multi_node"))
    assert a3.soc == "a3"
    assert a3.scene == "multi_node"


def test_update_table_rejects_incomplete_composite_key(tmp_path: Path):
    table = tmp_path / "good_table.csv"
    with pytest.raises(ValueError, match="invalid soc"):
        update_table(str(table), _row(name="m", path="cases/model.yaml", soc="", scene="single_node", commit="a2"))
    with pytest.raises(ValueError, match="invalid soc"):
        update_table(
            str(table),
            _row(name="m", path="cases/model.yaml", soc="unknown", scene="single_node", commit="a2"),
        )
    with pytest.raises(ValueError, match="invalid scene"):
        update_table(str(table), _row(name="m", path="cases/model.yaml", soc="a2", scene="", commit="a2"))
    with pytest.raises(ValueError, match="yaml/path"):
        update_table(str(table), _row(name="m", path="", soc="a2", scene="single_node", commit="a2"))
    assert not table.exists()


def test_update_replaces_only_same_composite_key(tmp_path: Path):
    table = tmp_path / "good_table.csv"

    update_table(str(table), _row(name="shared", path="cases/model.yaml", soc="a2", scene="single_node", commit="a2"))
    update_table(str(table), _row(name="shared", path="cases/model.yaml", soc="a3", scene="single_node", commit="a3"))
    update_table(
        str(table),
        _row(name="renamed", path="cases/model.yaml", soc="a2", scene="single_node", commit="a2-new"),
    )

    rows = load_rows(str(table))
    assert len(rows) == 2
    assert {(row["soc"], row["vLLM-Ascend Git information"]) for row in rows} == {
        ("a2", "a2-new"),
        ("a3", "a3"),
    }


def test_update_migrates_matching_legacy_row(tmp_path: Path):
    table = tmp_path / "good_table.csv"
    table.write_text(
        "name,yaml/path,link,status,vLLM Git information,VLLM-Ascend Git information,time\n"
        "legacy,cases/model.yaml,old,success,vllm-old,asc-old,2026-07-20 10:00:00 +08:00\n",
        encoding="utf-8",
    )

    update_table(
        str(table),
        _row(name="legacy", path="cases/model.yaml", soc="a2", scene="single_node", commit="asc-new"),
    )

    with table.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == HEADER
    assert len(rows) == 1
    assert rows[0]["soc"] == "a2"
    assert rows[0]["scene"] == "single_node"
    assert rows[0]["time"] == "2026-07-29 10:00:00 +08:00"


def test_update_creates_table_with_header_and_reports_new(tmp_path: Path):
    table = tmp_path / "good_table.csv"

    is_new = update_table(
        str(table), _row(name="m", path="cases/model.yaml", soc="a2", scene="single_node", commit="a2")
    )

    assert is_new is True
    with table.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == HEADER
    assert len(rows) == 1
    assert rows[0]["name"] == "m"
    assert rows[0]["soc"] == "a2"
    assert rows[0]["scene"] == "single_node"


def test_update_reports_false_when_table_exists(tmp_path: Path):
    table = tmp_path / "good_table.csv"
    row = _row(name="m", path="cases/model.yaml", soc="a2", scene="single_node", commit="a2")

    assert update_table(str(table), row) is True
    assert update_table(str(table), row) is False


def test_update_keeps_rows_for_different_scene(tmp_path: Path):
    table = tmp_path / "good_table.csv"

    update_table(str(table), _row(name="m", path="cases/model.yaml", soc="a2", scene="single_node", commit="s"))
    update_table(str(table), _row(name="m", path="cases/model.yaml", soc="a2", scene="multi_node", commit="m"))

    rows = load_rows(str(table))
    assert len(rows) == 2
    assert {(row["scene"], row["vLLM-Ascend Git information"]) for row in rows} == {
        ("single_node", "s"),
        ("multi_node", "m"),
    }


def test_update_preserves_unrelated_rows(tmp_path: Path):
    table = tmp_path / "good_table.csv"
    table.write_text(
        "name,yaml/path,link,status,vLLM Git information,VLLM-Ascend Git information,soc,scene,time\n"
        "other,cases/other.yaml,other,success,vllm-other,asc-other,a2,single_node,2026-07-20 10:00:00 +08:00\n",
        encoding="utf-8",
    )

    update_table(str(table), _row(name="m", path="cases/model.yaml", soc="a2", scene="single_node", commit="asc-m"))

    rows = load_rows(str(table))
    assert len(rows) == 2
    assert {row["name"] for row in rows} == {"m", "other"}


def test_update_normalises_path_separators(tmp_path: Path):
    table = tmp_path / "good_table.csv"
    table.write_text(
        "name,yaml/path,link,status,vLLM Git information,VLLM-Ascend Git information,soc,scene,time\n"
        "m,cases\\model.yaml/,old,success,vllm-old,asc-old,a2,single_node,2026-07-20 10:00:00 +08:00\n",
        encoding="utf-8",
    )

    update_table(str(table), _row(name="m", path="cases/model.yaml", soc="a2", scene="single_node", commit="asc-new"))

    rows = load_rows(str(table))
    assert len(rows) == 1
    assert rows[0]["yaml/path"] == "cases/model.yaml"
    assert rows[0]["vLLM-Ascend Git information"] == "asc-new"


def test_load_rows_returns_empty_for_missing_file(tmp_path: Path):
    assert load_rows(str(tmp_path / "no_such_table.csv")) == []


def test_resolve_test_path_joins_bare_filename_with_config_base():
    assert (
        resolve_test_path("model.yaml", "tests/e2e/weekly/single_node/configs", "single_node")
        == "tests/e2e/weekly/single_node/configs/model.yaml"
    )


def test_resolve_test_path_keeps_explicit_directory_unchanged():
    full = "tests/e2e/nightly/multi_node/external_dp/config/model.yaml"
    assert resolve_test_path(full, "ignored", "multi_node") == full
