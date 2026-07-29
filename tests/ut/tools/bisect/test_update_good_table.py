import csv
from pathlib import Path

from tests.e2e.nightly.scripts.update_good_table import HEADER, load_rows, update_table


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
