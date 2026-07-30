from pathlib import Path
from textwrap import dedent

from tools.bisect.good_table import GoodTable, _norm


def _write_table(path: Path) -> None:
    path.write_text(
        dedent(
            """
            name,yaml/path,link,status,vLLM Git information,VLLM-Ascend Git information,time
            llama,cases/llama.yaml,old,success,vllm-old,asc-old,2026-01-01 01:00:00 +0800
            llama,cases/llama.yaml,failed,failure,vllm-failed,asc-failed,2026-01-03 01:00:00 +0800
            llama,cases/llama.yaml,new,success,vllm-new,asc-new,2026-01-02 01:00:00 +0800
            other,cases/other.yaml,other,success,vllm-other,asc-other,2026-01-04 01:00:00 +0800
            """
        ).lstrip(),
        encoding="utf-8",
    )


def test_lookup_last_good_by_name_uses_latest_success(tmp_path: Path):
    table_path = tmp_path / "good_table.csv"
    _write_table(table_path)

    entry = GoodTable(str(table_path)).lookup_last_good(name="llama")

    assert entry is not None
    assert entry.link == "new"
    assert entry.vllm_commit == "vllm-new"
    assert entry.vllm_ascend_commit == "asc-new"


def test_lookup_last_good_by_yaml_basename(tmp_path: Path):
    table_path = tmp_path / "good_table.csv"
    _write_table(table_path)

    entry = GoodTable(str(table_path)).lookup_last_good(config_yaml="llama.yaml")

    assert entry is not None
    assert entry.name == "llama"
    assert entry.vllm_ascend_commit == "asc-new"


def test_lookup_last_good_returns_none_for_missing_or_failed_case(tmp_path: Path):
    table_path = tmp_path / "good_table.csv"
    _write_table(table_path)

    assert GoodTable(str(table_path)).lookup_last_good(name="missing") is None


def test_norm_detects_surplus_csv_columns():
    raw_row: dict[str | None, object] = {" Name ": " llama ", "status": " success ", None: ["extra", "columns"]}

    row, had_surplus = _norm(raw_row)

    assert had_surplus is True
    assert row == {"name": "llama", "status": "success"}
