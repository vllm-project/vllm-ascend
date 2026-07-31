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


def test_lookup_uses_all_supplied_case_dimensions(tmp_path: Path):
    table_path = tmp_path / "good_table.csv"
    table_path.write_text(
        dedent(
            """
            name,yaml/path,link,status,vLLM Git information,VLLM-Ascend Git information,soc,scene,time
            shared,cases/a2.yaml,a2,success,vllm-a2,asc-a2,a2,single_node,2026-01-01 01:00:00 +0800
            shared,cases/a3.yaml,a3,success,vllm-a3,asc-a3,a3,single_node,2026-01-02 01:00:00 +0800
            """
        ).lstrip(),
        encoding="utf-8",
    )

    entry = GoodTable(str(table_path)).lookup_last_good(
        name="shared",
        config_yaml="a2.yaml",
        soc="a2",
        scene="single_node",
    )

    assert entry is not None
    assert entry.vllm_ascend_commit == "asc-a2"


def test_lookup_legacy_row_without_dimensions_remains_compatible(tmp_path: Path):
    table_path = tmp_path / "good_table.csv"
    _write_table(table_path)

    entry = GoodTable(str(table_path)).lookup_last_good(
        name="llama",
        config_yaml="llama.yaml",
        soc="a2",
        scene="single_node",
    )

    assert entry is not None
    assert entry.vllm_ascend_commit == "asc-new"


def test_lookup_filters_by_soc(tmp_path: Path):
    table_path = tmp_path / "good_table.csv"
    table_path.write_text(
        dedent(
            """
            name,yaml/path,link,status,vLLM Git information,VLLM-Ascend Git information,soc,scene,time
            shared,cases/model.yaml,a2,success,vllm-a2,asc-a2,a2,single_node,2026-01-02 01:00:00 +0800
            shared,cases/model.yaml,a3,success,vllm-a3,asc-a3,a3,single_node,2026-01-01 01:00:00 +0800
            """
        ).lstrip(),
        encoding="utf-8",
    )
    table = GoodTable(str(table_path))

    assert table.lookup_last_good(
        name="shared", config_yaml="model.yaml", soc="a2", scene="single_node"
    ).vllm_ascend_commit == "asc-a2"
    assert table.lookup_last_good(
        name="shared", config_yaml="model.yaml", soc="a3", scene="single_node"
    ).vllm_ascend_commit == "asc-a3"
    assert (
        table.lookup_last_good(name="shared", config_yaml="model.yaml", soc="a4", scene="single_node")
        is None
    )


def test_lookup_filters_by_scene(tmp_path: Path):
    table_path = tmp_path / "good_table.csv"
    table_path.write_text(
        dedent(
            """
            name,yaml/path,link,status,vLLM Git information,VLLM-Ascend Git information,soc,scene,time
            shared,cases/model.yaml,single,success,vllm-s,asc-s,a2,single_node,2026-01-02 01:00:00 +0800
            shared,cases/model.yaml,multi,success,vllm-m,asc-m,a2,multi_node,2026-01-01 01:00:00 +0800
            """
        ).lstrip(),
        encoding="utf-8",
    )
    table = GoodTable(str(table_path))

    assert table.lookup_last_good(
        name="shared", config_yaml="model.yaml", soc="a2", scene="single_node"
    ).vllm_ascend_commit == "asc-s"
    assert table.lookup_last_good(
        name="shared", config_yaml="model.yaml", soc="a2", scene="multi_node"
    ).vllm_ascend_commit == "asc-m"
    assert (
        table.lookup_last_good(name="shared", config_yaml="model.yaml", soc="a2", scene="double_node")
        is None
    )


def test_lookup_without_dimensions_matches_new_schema_rows(tmp_path: Path):
    table_path = tmp_path / "good_table.csv"
    table_path.write_text(
        dedent(
            """
            name,yaml/path,link,status,vLLM Git information,VLLM-Ascend Git information,soc,scene,time
            shared,cases/a2.yaml,a2,success,vllm-a2,asc-a2,a2,single_node,2026-01-01 01:00:00 +0800
            shared,cases/a3.yaml,a3,success,vllm-a3,asc-a3,a3,single_node,2026-01-02 01:00:00 +0800
            """
        ).lstrip(),
        encoding="utf-8",
    )
    table = GoodTable(str(table_path))

    entry = table.lookup_last_good(name="shared")
    assert entry is not None
    assert entry.vllm_ascend_commit == "asc-a3"

    entry = table.lookup_last_good(name="shared", config_yaml="a2.yaml")
    assert entry is not None
    assert entry.vllm_ascend_commit == "asc-a2"


def test_lookup_requires_all_supplied_dimensions_to_match(tmp_path: Path):
    table_path = tmp_path / "good_table.csv"
    table_path.write_text(
        dedent(
            """
            name,yaml/path,link,status,vLLM Git information,VLLM-Ascend Git information,soc,scene,time
            shared,cases/model.yaml,ok,success,vllm-ok,asc-ok,a2,single_node,2026-01-01 01:00:00 +0800
            """
        ).lstrip(),
        encoding="utf-8",
    )
    table = GoodTable(str(table_path))

    assert table.lookup_last_good(name="shared", config_yaml="other.yaml") is None
    assert table.lookup_last_good(name="other", config_yaml="model.yaml") is None


def test_lookup_skips_success_rows_without_commit(tmp_path: Path):
    table_path = tmp_path / "good_table.csv"
    table_path.write_text(
        dedent(
            """
            name,yaml/path,link,status,vLLM Git information,VLLM-Ascend Git information,soc,scene,time
            m,cases/m.yaml,new-empty,success,vllm-new,,a2,single_node,2026-01-02 01:00:00 +0800
            m,cases/m.yaml,old,success,vllm-old,asc-old,a2,single_node,2026-01-01 01:00:00 +0800
            """
        ).lstrip(),
        encoding="utf-8",
    )

    entry = GoodTable(str(table_path)).lookup_last_good(
        name="m", config_yaml="m.yaml", soc="a2", scene="single_node"
    )

    assert entry is not None
    assert entry.vllm_ascend_commit == "asc-old"


def test_lookup_accepts_status_spelling_variants(tmp_path: Path):
    table_path = tmp_path / "good_table.csv"
    table_path.write_text(
        dedent(
            """
            name,yaml/path,link,status,vLLM Git information,VLLM-Ascend Git information,soc,scene,time
            m,cases/m.yaml,p1,PASS,vllm-p,asc-p,a2,single_node,2026-01-02 01:00:00 +0800
            m,cases/m.yaml,p2,ok,vllm-q,asc-q,a2,single_node,2026-01-01 01:00:00 +0800
            """
        ).lstrip(),
        encoding="utf-8",
    )

    entry = GoodTable(str(table_path)).lookup_last_good(
        name="m", config_yaml="m.yaml", soc="a2", scene="single_node"
    )

    assert entry is not None
    assert entry.vllm_ascend_commit == "asc-p"


def test_lookup_tolerates_mixed_time_formats(tmp_path: Path):
    table_path = tmp_path / "good_table.csv"
    table_path.write_text(
        dedent(
            """
            name,yaml/path,link,status,vLLM Git information,VLLM-Ascend Git information,soc,scene,time
            m,cases/m.yaml,naive,success,vllm-n,asc-n,a2,single_node,2026-01-01 10:00:00
            m,cases/m.yaml,aware,success,vllm-a,asc-a,a2,single_node,2026-01-02 10:00:00 +08:00
            m,cases/m.yaml,bad,success,vllm-b,asc-b,a2,single_node,garbage
            """
        ).lstrip(),
        encoding="utf-8",
    )

    entry = GoodTable(str(table_path)).lookup_last_good(
        name="m", config_yaml="m.yaml", soc="a2", scene="single_node"
    )

    assert entry is not None
    assert entry.vllm_ascend_commit == "asc-a"


def test_lookup_missing_table_returns_none(tmp_path: Path):
    assert GoodTable(str(tmp_path / "no_such_good_table.csv")).lookup_last_good(name="llama") is None


def test_lookup_last_good_returns_none_for_missing_or_failed_case(tmp_path: Path):
    table_path = tmp_path / "good_table.csv"
    _write_table(table_path)

    assert GoodTable(str(table_path)).lookup_last_good(name="missing") is None


def test_norm_detects_surplus_csv_columns():
    raw_row: dict[str | None, object] = {" Name ": " llama ", "status": " success ", None: ["extra", "columns"]}

    row, had_surplus = _norm(raw_row)

    assert had_surplus is True
    assert row == {"name": "llama", "status": "success"}
