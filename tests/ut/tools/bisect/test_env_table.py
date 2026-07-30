from pathlib import Path
from textwrap import dedent

from tools.bisect.config import BisectInput, Candidate
from tools.bisect.env_table import EnvTable

HEADER = (
    "name,yaml/path,link,status,vLLM Git information,vLLM-Ascend Git information,CANN Version,torch-npu Version,time"
)


def _candidate(commit: str) -> Candidate:
    return Candidate(commit=commit, pr_number=None, subject=commit)


def test_env_table_prefers_exact_commit_row(tmp_path: Path, monkeypatch):
    table = tmp_path / "env_table.csv"
    table.write_text(
        dedent(
            f"""
            {HEADER}
            case,cases/case.yaml,old,success,vllm-old,aaa111,9.0.0,2.5.0,2026-01-01 01:00:00 +08:00
            case,cases/case.yaml,new,failure,vllm-new,bbb222,9.0.1,2.6.0,2026-01-02 01:00:00 +08:00
            """
        ).lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setattr("tools.bisect.git_ops.is_ancestor", lambda _repo, ancestor, target: ancestor == target)

    resolved = EnvTable(str(table)).resolve_for_commits(
        tmp_path,
        BisectInput(scene="single_node", config_yaml="case.yaml", bad_commit="bbb222", name="case"),
        [_candidate("bbb222")],
    )

    assert resolved["bbb222"].vllm_ref == "vllm-new"
    assert resolved["bbb222"].cann_version == "9.0.1"
    assert resolved["bbb222"].torch_npu_version == "2.6.0"


def test_env_table_uses_closest_preceding_status_row(tmp_path: Path, monkeypatch):
    table = tmp_path / "env_table.csv"
    table.write_text(
        dedent(
            f"""
            {HEADER}
            case,cases/case.yaml,old,success,vllm-old,aaa,9.0.0,2.5.0,2026-01-01 01:00:00 +08:00
            case,cases/case.yaml,new,success,vllm-new,ccc,9.0.1,2.6.0,2026-01-02 01:00:00 +08:00
            """
        ).lstrip(),
        encoding="utf-8",
    )
    ancestry = {
        ("aaa", "bbb"): True,
        ("ccc", "bbb"): False,
        ("aaa", "ccc"): True,
    }

    def is_ancestor(_repo: Path, ancestor: str, target: str) -> bool:
        return ancestry[(ancestor, target)]

    monkeypatch.setattr("tools.bisect.git_ops.is_ancestor", is_ancestor)

    resolved = EnvTable(str(table)).resolve_for_commits(
        tmp_path,
        BisectInput(scene="single_node", config_yaml="case.yaml", bad_commit="bbb", name="case"),
        [_candidate("bbb")],
    )

    assert resolved["bbb"].vllm_ref == "vllm-old"
