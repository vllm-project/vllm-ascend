import pytest

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.pull_request.four_card.graph_accuracy_probe import run_static_golden_deterministic_probe
from tests.e2e.pull_request.four_card.test_graph_mode import CASE_DS_ACLGRAPH, CASE_DS_ACLGRAPH_ENPU


@wait_until_npu_memory_free(0.7)
@pytest.mark.parametrize(
    "cur_case",
    [
        pytest.param(CASE_DS_ACLGRAPH, id="deepseek-w8a8"),
        pytest.param(CASE_DS_ACLGRAPH_ENPU, id="deepseek-w8a8-enpu"),
    ],
)
def test_graph_against_static_golden_with_deterministic_settings(
    cur_case: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_static_golden_deterministic_probe(cur_case, monkeypatch)
