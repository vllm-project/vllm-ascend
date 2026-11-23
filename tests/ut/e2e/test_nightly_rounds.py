import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HELPER = Path(__file__).resolve().parents[3] / "tests/e2e/common/single_node/nightly_rounds.py"
SPEC = importlib.util.spec_from_file_location("nightly_rounds", HELPER)
assert SPEC is not None and SPEC.loader is not None
rounds = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rounds)


@pytest.mark.parametrize("invalid", [0, -1, True, "5", 1.5])
def test_invalid_round_count(invalid):
    config = SimpleNamespace(name="case", extra_config={"test_rounds": invalid})
    with pytest.raises(ValueError, match="positive integer"):
        rounds.expand_test_rounds([config])


def test_default_and_independent_round_configs():
    config = SimpleNamespace(name="case", extra_config={})
    assert rounds.expand_test_rounds([config]) == [(config, 1, 1)]
    assert rounds.round_id((config, 1, 1)) == "case"
    config.extra_config["test_rounds"] = 5
    expanded = rounds.expand_test_rounds([config])
    assert len(expanded) == 5
    expanded[0][0].extra_config["mutated"] = True
    assert "mutated" not in expanded[1][0].extra_config
    assert rounds.round_id(expanded[-1]) == "case-round-05-of-05"


def test_failed_round_continues_and_preserves_fresh_artifacts(tmp_path):
    # Exercise pytest's real failure/continuation semantics in a CPU-only child.
    test_file = tmp_path / "test_repeated.py"
    test_file.write_text(
        f"""import importlib.util, json
from pathlib import Path
from types import SimpleNamespace
import pytest
spec = importlib.util.spec_from_file_location('rounds', {str(HELPER)!r})
rounds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rounds)
config = SimpleNamespace(name='case', extra_config={{'test_rounds': 5}})
@pytest.mark.parametrize('config,number,count', rounds.expand_test_rounds([config]))
def test_round(config, number, count):
    with rounds.preserve_round(config, number, count, Path.cwd()):
        assert not Path('outputs/default').exists()
        assert not Path('benchmark_results').exists()
        if number == 2:
            raise AssertionError('intentional round 2 failure')
        Path('outputs/default').mkdir(parents=True)
        Path('outputs/default/prediction.json').write_text(json.dumps({{'round': number}}))
        Path('benchmark_results').mkdir()
        Path('benchmark_results/case.json').write_text(json.dumps({{'round': number}}))
""",
        encoding="utf-8",
    )
    process = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(test_file)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 1, process.stdout + process.stderr
    assert "1 failed, 4 passed" in process.stdout
    root = tmp_path / "outputs/nightly_rounds/case"
    statuses = [json.loads((root / f"round-{number:02d}/status.json").read_text()) for number in range(1, 6)]
    assert [status["outcome"] for status in statuses] == ["passed", "failed", "passed", "passed", "passed"]
    assert not (root / "round-02/benchmark_results").exists()
    for number in (1, 3, 4, 5):
        assert json.loads((root / f"round-{number:02d}/benchmark_results/case.json").read_text()) == {"round": number}
    assert json.loads((tmp_path / "benchmark_results/case.json").read_text()) == {"round": 5}
