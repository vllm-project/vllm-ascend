"""Collect independent pytest rounds and retain their benchmark artifacts."""

import json
import shutil
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any


def expand_test_rounds(configs: Sequence[Any]) -> list[tuple[Any, int, int]]:
    rounds = []
    for config in configs:
        count = config.extra_config.get("test_rounds", 1)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("test_rounds must be a positive integer")
        for number in range(1, count + 1):
            rounds.append((deepcopy(config) if count > 1 else config, number, count))
    return rounds


def round_id(parameters: tuple[Any, int, int]) -> str:
    config, number, count = parameters
    return config.name if count == 1 else f"{config.name}-round-{number:02d}-of-{count:02d}"


@contextmanager
def preserve_round(config: Any, number: int, count: int, work_dir: Path) -> Iterator[None]:
    if count == 1:
        yield
        return

    output_root = work_dir / "outputs"
    # One YAML file may contain several independently repeated cases.
    case_dir = output_root / "nightly_rounds" / config.name
    target = case_dir / f"round-{number:02d}"
    target.mkdir(parents=True, exist_ok=True)
    status = {"name": config.name, "round": number, "total_rounds": count, "outcome": "passed"}
    print(f"\nNightly round {number}/{count}: {config.name}", flush=True)
    try:
        yield
    except BaseException as error:
        status.update(outcome="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        # Keep the normal outputs/ upload location while removing AISBench's
        # previous work directory before the next independent server/test run.
        for source in output_root.iterdir():
            if source.name != "nightly_rounds":
                shutil.move(str(source), str(target / source.name))
        for name in ("benchmark_results", "log_archives"):
            source = work_dir / name
            if source.exists():
                shutil.move(str(source), str(target / name))
                if number == count:
                    # Existing workflow summary/upload steps still see the
                    # last round; the full history lives in outputs/.
                    shutil.copytree(target / name, source)
        for name in ("output_accuracy.txt", "output_performance.txt"):
            source = work_dir / name
            if source.exists():
                shutil.move(str(source), str(target / name))
        (target / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"Nightly round {number}/{count}: {status['outcome']}; artifacts: {target}", flush=True)
