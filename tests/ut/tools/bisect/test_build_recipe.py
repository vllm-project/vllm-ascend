from pathlib import Path

import pytest

from tools.bisect import git_ops
from tools.bisect.build_recipe import ROOT_DOCKERFILES, recipe_at_commit


@pytest.mark.parametrize(
    ("soc", "target"),
    [("a2", "a2"), ("910b", "a2"), ("a3", "a3"), ("310p", "310p"), ("a5", "a5"), ("950", "a5")],
)
def test_recipe_selects_hardware_dockerfiles(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, soc: str, target: str):
    files = {
        ROOT_DOCKERFILES[target]: (
            f"FROM cann:{target}\nARG VLLM_TAG=v0.32.0\n"
            "ARG PYTORCH_INDEX_URL=https://pytorch.example/cpu\n"
            "ARG ASCEND_INDEX_URL=https://ascend.example/pypi\n"
        ),
        f".github/workflows/dockerfiles/Dockerfile.nightly.{target}": (
            "ARG PIP_EXTRA_INDEX_URL=https://nightly.example/pypi\n"
            "ARG AIS_BENCH_TAG=v3\nARG AIS_BENCH_URL=https://example.com/benchmark.git\n"
        ),
        "requirements.txt": "torch-npu==2.10.0.post7\n",
        "pyproject.toml": "[build-system]\nrequires = []\n",
        "requirements-nightly.txt": "pytest\n",
    }
    monkeypatch.setattr(git_ops, "file_at_commit", lambda repo, commit, path: files.get(path))

    recipe = recipe_at_commit(tmp_path, "a" * 40, soc)

    assert recipe.soc == target
    assert recipe.root_dockerfile == ROOT_DOCKERFILES[target]
    assert recipe.nightly_dockerfile.endswith(f"Dockerfile.nightly.{target}")
    assert recipe.cann_image == f"cann:{target}"
    assert recipe.requirements == "torch-npu==2.10.0.post7\n"
    assert recipe.requirements_nightly == "pytest\n"


def test_recipe_rejects_missing_build_input(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(git_ops, "file_at_commit", lambda repo, commit, path: None)

    with pytest.raises(ValueError, match="build recipe file missing"):
        recipe_at_commit(tmp_path, "a" * 40, "a3")


def test_recipe_selects_openeuler_root_dockerfile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    files = {
        "Dockerfile.a3.openEuler": "FROM cann:openeuler\nARG VLLM_TAG=v0.32.0\n",
        ".github/workflows/dockerfiles/Dockerfile.nightly.a3": "ARG AIS_BENCH_TAG=v3\n",
        "requirements.txt": "torch-npu==2.10.0.post7\n",
        "pyproject.toml": "[build-system]\nrequires = []\n",
        "requirements-nightly.txt": "pytest\n",
    }
    monkeypatch.setattr(git_ops, "file_at_commit", lambda repo, commit, path: files.get(path))

    recipe = recipe_at_commit(tmp_path, "a" * 40, "a3", os_variant="ID=openEuler")

    assert recipe.root_dockerfile == "Dockerfile.a3.openEuler"
