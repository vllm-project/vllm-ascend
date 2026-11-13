# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / ".github/workflows/scripts/collect_download.py"
_SPEC = importlib.util.spec_from_file_location("collect_download", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
collector = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(collector)


def test_multiple_paths_preserve_duplicate_filenames(tmp_path):
    sources = [tmp_path / "outputs", tmp_path / "logs"]
    for index, source in enumerate(sources):
        source.mkdir()
        (source / "result.txt").write_text(str(index))
        (source / ".env").write_text("secret")
        (source / "private.key").write_text("secret")
        (source / ".git").mkdir()
        (source / ".git/config").write_text("secret")
    destination = tmp_path / "artifact.tar.gz"
    collector.archive_path(sources, destination)
    with tarfile.open(destination) as archive:
        assert archive.getnames() == ["path-1/result.txt", "path-2/result.txt"]
        for index in range(2):
            assert archive.extractfile(f"path-{index + 1}/result.txt").read() == str(index).encode()


@pytest.mark.parametrize("path", ["/", "/etc/passwd", "../secret", "/root/.ssh/id_rsa"])
def test_unsafe_paths_rejected(path):
    with pytest.raises(ValueError):
        collector.parse_request(path)


def test_multiple_paths_for_single_node():
    request = '[{"node0":"outputs"},{"node0":"log_archives"}]'
    assert collector.parse_request(request) == {"node0": ["outputs", "log_archives"]}


def test_missing_path_and_symlink_rejected(tmp_path):
    with pytest.raises(ValueError):
        collector.validate_path(str(tmp_path / "missing"))
    target = tmp_path / "target"
    target.write_text("log")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(ValueError):
        collector.validate_path(str(link))


def test_cli_skips_missing_optional_path(tmp_path):
    existing = tmp_path / "logs"
    existing.mkdir()
    (existing / "result.txt").write_text("result")
    missing = tmp_path / "missing"
    destination = tmp_path / "artifact.tar.gz"
    request = json.dumps([{"node0": str(existing)}, {"node0": str(missing)}])
    command = [sys.executable, str(_SCRIPT), request, str(destination)]
    assert subprocess.run(command, capture_output=True).returncode != 0
    result = subprocess.run(command + ["--skip-missing"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Skipping missing optional path" in result.stdout
    with tarfile.open(destination) as archive:
        assert archive.getnames() == ["result.txt"]
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), str(missing), str(destination), "--skip-missing"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "no requested paths exist" in result.stderr
