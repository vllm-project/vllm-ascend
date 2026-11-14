# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name != "posix", reason="Requires POSIX permissions")


@pytest.fixture
def permission_tree(tmp_path):
    repo = tmp_path / "repo"
    tools = repo / "tools"
    tools.mkdir(parents=True)
    script = tools / "fix_runtime_permissions.sh"
    source = Path(__file__).resolve().parents[3] / "tools" / script.name
    shutil.copyfile(source, script)
    return repo, tmp_path / "cann", script


def test_runtime_files_readable_but_not_writable(permission_tree):
    repo, cann, script = permission_tree
    paths = [
        repo / "vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib/libcust_opapi.so",
        cann / "opp/vendors/batch_invariant/op_api/lib/libcust_opapi.so",
        cann / "opp/vendors/cust/op_impl/aicpu/hybm/kernel/kernel.tar.gz",
        cann / "tools/msmemscope/lib64/libmsmemscope.so",
        cann / "share/info/ascendnpu-ir/bin/set_env.sh",
    ]
    for path, mode in zip(paths, (0o750, 0o750, 0o640, 0o750, 0o751)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("runtime file\n")
        path.chmod(mode)
    for root in (repo / "vllm_ascend/_cann_ops_custom/vendors", cann / "opp/vendors", cann / "tools/msmemscope"):
        root.chmod(0o750)
        for path in root.rglob("*"):
            if path.is_dir():
                path.chmod(0o770)
    owners = {path: (path.stat().st_uid, path.stat().st_gid) for path in paths}

    for _ in range(2):
        subprocess.run(["bash", str(script), str(cann)], check=True)
        for path in paths:
            mode = stat.S_IMODE(path.stat().st_mode)
            assert mode & 0o044 == 0o044
            assert mode & 0o022 == 0
            assert (path.stat().st_uid, path.stat().st_gid) == owners[path]
        assert not paths[2].stat().st_mode & 0o111
        for root in (repo / "vllm_ascend/_cann_ops_custom/vendors", cann / "opp/vendors", cann / "tools/msmemscope"):
            for path in [root, *root.rglob("*")]:
                if path.is_dir():
                    mode = stat.S_IMODE(path.stat().st_mode)
                    assert mode & 0o055 == 0o055
                    assert mode & 0o022 == 0


def test_optional_paths_and_unrelated_files_unchanged(permission_tree):
    _, cann, script = permission_tree
    subprocess.run(["bash", str(script), str(cann)], check=True)
    private_file = cann / "script/uninstall.sh"
    private_file.parent.mkdir(parents=True)
    private_file.write_text("private\n")
    private_file.chmod(0o700)
    vendor = cann / "opp/vendors/custom"
    vendor.mkdir(parents=True)
    (vendor / "outside").symlink_to(private_file)
    subprocess.run(["bash", str(script), str(cann)], check=True)
    assert stat.S_IMODE(private_file.stat().st_mode) == 0o700
