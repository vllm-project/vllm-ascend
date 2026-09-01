# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json
import os
import shutil
import subprocess
from pathlib import Path

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = TOOLKIT_ROOT / "scripts" / "preflight.sh"
GIT = shutil.which("git")
assert GIT is not None


def run_preflight(
    tmp_path: Path, *args: str, token: str | None = None, author: bool = False
):
    global_config = tmp_path / "global.gitconfig"
    if author:
        subprocess.run(
            [GIT, "config", "--file", str(global_config), "user.name", "Toolkit Bot"],
            check=True,
        )
        subprocess.run(
            [
                GIT,
                "config",
                "--file",
                str(global_config),
                "user.email",
                "toolkit@example.com",
            ],
            check=True,
        )

    env = os.environ.copy()
    env.pop("GITCODE_TOKEN", None)
    env["GIT_CONFIG_GLOBAL"] = str(global_config)
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    if token is not None:
        env["GITCODE_TOKEN"] = token

    executable_preflight = tmp_path / "preflight.sh"
    shutil.copyfile(PREFLIGHT, executable_preflight)
    executable_preflight.chmod(0o700)
    result = subprocess.run(
        [str(executable_preflight), *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    return result, json.loads(result.stdout)


def test_preflight_keeps_legacy_full_check_contract(tmp_path: Path):
    result, report = run_preflight(
        tmp_path, "--skip-git-author", token="compatibility-token"
    )

    assert result.returncode == 0
    assert set(report) == {"results", "summary"}
    assert [item["item"] for item in report["results"]] == [
        "token",
        "git",
        "curl",
        "python3",
        "tmp",
        "git_author",
    ]
    assert report["results"][-1] == {
        "item": "git_author",
        "status": "skip",
        "detail": "--skip-git-author",
    }
    assert report["summary"] == {"pass": 6, "fail": 0, "total": 6}


def test_preflight_reports_missing_token_and_global_author(tmp_path: Path):
    result, report = run_preflight(tmp_path)

    assert result.returncode != 0
    failures = {
        item["item"] for item in report["results"] if item["status"] == "fail"
    }
    assert failures == {"token", "git_author"}
    assert report["summary"] == {"pass": 4, "fail": 2, "total": 6}


def test_preflight_uses_global_author_and_redacts_token(tmp_path: Path):
    secret = "token-value-must-not-appear"
    result, report = run_preflight(tmp_path, token=secret, author=True)

    assert result.returncode == 0
    assert secret not in result.stdout
    author = next(item for item in report["results"] if item["item"] == "git_author")
    assert author == {
        "item": "git_author",
        "status": "pass",
        "detail": "Toolkit Bot <toolkit@example.com>",
        "source": "global",
    }
