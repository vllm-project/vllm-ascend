# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLKIT_ROOT = REPO_ROOT / "infra" / "gitcode-toolkit"


def read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_existing_consumers_keep_step0_full_preflight_contract():
    toolkit_skill = read("infra/gitcode-toolkit/SKILL.md")
    env_contract = read("infra/gitcode-toolkit/references/env-check.md")

    assert "| Step 0 | 环境预检（token / git / curl / /tmp / git-author）" in toolkit_skill
    assert "`bash scripts/preflight.sh`" in toolkit_skill
    assert "执行任何业务步骤之前" in env_contract
    assert "bash scripts/preflight.sh [--skip-git-author]" in env_contract

    for consumer in ("gitcode-pr-handler", "gitcode-issue-gen"):
        skill = read(f"infra/{consumer}/SKILL.md")
        assert "### Step 0：环境预检（必经）" in skill
        assert "gitcode-toolkit/references/env-check.md" in skill
        assert "token / git / curl / python3 / /tmp 任一缺失" in skill


def test_toolkit_has_no_issue_handler_private_runtime_contract():
    forbidden = (
        "ISSUE_HANDLER_TMP_DIR",
        ".cannbot/gitcode-issue-handler",
        "policy_query",
        "no_attention",
        "--checks api",
        "--checks git",
        "--checks tmp",
        "--checks author",
    )
    paths = [
        TOOLKIT_ROOT / "SKILL.md",
        TOOLKIT_ROOT / "references" / "env-check.md",
        TOOLKIT_ROOT / "scripts" / "preflight.sh",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    for term in forbidden:
        assert term not in combined
