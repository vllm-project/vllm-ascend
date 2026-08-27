# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "trigger_pr_pipeline.sh"
REFERENCE = Path(__file__).resolve().parents[1] / "references" / "authorization-contract.md"


def run_trigger(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("GITCODE_TOKEN", None)
    return subprocess.run(
        [str(SCRIPT), "--repo", "cann/ops-math", "--pr", "1", *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_pipeline_trigger_keeps_legacy_cli_compatible() -> None:
    result = run_trigger()

    assert result.returncode == 1
    assert "GITCODE_TOKEN not set" in result.stdout


def test_authorization_reference_stays_platform_generic() -> None:
    contract = REFERENCE.read_text(encoding="utf-8")

    assert "写前精确确认" in contract
    assert "证据与失效" in contract
    assert "写后回查" in contract
    assert "不定义具体业务的批次、Issue 清单或聚合执行模型" in contract

    business_fragments = (
        "approved_batch",
        "issue_iids",
        "auto-close-stale",
        "followup_state.py",
    )
    assert not any(fragment in contract for fragment in business_fragments)
