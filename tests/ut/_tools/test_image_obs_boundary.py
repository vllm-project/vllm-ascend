# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from __future__ import annotations

from .build_cache_test_utils import REPO_ROOT


def test_image_pr_does_not_receive_obs_write_credentials_or_publish_l1():
    workflows = REPO_ROOT / ".github" / "workflows"
    caller = (workflows / "schedule_image_build_and_push.yaml").read_text(encoding="utf-8")
    image = (workflows / "_schedule_image_build.yaml").read_text(encoding="utf-8")

    for secret in ("HW_OBS_AK", "HW_OBS_SK"):
        assert f"{secret}: ${{{{ github.event_name != 'pull_request' && secrets.{secret} || '' }}}}" in caller

    for step_name in (
        "Checkout incremental-cache workflow helpers",
        "Export incremental cache from image",
        "Save incremental cache for image build",
    ):
        step = image.split(f"    - name: {step_name}\n", 1)[1].split("\n    - ", 1)[0]
        assert "github.event_name != 'pull_request'" in step

    assert "OBS_CACHE_WRITE_ACCESS_KEY_ID" not in image
    assert "OBS_CACHE_WRITE_SECRET_ACCESS_KEY" not in image
