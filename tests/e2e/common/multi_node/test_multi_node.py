"""Common pytest entry for internal and external DP multi-node cases."""

import pytest
import yaml

from tests.e2e.common.multi_node.dp_mode import resolve_config_path, resolve_dp_load_balancing


@pytest.mark.asyncio
async def test_multi_node() -> None:
    with resolve_config_path().open(encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)
    if isinstance(raw, dict) and "logits_regression" in raw:
        from tests.e2e.common.multi_node.glm53_logits.runner import run

        await run(raw)
        return
    mode = resolve_dp_load_balancing()

    # Import only the selected runtime. The two implementations have different
    # dependencies and startup paths, and remain unchanged during this migration.
    if mode == "external":
        from tests.e2e.common.multi_node.external_dp.test_external_dp import (
            test_external_dp,
        )

        test_external_dp()
        return

    from tests.e2e.common.multi_node.internal_dp.test_multi_node import (
        test_multi_node as test_internal_dp,
    )

    await test_internal_dp()
