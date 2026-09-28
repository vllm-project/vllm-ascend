"""Common pytest entry for internal and external DP multi-node cases."""

import pytest

from tests.e2e.nightly.multi_node.scripts.dp_mode import resolve_dp_load_balancing


@pytest.mark.asyncio
async def test_multi_node() -> None:
    mode = resolve_dp_load_balancing()

    # Import only the selected runtime. The two implementations have different
    # dependencies and startup paths, and remain unchanged during this migration.
    if mode == "external":
        from tests.e2e.nightly.multi_node.external_dp.scripts.test_external_dp import (
            test_external_dp,
        )

        test_external_dp()
        return

    from tests.e2e.nightly.multi_node.internal_dp.scripts.test_multi_node import (
        test_multi_node as test_internal_dp,
    )

    await test_internal_dp()
