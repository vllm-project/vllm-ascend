"""Bridge the current CI entry point to the historical test harness."""

from tests.e2e.nightly.single_node.models.scripts import test_single_node as _legacy

test_single_node = _legacy.test_single_node

__all__ = ["test_single_node"]
