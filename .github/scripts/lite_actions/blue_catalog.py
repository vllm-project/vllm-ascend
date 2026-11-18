"""Smoke case definitions shared with the model-owned launcher."""

from tests.integration_tests.training_cases.dispatch_smoke import (
    SMOKE_DEFAULTS,
    SMOKE_PROFILES,
    build_smoke_list,
)

BLUE_CASES = {case.test_name: case for case in build_smoke_list()}
BLUE_SMOKE_SUITE = tuple(SMOKE_PROFILES["dispatch_smoke"])
BLUE_DEFAULTS = SMOKE_DEFAULTS
