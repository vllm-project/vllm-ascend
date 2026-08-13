#!/bin/bash

# Build the optional arguments shared by single-node and multi-node AOP.
# The caller receives the result in BISECT_EXTRA_ARGS as a Bash array.
build_bisect_extra_args() {
  BISECT_EXTRA_ARGS=()
  [ -n "${BISECT_GOOD_COMMIT:-}" ] &&
    BISECT_EXTRA_ARGS+=(--good-commit "$BISECT_GOOD_COMMIT")
  [ -n "${BISECT_FAIL_CONFIRM_RETRIES:-}" ] &&
    BISECT_EXTRA_ARGS+=(--fail-confirm-retries "$BISECT_FAIL_CONFIRM_RETRIES")
  [ -n "${BISECT_TRIAL_TIMEOUT:-}" ] &&
    BISECT_EXTRA_ARGS+=(--trial-timeout-s "$BISECT_TRIAL_TIMEOUT")
  [ -n "${BISECT_BARRIER_TIMEOUT:-}" ] &&
    BISECT_EXTRA_ARGS+=(--barrier-timeout-s "$BISECT_BARRIER_TIMEOUT")
  [ "${BISECT_NO_VERIFY_GOOD:-}" = "true" ] &&
    BISECT_EXTRA_ARGS+=(--no-verify-good)
  [ "${BISECT_NO_VERIFY_BAD:-}" = "true" ] &&
    BISECT_EXTRA_ARGS+=(--no-verify-bad)
  [ "${BISECT_FORCE_INITIAL_BUILD:-}" = "true" ] &&
    BISECT_EXTRA_ARGS+=(--force-initial-build)

  # The rebuild policy is owned by AOP and is intentionally not user-overridable.
  BISECT_EXTRA_ARGS+=(--native-check since-build)
  [ -n "${BISECT_CONFIG_BASE_PATH:-}" ] &&
    BISECT_EXTRA_ARGS+=(--config-base-path "$BISECT_CONFIG_BASE_PATH")
}
