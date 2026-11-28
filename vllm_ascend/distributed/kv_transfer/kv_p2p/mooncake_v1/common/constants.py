# SPDX-License-Identifier: Apache-2.0
"""Shared timing constants for the mooncake V1 transfer threads."""

# Poll/timeout used by the background threads so a stop request can interrupt a
# blocking wait. The threads observe the stop event and exit on their own; the
# ZMQ socket is never closed from another thread.
THREAD_STOP_POLL_SECONDS = 0.2


# Best-effort upper bound for joining a background thread during shutdown.
SHUTDOWN_JOIN_TIMEOUT_SECONDS = 5.0
# Startup wait bound for the transfer threads; on timeout the worker rolls back.
KV_CACHE_THREAD_STARTUP_TIMEOUT_SECONDS = 5 * 60
