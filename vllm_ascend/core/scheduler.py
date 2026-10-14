# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ascend scheduler subclasses retaining upstream scheduling and async hooks."""

from vllm.v1.core.sched.async_scheduler import AsyncScheduler

from vllm_ascend.patch.platform.patch_balance_schedule import BalanceScheduler


class AscendScheduler(BalanceScheduler):
    """Retain existing Balance/SWA hooks and share per-group KV reservations."""


class AscendAsyncScheduler(AsyncScheduler, AscendScheduler):
    """Keep upstream asynchronous hooks and share Ascend admission accounting."""
