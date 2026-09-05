# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from vllm_ascend.patch.platform.patch_partial_wake_up import (
    _wake_up_without_early_scheduler_resume,
)


class _FakeExecutor:
    def __init__(self) -> None:
        self.sleeping_tags = {"weights", "kv_cache"}
        self.wake_calls: list[list[str] | None] = []

    @property
    def is_sleeping(self) -> bool:
        return bool(self.sleeping_tags)

    def wake_up(self, tags: list[str] | None = None) -> None:
        self.wake_calls.append(tags)
        if tags is None:
            self.sleeping_tags.clear()
        else:
            self.sleeping_tags.difference_update(tags)


class _FakeEngineCore:
    def __init__(self) -> None:
        self.model_executor = _FakeExecutor()
        self.resume_count = 0

    def resume_scheduler(self) -> None:
        self.resume_count += 1


def test_weights_only_wake_keeps_scheduler_paused() -> None:
    core = _FakeEngineCore()

    _wake_up_without_early_scheduler_resume(core, ["weights"])

    assert core.model_executor.wake_calls == [["weights"]]
    assert core.model_executor.sleeping_tags == {"kv_cache"}
    assert core.resume_count == 0


def test_final_kv_cache_wake_resumes_scheduler() -> None:
    core = _FakeEngineCore()
    _wake_up_without_early_scheduler_resume(core, ["weights"])

    _wake_up_without_early_scheduler_resume(core, ["kv_cache"])

    assert core.model_executor.sleeping_tags == set()
    assert core.resume_count == 1


def test_full_wake_resumes_scheduler() -> None:
    core = _FakeEngineCore()

    _wake_up_without_early_scheduler_resume(core)

    assert core.model_executor.wake_calls == [None]
    assert core.resume_count == 1


def test_explicit_scheduling_wake_resumes_without_memory_wake() -> None:
    core = _FakeEngineCore()

    _wake_up_without_early_scheduler_resume(core, ["scheduling"])

    assert core.model_executor.wake_calls == []
    assert core.resume_count == 1
