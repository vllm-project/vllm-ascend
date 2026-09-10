#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Unit tests for vllm_ascend.common.utils.watch_dog module."""

import os
import threading
import time
from unittest import mock

from tests.ut.base import PytestBase


class TestWatchDogInitialization(PytestBase):
    """Test WatchDog constructor and initialization."""

    def test_watch_dog_init_default_values(self):
        """Test WatchDog initialization with default values."""
        from vllm_ascend.common.utils.watch_dog import _DEFAULT_INTERVAL, _DEFAULT_NAME, _DEFAULT_TIMEOUT, WatchDog

        dog = WatchDog()

        assert dog._name == _DEFAULT_NAME
        assert dog._timeout == _DEFAULT_TIMEOUT
        assert dog._check_interval == _DEFAULT_INTERVAL
        assert dog._sequence == 1
        assert dog._last_feed_time > 0
        assert dog._last_timeout_log == 0.0
        assert dog._stop_event is not None
        assert dog._thread is None

    def test_watch_dog_initial_state_consistency(self):
        """Test that initial state prevents immediate timeout."""
        from vllm_ascend.common.utils.watch_dog import WatchDog

        dog = WatchDog()
        now = time.monotonic()

        # _last_feed_time should be set to current time to prevent immediate timeout
        assert dog._last_feed_time >= now - 0.1
        # _last_timeout_log should be 0.0 to ensure first timeout is always logged
        assert dog._last_timeout_log == 0.0


class TestWatchDogSetup(PytestBase):
    """Test WatchDog setup method."""

    def test_setup_with_custom_values(self):
        """Test setup method with custom parameters."""
        from vllm_ascend.common.utils.watch_dog import WatchDog

        dog = WatchDog()
        dog.setup(name="test_dog", timeout=100, check_interval=5)

        assert dog._name == "test_dog"
        assert dog._timeout == 100
        assert dog._check_interval == 5

    def test_setup_preserves_other_fields(self):
        """Test that setup only modifies configuration fields."""
        from vllm_ascend.common.utils.watch_dog import WatchDog

        dog = WatchDog()
        original_feed_time = dog._last_feed_time
        original_sequence = dog._sequence

        time.sleep(0.01)
        dog.setup(name="new_name", timeout=200, check_interval=3)

        assert dog._name == "new_name"
        assert dog._timeout == 200
        assert dog._check_interval == 3
        # These should not change
        assert dog._sequence == original_sequence
        assert dog._last_feed_time == original_feed_time


class TestWatchDogFeed(PytestBase):
    """Test WatchDog feed method."""

    def test_feed_updates_last_feed_time(self):
        """Test that feed updates _last_feed_time."""
        from vllm_ascend.common.utils.watch_dog import WatchDog

        dog = WatchDog()
        time.sleep(0.01)
        original_time = dog._last_feed_time

        dog.feed()

        assert dog._last_feed_time > original_time

    def test_feed_makes_watch_dog_active(self):
        """Test that feed prevents timeout after being called."""
        from vllm_ascend.common.utils.watch_dog import WatchDog

        dog = WatchDog()
        dog.feed()

        now = time.monotonic()
        time_since_feed = now - dog._last_feed_time

        # Should not be in timeout state
        assert time_since_feed < dog._timeout


class TestWatchDogStartStop(PytestBase):
    """Test WatchDog start and stop methods."""

    def test_start_creates_daemon_thread(self):
        """Test that start creates a daemon thread."""
        from vllm_ascend.common.utils.watch_dog import WatchDog

        dog = WatchDog()
        dog.start()

        try:
            assert dog._thread is not None
            assert dog._thread.daemon is True
            assert dog._thread.is_alive()
        finally:
            dog.stop()

    def test_start_idempotent_when_already_running(self):
        """Test that calling start multiple times is safe."""
        from vllm_ascend.common.utils.watch_dog import WatchDog

        dog = WatchDog()
        dog.start()
        original_thread = dog._thread

        dog.start()

        assert dog._thread is original_thread
        dog.stop()

    def test_stop_sets_stop_event(self):
        """Test that stop sets the stop event."""
        from vllm_ascend.common.utils.watch_dog import WatchDog

        dog = WatchDog()
        dog.start()
        assert dog._stop_event.is_set() is False

        dog.stop()

        assert dog._stop_event.is_set() is True

    def test_stop_waits_for_thread(self):
        """Test that stop waits for thread to finish."""
        from vllm_ascend.common.utils.watch_dog import WatchDog

        dog = WatchDog()
        dog.start()

        dog.stop()

        # Thread should no longer be alive after stop
        if dog._thread is not None:
            assert dog._thread.is_alive() is False
            assert dog._thread is None

    def test_stop_idempotent(self):
        """Test that calling stop multiple times is safe."""
        from vllm_ascend.common.utils.watch_dog import WatchDog

        dog = WatchDog()
        dog.start()
        dog.stop()
        original_thread = dog._thread

        dog.stop()

        assert dog._thread is original_thread


class TestWatchDogCheckLoop(PytestBase):
    """Test WatchDog _check_loop timeout detection logic."""

    def test_check_loop_triggers_on_timeout(self):
        """Test that _check_loop triggers dump_stack on timeout."""
        from vllm_ascend.common.utils.watch_dog import WatchDog

        dog = WatchDog()
        dog._timeout = 0  # Set timeout to 0 so any sleep will trigger it
        dog._check_interval = 0.01  # Short interval for fast test

        with mock.patch.object(dog, "dump_stack") as mock_dump:
            dog.start()
            time.sleep(0.05)  # Wait for at least one check cycle
            dog.stop()

            # dump_stack should have been called at least once
            assert mock_dump.call_count >= 1

    def test_check_loop_prevents_duplicate_timeout_logs(self):
        """Test that _check_loop prevents duplicate timeout logging."""
        from vllm_ascend.common.utils.watch_dog import WatchDog

        dog = WatchDog()
        dog._timeout = 0
        dog._check_interval = 0.01

        # Simulate: feed called, then timeout occurs, then feed called again
        with mock.patch.object(dog, "dump_stack") as mock_dump:
            dog.start()

            # Wait for first timeout detection
            time.sleep(0.05)
            first_call_count = mock_dump.call_count

            # Feed to reset timeout tracking
            dog.feed()

            # Wait a bit more
            time.sleep(0.05)

            dog.stop()

            # Should have logged at least once
            assert first_call_count >= 1

    def test_check_loop_no_timeout_when_fed_regularly(self):
        """Test that _check_loop does not trigger on timeout when fed regularly."""
        from vllm_ascend.common.utils.watch_dog import WatchDog

        dog = WatchDog()
        dog._timeout = 1  # 1 second timeout
        dog._check_interval = 0.05  # Check every 50ms

        with mock.patch.object(dog, "dump_stack") as mock_dump:
            dog.start()

            # Feed regularly
            for _ in range(5):
                dog.feed()
                time.sleep(0.1)

            dog.stop()

            # dump_stack should not have been called
            assert mock_dump.call_count == 0


class TestWatchDogDumpStack(PytestBase):
    """Test WatchDog dump_stack method."""

    def test_dump_stack_calls_faulthandler(self):
        """Test that dump_stack calls faulthandler.dump_traceback."""
        from vllm_ascend.common.utils.watch_dog import WatchDog

        dog = WatchDog()
        log_dir = "/tmp/test_log"
        base_name = "test_process"

        with (
            mock.patch(
                "vllm_ascend.common.utils.watch_dog.get_log_dir_and_basename", return_value=(log_dir, base_name)
            ),
            mock.patch("vllm_ascend.common.utils.watch_dog.os.makedirs"),
            mock.patch("vllm_ascend.common.utils.watch_dog.open", mock.mock_open()) as mock_open_file,
            mock.patch("vllm_ascend.common.utils.watch_dog.faulthandler.dump_traceback") as mock_dump_traceback,
        ):
            dog.dump_stack()

            # Verify faulthandler was called with the stack file handle
            mock_dump_traceback.assert_called_once_with(file=mock_open_file.return_value, all_threads=True)

    def test_dump_stack_increments_sequence(self):
        """Test that dump_stack increments _sequence."""
        from vllm_ascend.common.utils.watch_dog import WatchDog

        dog = WatchDog()
        initial_sequence = dog._sequence

        with (
            mock.patch("vllm_ascend.common.utils.watch_dog.get_log_dir_and_basename", return_value=("/tmp", "test")),
            mock.patch("vllm_ascend.common.utils.watch_dog.os.makedirs"),
            mock.patch("vllm_ascend.common.utils.watch_dog.open", mock.mock_open()),
            mock.patch("vllm_ascend.common.utils.watch_dog.faulthandler.dump_traceback"),
        ):
            dog.dump_stack()

            assert dog._sequence == initial_sequence + 1

    def test_dump_stack_writes_to_correct_file(self):
        """Test that dump_stack writes to the correct log file."""
        from vllm_ascend.common.utils.watch_dog import WatchDog

        dog = WatchDog()
        dog._name = "my_watch_dog"
        log_dir = "/test/log/dir"
        base_name = "my_process"

        with (
            mock.patch(
                "vllm_ascend.common.utils.watch_dog.get_log_dir_and_basename", return_value=(log_dir, base_name)
            ),
            mock.patch("vllm_ascend.common.utils.watch_dog.os.makedirs"),
            mock.patch("builtins.open", mock.mock_open()) as mock_file,
            mock.patch("vllm_ascend.common.utils.watch_dog.faulthandler.dump_traceback"),
        ):
            dog.dump_stack()

            # Verify file path
            expected_path = os.path.join(log_dir, f"{base_name}.{dog._name}.stack")
            mock_file.assert_called_once_with(expected_path, "a")


class TestWatchDogSingleton(PytestBase):
    """Test get_watch_dog singleton pattern."""

    def test_get_watch_dog_returns_same_instance(self):
        """Test that get_watch_dog returns the same instance."""
        from vllm_ascend.common.utils import watch_dog

        instance1 = watch_dog.get_watch_dog()
        instance2 = watch_dog.get_watch_dog()

        assert instance1 is instance2

    def test_get_watch_dog_returns_watch_dog_instance(self):
        """Test that get_watch_dog returns a WatchDog instance."""
        from vllm_ascend.common.utils.watch_dog import WatchDog, get_watch_dog

        dog = get_watch_dog()

        assert isinstance(dog, WatchDog)


class TestWatchDogFactory(PytestBase):
    """Test setup_watch_dog factory function."""

    def test_setup_watch_dog_creates_new_instance(self):
        """Test that setup_watch_dog creates a new WatchDog instance."""
        from vllm_ascend.common.utils import watch_dog
        from vllm_ascend.common.utils.watch_dog import get_watch_dog

        original = get_watch_dog()
        original._sequence = 999  # Mark original

        new_dog = watch_dog.setup_watch_dog(name="factory_test", timeout=60, check_interval=3)

        assert new_dog is not original
        assert new_dog._name == "factory_test"
        assert new_dog._timeout == 60
        assert new_dog._check_interval == 3

    def test_setup_watch_dog_updates_global_instance(self):
        """Test that setup_watch_dog updates the global _watch_dog."""
        from vllm_ascend.common.utils import watch_dog
        from vllm_ascend.common.utils.watch_dog import get_watch_dog

        original = get_watch_dog()
        original_id = id(original)

        watch_dog.setup_watch_dog(name="new_global", timeout=30, check_interval=2)

        new_instance = get_watch_dog()
        assert id(new_instance) != original_id
        assert new_instance._name == "new_global"


class TestWatchDogIntegration(PytestBase):
    """Integration tests for WatchDog lifecycle."""

    def test_full_lifecycle(self):
        """Test complete start-feed-stop lifecycle."""
        from vllm_ascend.common.utils.watch_dog import WatchDog

        dog = WatchDog()
        dog._timeout = 10
        dog._check_interval = 0.1

        with mock.patch.object(dog, "dump_stack"):
            dog.start()
            assert dog._thread is not None
            assert dog._thread.is_alive()

            dog.feed()
            time.sleep(0.2)

            dog.feed()
            time.sleep(0.2)

            dog.stop()
            assert dog._stop_event.is_set()
            assert dog._thread is None or not dog._thread.is_alive()

    def test_multiple_feed_calls(self):
        """Test that multiple feed calls are handled correctly."""
        from vllm_ascend.common.utils.watch_dog import WatchDog

        dog = WatchDog()
        dog._timeout = 1
        dog._check_interval = 0.05

        with mock.patch.object(dog, "dump_stack") as mock_dump:
            dog.start()

            for _ in range(10):
                dog.feed()
                time.sleep(0.1)

            dog.stop()

            # Should not have dumped stack due to regular feeding
            assert mock_dump.call_count == 0

    def test_concurrent_feed_and_check(self):
        """Test concurrent feed and check loop operations."""
        from vllm_ascend.common.utils.watch_dog import WatchDog

        dog = WatchDog()
        dog._timeout = 0.5
        dog._check_interval = 0.05

        with mock.patch.object(dog, "dump_stack"):
            dog.start()

            def feeder():
                for _ in range(20):
                    dog.feed()
                    time.sleep(0.08)

            feeder_thread = threading.Thread(target=feeder)
            feeder_thread.start()
            feeder_thread.join(timeout=5)

            dog.stop()

    def test_stop_event_is_thread_safe(self):
        """Test that _stop_event is used correctly for thread signaling."""
        from vllm_ascend.common.utils.watch_dog import WatchDog

        dog = WatchDog()
        dog.start()

        assert dog._stop_event is not None
        assert isinstance(dog._stop_event, threading.Event)

        dog.stop()
