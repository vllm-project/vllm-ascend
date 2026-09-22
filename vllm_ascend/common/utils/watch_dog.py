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
import faulthandler
import os
import threading
import time

from vllm_ascend.common.utils.log_file import get_log_dir_and_basename

_DEFAULT_NAME = "vllm_ascend"
_DEFAULT_TIMEOUT = 300
_DEFAULT_INTERVAL = 10


class WatchDog:
    def __init__(self, name=_DEFAULT_NAME, timeout=_DEFAULT_TIMEOUT, check_interval=_DEFAULT_INTERVAL):
        """
        :param timeout: Timeout threshold in seconds
        :param check_interval: Check interval in seconds
        """
        self._name = name
        self._timeout = timeout
        self._check_interval = check_interval
        self._sequence = 1

        # Initialize feed time to current time to avoid immediate timeout on startup
        self._last_feed_time = time.monotonic()
        # Last timeout log time, initialized to 0 to ensure first timeout is always logged
        self._last_timeout_log = 0.0

        self._stop_event = threading.Event()
        self._thread = None
        self._dump_lock = threading.Lock()

    def setup(self, name=_DEFAULT_NAME, timeout=_DEFAULT_TIMEOUT, check_interval=_DEFAULT_INTERVAL):
        self._name = name
        self._timeout = timeout
        self._check_interval = check_interval

    def feed(self):
        """Feed interface, external callers use this to update last feed time"""
        # Single float assignment is atomic in CPython
        self._last_feed_time = time.monotonic()

    def dump_stack(self):
        with self._dump_lock:
            log_dir, base_name = get_log_dir_and_basename()
            stack_filename = os.path.join(log_dir, f"{base_name}.{self._name}.stack")
            with open(stack_filename, "a") as f:
                f.write(f"\nCall Stack: the {self._sequence} time at [{time.time()}] {time.ctime()}\n\n")
                faulthandler.dump_traceback(file=f, all_threads=True)
                f.write("\n================================================================\n")
            self._sequence += 1

    def _check_loop(self):
        """Main loop of the background check thread"""
        while not self._stop_event.is_set():
            time.sleep(self._check_interval)

            now = time.monotonic()
            # Calculate time elapsed since last feed
            if now - self._last_feed_time > self._timeout:
                # If last timeout log time is earlier than last feed time,
                # it means timeout hasn't been logged since last feed, need to log and record
                if self._last_timeout_log < self._last_feed_time:
                    # Update log timestamp to current time to avoid duplicate logs
                    self._last_timeout_log = now
                    self.dump_stack()
            # If not timed out, do nothing and keep _last_timeout_log unchanged

    def start(self):
        """Start the watchdog background thread (daemon thread)"""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._check_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the watchdog background thread"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None


_watch_dog = WatchDog()


def get_watch_dog() -> WatchDog:
    return _watch_dog


def setup_watch_dog(name: str, timeout=5, check_interval=1) -> WatchDog:
    global _watch_dog
    _watch_dog = WatchDog(name, timeout, check_interval)
    return get_watch_dog()
