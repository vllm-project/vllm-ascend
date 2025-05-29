#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/tests/utils.py
# Copyright 2023 The vLLM team.
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
#

import functools
import os
import signal
from typing import Callable

from typing_extensions import ParamSpec

_P = ParamSpec("_P")


def fork_new_process_for_each_test(
        f: Callable[_P, None]) -> Callable[_P, None]:
    """Decorator to fork a new process for each test function.
    See https://github.com/vllm-project/vllm/issues/7053 for more details.
    """

    @functools.wraps(f)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> None:
        # Make the process the leader of its own process group
        # to avoid sending SIGTERM to the parent process
        os.setpgrp()
        from _pytest.outcomes import Skipped
        pid = os.fork()
        print(f"Fork a new process to run a test {pid}")
        if pid == 0:
            try:
                f(*args, **kwargs)
            except Skipped as e:
                # convert Skipped to exit code 0
                print(str(e))
                os._exit(0)
            except Exception:
                import traceback
                traceback.print_exc()
                os._exit(1)
            else:
                os._exit(0)
        else:
            pgid = os.getpgid(pid)
            _pid, _exitcode = os.waitpid(pid, 0)
            # ignore SIGTERM signal itself
            old_signal_handler = signal.signal(signal.SIGTERM, signal.SIG_IGN)
            # kill all child processes
            os.killpg(pgid, signal.SIGTERM)
            # restore the signal handler
            signal.signal(signal.SIGTERM, old_signal_handler)
            assert _exitcode == 0, (f"function {f} failed when called with"
                                    f" args {args} and kwargs {kwargs}")

    return wrapper
