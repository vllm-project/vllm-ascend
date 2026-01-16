#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""
Debug and printing utilities for vLLM Ascend.

This module provides functionality for:
- Printing debug information from within ACL graphs
- Managing print streams
"""

import atexit
from threading import Lock

import torch_npu  # noqa: F401

_GRAPH_PRINT_STREAM = None
_GRAPH_PRINT_STREAM_LOCK = Lock()
_SUBSCRIBED_COMPUTE_STREAMS = set()


def _print_callback_on_stream(*args):
    """Callback function to print arguments on the dedicated print stream."""
    global _GRAPH_PRINT_STREAM
    with torch_npu.npu.stream(_GRAPH_PRINT_STREAM):
        print(*args, flush=True)


def acl_graph_print(*args):
    """
    Prints arguments from within an ACL graph.

    This function is provided for developers to print debug information when encountering
    issues within an ACL graph, pretty handy for dumping input/output tensor values, or
    resolving unexpected hangs. Usage:
    ```python
    from vllm_ascend.utils import acl_graph_print
    ...
    acl_graph_print("Debug info")
    ```

    This function launches a host function on the current compute stream to print
    the given arguments. It uses a dedicated stream for printing to avoid
    interfering with computation.

    NOTE: torch.compile does not support this function, only use this in non-compiled code.
    For example, those custom ops like `unified_attention_with_output` or `moe_forward`.
    """
    global _SUBSCRIBED_COMPUTE_STREAMS
    global _GRAPH_PRINT_STREAM

    current_compute_stream = torch_npu.npu.current_stream()

    with _GRAPH_PRINT_STREAM_LOCK:
        if _GRAPH_PRINT_STREAM is None:
            _GRAPH_PRINT_STREAM = torch_npu.npu.Stream()

        if current_compute_stream not in _SUBSCRIBED_COMPUTE_STREAMS:
            # Subscribe the compute stream to allow launching host functions.
            torch_npu.npu._subscribe_report(current_compute_stream)
            _SUBSCRIBED_COMPUTE_STREAMS.add(current_compute_stream)

    torch_npu.npu._launch_host_func(current_compute_stream,
                                    _print_callback_on_stream, args)


def _unregister_print_streams_on_exit():
    """Unsubscribe all compute streams used for printing at exit."""
    global _SUBSCRIBED_COMPUTE_STREAMS
    with _GRAPH_PRINT_STREAM_LOCK:
        for stream in _SUBSCRIBED_COMPUTE_STREAMS:
            torch_npu.npu._unsubscribe_report(stream)


atexit.register(_unregister_print_streams_on_exit)
