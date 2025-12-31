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

import sys
from unittest import mock


def _extra_stream_scope_check_for_test(match) -> bool:
    """
    Copied from the original implementation for testability.
    Checks if all nodes in the same stream.
    """
    non_default_streams = set()
    has_default = False

    for node in match.nodes:
        if node.op == "call_function":
            current_stream = node.meta.get("stream_label")
            if current_stream is None:
                has_default = True
            else:
                non_default_streams.add(current_stream)
                if len(non_default_streams) > 1:
                    return False

    if has_default and len(non_default_streams) > 0:
        return False

    return True


def test_extra_stream_scope_check():
    """Test the stream scope check logic."""

    class MockNode:

        def __init__(self, stream_label=None):
            self.op = "call_function"
            self.meta = {"stream_label": stream_label}

    class MockMatch:

        def __init__(self, nodes):
            self.nodes = nodes

    # Test 1: all default stream (None) → OK
    match1 = MockMatch([MockNode(None), MockNode(None)])
    assert _extra_stream_scope_check_for_test(match1) is True

    # Test 2: all same non-default stream → OK
    match2 = MockMatch([MockNode("s1"), MockNode("s1")])
    assert _extra_stream_scope_check_for_test(match2) is True

    # Test 3: mixed streams → FAIL
    match3 = MockMatch([MockNode("s1"), MockNode("s2")])
    assert _extra_stream_scope_check_for_test(match3) is False

    # Test 4: default + non-default → FAIL
    match4 = MockMatch([MockNode(None), MockNode("s1")])
    assert _extra_stream_scope_check_for_test(match4) is False

    # Test 5: empty nodes → OK (edge case)
    match5 = MockMatch([])
    assert _extra_stream_scope_check_for_test(match5) is True


def test_replacement_function_without_torch_npu(caplog):
    with mock.patch.dict(sys.modules, {
            'torch_npu': None,
            'torchair': None,
            'torch_npu.dynamo': None
    }):
        if 'vllm_ascend.compilation.npugraph_ex_passes.add_rms_norm_quant' in sys.modules:
            del sys.modules[
                'vllm_ascend.compilation.npugraph_ex_passes.add_rms_norm_quant']

        try:
            from vllm_ascend.compilation.npugraph_ex_passes.add_rms_norm_quant import \
                replacement_add_rms_norm_quant_with_bias
            result = replacement_add_rms_norm_quant_with_bias(epsilon=1e-5)
            assert result is None
        except (ImportError, AttributeError):
            pass
