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

from vllm_ascend.compilation.passes.utils.npugraph_ex_utils_check import \
    extra_stream_scope_check


def test_extra_stream_scope_check_logic():
    """
    Test the extra_stream_scope_check logic used in both fusion patterns.
    This is a pure function test (copied logic for testability).
    """

    class MockNode:

        def __init__(self, stream_label=None):
            self.op = "call_function"
            self.meta = {"stream_label": stream_label}

    class MockMatch:

        def __init__(self, nodes):
            self.nodes = nodes

    # Test 1: all default → OK
    assert extra_stream_scope_check(
        MockMatch([MockNode(None), MockNode(None)])) is True

    # Test 2: same non-default → OK
    assert extra_stream_scope_check(
        MockMatch([MockNode("s1"), MockNode("s1")])) is True

    # Test 3: mixed non-default → FAIL
    assert extra_stream_scope_check(
        MockMatch([MockNode("s1"), MockNode("s2")])) is False

    # Test 4: default + non-default → FAIL
    assert extra_stream_scope_check(
        MockMatch([MockNode(None), MockNode("s1")])) is False

    # Test 5: empty → OK
    assert extra_stream_scope_check(MockMatch([])) is True
