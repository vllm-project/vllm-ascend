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
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.compilation.compiler_interface import AscendCompiler, _configure_backend


def _make_vllm_config():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(local_world_size=2, data_parallel_size_local=2),
        speculative_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 4, 8, 16]),
    )


@pytest.mark.parametrize("enable_super_kernel", [False, True])
def test_configure_npugraph_backend_respects_super_kernel_setting(enable_super_kernel):
    process_kwargs_options = MagicMock()
    ascend_compilation_config = SimpleNamespace(
        enable_static_kernel=True,
        enable_super_kernel=enable_super_kernel,
    )

    with patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "4"}):
        _configure_backend(
            MagicMock(),
            ascend_compilation_config,
            _make_vllm_config(),
            process_kwargs_options=process_kwargs_options,
        )

    options = process_kwargs_options.call_args.args[1]["options"]
    assert options["static_kernel_compile"] is True
    assert options["_vllm_aclnn_static_kernel_sym_range"] == [1, 4, 8]
    if enable_super_kernel:
        assert options["super_kernel_optimize"] is True
        assert options["super_kernel_optimize_options"] == {"dcci_after_kernel_end": [".*"]}
        assert options["super_kernel_debug_options"] == {}
    else:
        assert "super_kernel_optimize" not in options
        assert "super_kernel_optimize_options" not in options
        assert "super_kernel_debug_options" not in options


def test_compiler_hash_changes_when_super_kernel_setting_changes():
    disabled_config = SimpleNamespace(
        ascend_compilation_config=SimpleNamespace(
            enable_npugraph_ex=True,
            enable_static_kernel=True,
            enable_super_kernel=False,
        )
    )
    enabled_config = SimpleNamespace(
        ascend_compilation_config=SimpleNamespace(
            enable_npugraph_ex=True,
            enable_static_kernel=True,
            enable_super_kernel=True,
        )
    )
    compiler = AscendCompiler()

    with (
        patch.dict(sys.modules, {"torch_npu": SimpleNamespace(__version__="test-version")}),
        patch(
            "vllm_ascend.compilation.compiler_interface.get_ascend_config",
            side_effect=[disabled_config, enabled_config],
        ),
    ):
        disabled_hash = compiler.compute_hash(MagicMock())
        enabled_hash = compiler.compute_hash(MagicMock())

    assert disabled_hash != enabled_hash
