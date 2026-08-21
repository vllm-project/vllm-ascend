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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor

from vllm_ascend._310p.graph_input_contract import GraphInputSource
from vllm_ascend.compilation.acl_graph import (
    ACLGraphEntry,
    ACLGraphWrapper,
    _prepare_aclgraph_debug_dump,
)


def _config():
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dflash",
            num_speculative_tokens=15,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            cudagraph_capture_sizes=[64, 32, 16],
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=16),
    )


def _wrapper(component: str) -> ACLGraphWrapper:
    with (
        patch(
            "vllm_ascend.compilation.acl_graph.current_platform.get_global_graph_pool",
            return_value=MagicMock(),
        ),
        patch(
            "vllm_ascend.compilation.acl_graph.is_310p_dflash_full_decode_only",
            return_value=True,
            create=True,
        ),
        patch(
            "vllm_ascend.compilation.acl_graph.is_310p_dflash_piecewise",
            return_value=False,
        ),
        patch("vllm_ascend.compilation.acl_graph.envs.VLLM_LOGGING_LEVEL", "INFO"),
    ):
        return ACLGraphWrapper(
            runnable=MagicMock(),
            vllm_config=_config(),
            runtime_mode=CUDAGraphMode.FULL,
            component=component,
            retained_input_provider=lambda _context, _descriptor: (),
        )


def test_full_decode_wrapper_uses_explicit_component_identity():
    target = _wrapper("target")
    draft = _wrapper("draft")

    assert target.component == "target"
    assert draft.component == "draft"
    assert target.validate_input_contracts is True
    assert draft.validate_input_contracts is True


def test_full_decode_wrapper_rejects_implicit_component_identity():
    with (
        patch(
            "vllm_ascend.compilation.acl_graph.current_platform.get_global_graph_pool",
            return_value=MagicMock(),
        ),
        patch(
            "vllm_ascend.compilation.acl_graph.is_310p_dflash_full_decode_only",
            return_value=True,
            create=True,
        ),
        patch(
            "vllm_ascend.compilation.acl_graph.is_310p_dflash_piecewise",
            return_value=False,
        ),
        patch("vllm_ascend.compilation.acl_graph.envs.VLLM_LOGGING_LEVEL", "INFO"),
        pytest.raises(ValueError, match="explicit.*component"),
    ):
        ACLGraphWrapper(
            runnable=MagicMock(),
            vllm_config=_config(),
            runtime_mode=CUDAGraphMode.FULL,
        )


def test_full_decode_wrapper_requires_explicit_retained_input_provider():
    with (
        patch(
            "vllm_ascend.compilation.acl_graph.current_platform.get_global_graph_pool",
            return_value=MagicMock(),
        ),
        patch(
            "vllm_ascend.compilation.acl_graph.is_310p_dflash_full_decode_only",
            return_value=True,
            create=True,
        ),
        patch(
            "vllm_ascend.compilation.acl_graph.is_310p_dflash_piecewise",
            return_value=False,
        ),
        patch("vllm_ascend.compilation.acl_graph.envs.VLLM_LOGGING_LEVEL", "INFO"),
        pytest.raises(ValueError, match="retained input provider"),
    ):
        ACLGraphWrapper(
            runnable=MagicMock(),
            vllm_config=_config(),
            runtime_mode=CUDAGraphMode.FULL,
            component="target",
        )


def test_full_decode_contract_combines_call_and_provider_roles():
    retained = torch.ones(4, dtype=torch.int32)
    provider = MagicMock(
        return_value=(
            GraphInputSource(
                role="target.runner.positions",
                tensor=retained,
                ownership="runner-persistent-buffer",
                required_alignment=16,
                alignment_source="test-consumer",
                mutable=True,
                bounded_view=True,
            ),
        )
    )
    wrapper = _wrapper("target")
    wrapper.retained_input_provider = provider
    descriptor = BatchDescriptor(num_tokens=16, num_reqs=1, uniform=True)
    context = SimpleNamespace()
    call_tensor = torch.ones(4)

    contracts = wrapper._capture_input_contracts(
        (call_tensor,),
        {},
        forward_context=context,
        batch_descriptor=descriptor,
    )

    assert contracts is not None
    assert [contract.path for contract in contracts] == [
        "args[0]",
        "target.runner.positions",
    ]
    provider.assert_called_once_with(context, descriptor)


def test_full_decode_capture_completion_replays_then_records_manifest():
    wrapper = _wrapper("draft")
    descriptor = BatchDescriptor(
        num_tokens=16,
        num_reqs=1,
        uniform=True,
    )
    graph = MagicMock()
    entry = ACLGraphEntry(
        batch_descriptor=descriptor,
        aclgraph=graph,
        output=torch.ones(1),
        input_contracts=(),
        component="draft",
        rank=2,
        capture_count=1,
    )

    with (
        patch.object(
            wrapper,
            "_validate_replay_input_contracts",
        ) as validate_contracts,
        patch(
            "vllm_ascend.compilation.acl_graph.torch.npu.current_stream",
        ) as current_stream,
        patch(
            "vllm_ascend.compilation.acl_graph.record_full_decode_capture",
            create=True,
        ) as record_capture,
    ):
        wrapper._complete_full_decode_capture(entry, (), {})

    validate_contracts.assert_called_once_with(entry, (), {})
    graph.replay.assert_called_once_with()
    current_stream.return_value.synchronize.assert_called_once_with()
    assert entry.contract_validated is True
    assert entry.warmup_replay_count == 1
    record_capture.assert_called_once_with(
        component="draft",
        local_rank=2,
        runtime_mode=CUDAGraphMode.FULL,
        descriptor=descriptor,
        capture_count=1,
        warmup_replay_count=1,
        output_bound=True,
        contract_validated=True,
    )


def test_full_decode_debug_dump_is_opt_in_and_descriptor_scoped(tmp_path):
    graph = MagicMock()
    descriptor = BatchDescriptor(num_tokens=32, num_reqs=2, uniform=True)

    with patch.dict(
        "os.environ",
        {"VLLM_ASCEND_DFLASH_GRAPH_DUMP_DIR": str(tmp_path)},
        clear=False,
    ):
        dump_path = _prepare_aclgraph_debug_dump(
            graph,
            component="target",
            rank=0,
            descriptor=descriptor,
        )

    graph.enable_debug_mode.assert_called_once_with()
    assert dump_path is not None
    assert dump_path.parent == tmp_path
    assert "target-rank0" in dump_path.name
    assert "tokens32-reqs2" in dump_path.name


def test_full_decode_debug_dump_is_disabled_by_default():
    graph = MagicMock()

    with patch.dict("os.environ", {}, clear=True):
        dump_path = _prepare_aclgraph_debug_dump(
            graph,
            component="target",
            rank=0,
            descriptor=BatchDescriptor(num_tokens=32, num_reqs=2, uniform=True),
        )

    assert dump_path is None
    graph.enable_debug_mode.assert_not_called()


def test_full_decode_replay_error_reports_component_and_runtime_shape():
    """A blocking replay failure must identify the graph that actually failed."""
    wrapper = _wrapper("draft")
    descriptor = BatchDescriptor(
        num_tokens=160,
        num_reqs=10,
        uniform=True,
    )
    graph = MagicMock()
    graph.replay.side_effect = RuntimeError("device replay failed")
    wrapper.concrete_aclgraph_entries[descriptor] = ACLGraphEntry(
        batch_descriptor=descriptor,
        aclgraph=graph,
        output=torch.ones(1),
        input_contracts=(),
        component="draft",
        rank=0,
        capture_count=1,
    )
    context = SimpleNamespace(
        batch_descriptor=descriptor,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        num_actual_tokens=96,
    )

    with (
        patch(
            "vllm_ascend.compilation.acl_graph.get_forward_context",
            return_value=context,
        ),
        patch(
            "vllm_ascend.ascend_forward_context.get_forward_context",
            return_value=SimpleNamespace(is_draft_model=True),
        ),
        patch.object(wrapper, "_validate_replay_input_contracts"),
        patch(
            "vllm_ascend.compilation.acl_graph.torch.npu.current_stream",
        ),
        pytest.raises(
            RuntimeError,
            match=(
                "FULL graph replay failed: component=draft.*"
                "descriptor=BatchDescriptor\\(num_tokens=160.*"
                "actual_tokens=96"
            ),
        ) as exc_info,
    ):
        wrapper()

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "device replay failed"


def test_full_decode_debug_sync_reports_async_replay_error():
    """DEBUG must attribute an asynchronous device error to this replay."""
    wrapper = _wrapper("target")
    wrapper.is_debugging_mode = True
    descriptor = BatchDescriptor(
        num_tokens=160,
        num_reqs=10,
        uniform=True,
    )
    graph = MagicMock()
    wrapper.concrete_aclgraph_entries[descriptor] = ACLGraphEntry(
        batch_descriptor=descriptor,
        aclgraph=graph,
        output=torch.ones(1),
        input_contracts=(),
        input_addresses=[],
        component="target",
        rank=0,
        capture_count=1,
    )
    context = SimpleNamespace(
        batch_descriptor=descriptor,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        num_actual_tokens=96,
    )
    stream = MagicMock()
    stream.synchronize.side_effect = [
        None,
        RuntimeError("asynchronous device failure"),
    ]

    with (
        patch(
            "vllm_ascend.compilation.acl_graph.get_forward_context",
            return_value=context,
        ),
        patch(
            "vllm_ascend.ascend_forward_context.get_forward_context",
            return_value=SimpleNamespace(is_draft_model=False),
        ),
        patch.object(wrapper, "_validate_replay_input_contracts"),
        patch(
            "vllm_ascend.compilation.acl_graph.torch.npu.current_stream",
            return_value=stream,
        ),
        pytest.raises(
            RuntimeError,
            match=(
                "FULL graph replay failed: component=target.*"
                "descriptor=BatchDescriptor\\(num_tokens=160.*"
                "actual_tokens=96"
            ),
        ) as exc_info,
    ):
        wrapper()

    graph.replay.assert_called_once_with()
    assert stream.synchronize.call_count == 2
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "asynchronous device failure"


def test_full_decode_replay_does_not_construct_or_move_device_tensors():
    """Eligible replay must consume retained storage without allocating it."""
    wrapper = _wrapper("draft")
    descriptor = BatchDescriptor(
        num_tokens=160,
        num_reqs=10,
        uniform=True,
    )
    graph = MagicMock()
    output = object()
    wrapper.concrete_aclgraph_entries[descriptor] = ACLGraphEntry(
        batch_descriptor=descriptor,
        aclgraph=graph,
        output=output,
        input_contracts=(),
        component="draft",
        rank=0,
        capture_count=1,
    )
    context = SimpleNamespace(
        batch_descriptor=descriptor,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        num_actual_tokens=160,
    )
    forbidden = MagicMock(side_effect=AssertionError("eligible replay allocated a device tensor"))

    with (
        patch(
            "vllm_ascend.compilation.acl_graph.get_forward_context",
            return_value=context,
        ),
        patch(
            "vllm_ascend.ascend_forward_context.get_forward_context",
            return_value=SimpleNamespace(is_draft_model=True),
        ),
        patch.object(wrapper, "_validate_replay_input_contracts"),
        patch("vllm_ascend.compilation.acl_graph.torch.npu.current_stream"),
        patch("vllm_ascend.compilation.acl_graph.torch.empty", forbidden),
        patch("vllm_ascend.compilation.acl_graph.torch.zeros", forbidden),
        patch("vllm_ascend.compilation.acl_graph.torch.ones", forbidden),
        patch("vllm_ascend.compilation.acl_graph.torch.tensor", forbidden),
    ):
        actual = wrapper()

    assert actual is output
    forbidden.assert_not_called()
    graph.replay.assert_called_once_with()


def test_full_decode_replay_orders_contract_sync_launch_and_consumption():
    """Input validation/sync precede launch and output is returned afterward."""
    wrapper = _wrapper("target")
    descriptor = BatchDescriptor(
        num_tokens=160,
        num_reqs=10,
        uniform=True,
    )
    output = object()
    graph = MagicMock()
    wrapper.concrete_aclgraph_entries[descriptor] = ACLGraphEntry(
        batch_descriptor=descriptor,
        aclgraph=graph,
        output=output,
        input_contracts=(),
        component="target",
        rank=0,
        capture_count=1,
    )
    context = SimpleNamespace(
        batch_descriptor=descriptor,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        num_actual_tokens=160,
    )
    stream = MagicMock()
    lifecycle: list[str] = []
    validate = MagicMock(side_effect=lambda *_args: lifecycle.append("validate"))
    stream.synchronize.side_effect = lambda: lifecycle.append("synchronize")
    graph.replay.side_effect = lambda: lifecycle.append("replay")

    with (
        patch(
            "vllm_ascend.compilation.acl_graph.get_forward_context",
            return_value=context,
        ),
        patch(
            "vllm_ascend.ascend_forward_context.get_forward_context",
            return_value=SimpleNamespace(is_draft_model=False),
        ),
        patch(
            "vllm_ascend.compilation.acl_graph.torch.npu.current_stream",
            return_value=stream,
        ),
        patch.object(wrapper, "_validate_replay_input_contracts", validate),
    ):
        actual = wrapper()

    assert lifecycle == ["validate", "synchronize", "replay"]
    assert actual is output
