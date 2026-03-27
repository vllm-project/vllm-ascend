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
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
from typing import Any
from unittest.mock import MagicMock, call

import pytest


_WORKTREE_ROOT = Path(__file__).resolve().parents[5]
_PATCH_MODULE_PATH = _WORKTREE_ROOT / "vllm_ascend/patch/worker/patch_distributed.py"
_PATCH_MODULE_NAME = "vllm_ascend.patch.worker.patch_distributed"
_REGISTRY_MODULE_PATH = _WORKTREE_ROOT / "vllm_ascend/patch/worker/_hccl_pg_registry.py"
_REGISTRY_MODULE_NAME = "vllm_ascend.patch.worker._hccl_pg_registry"


class FakeBackend(str):
    pass


class FakeTensor:
    def __init__(self, shape: tuple[int, ...]):
        self._shape = shape

    def dim(self) -> int:
        return len(self._shape)

    def size(self) -> tuple[int, ...]:
        return self._shape


class FakeProcessGroup:
    def __init__(
        self,
        backend: str,
        ranks: tuple[int, ...],
        sequence: int,
        pg_options: object | None = None,
    ):
        self.backend = backend
        self.ranks = ranks
        self.sequence = sequence
        self.pg_options = pg_options


@dataclass
class RealisticFakeHcclOptions:
    backend: str = "hccl"
    global_ranks_in_group: list[int] | tuple[int, ...] = ()
    group_id: str = ""
    group_name: str = ""
    hccl_config: dict[str, int] | None = None
    is_high_priority_stream: bool = False
    op_timeout: object = timedelta(seconds=10)


def _load_module(module_name: str, module_path: Path) -> Any:
    spec = spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {module_name}")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def _load_patch_distributed_module():
    non_group_member = object()
    new_group_calls: list[dict[str, object]] = []
    destroy_process_group = MagicMock()
    destroy_distributed_environment = MagicMock(
        name="destroy_distributed_environment"
    )
    get_rank = MagicMock(return_value=0)
    current_device = MagicMock(return_value="npu:0")
    communicator_instances: list[object] = []
    unique_name_counter = {"value": 0}
    sequence_counter = {"value": 0}
    shared_hccl_options = {"hccl_config": {"hccl_buffer_size": 200}}

    torch_module: Any = ModuleType("torch")
    torch_distributed: Any = ModuleType("torch.distributed")
    torch_distributed_c10d: Any = ModuleType("torch.distributed.distributed_c10d")

    def new_group(ranks, backend, pg_options=None):
        backend_name = str(backend)
        new_group_calls.append({
            "ranks": tuple(ranks),
            "backend": backend_name,
            "pg_options": pg_options,
        })
        if get_rank() not in ranks:
            return non_group_member
        handle = FakeProcessGroup(
            backend=backend_name,
            ranks=tuple(ranks),
            sequence=sequence_counter["value"],
            pg_options=pg_options,
        )
        sequence_counter["value"] += 1
        return handle

    class GroupMember:
        NON_GROUP_MEMBER = non_group_member

    torch_distributed.Backend = FakeBackend
    torch_distributed.get_rank = get_rank
    torch_distributed.new_group = new_group
    torch_distributed.destroy_process_group = destroy_process_group
    torch_distributed.distributed_c10d = torch_distributed_c10d
    torch_distributed_c10d.GroupMember = GroupMember

    torch_module.Tensor = FakeTensor
    torch_module.distributed = torch_distributed
    torch_module.equal = lambda lhs, rhs: lhs is rhs
    torch_module.randn = lambda *shape: FakeTensor(shape)
    torch_module.npu = SimpleNamespace(current_device=current_device)

    vllm_module: Any = ModuleType("vllm")
    vllm_distributed: Any = ModuleType("vllm.distributed")
    parallel_state_module: Any = ModuleType("vllm.distributed.parallel_state")

    class BaseGroupCoordinator:
        pass

    def _get_unique_name(group_name: str) -> str:
        unique_name_counter["value"] += 1
        return f"{group_name}-{unique_name_counter['value']}"

    parallel_state_module.GroupCoordinator = BaseGroupCoordinator
    parallel_state_module._get_unique_name = _get_unique_name
    parallel_state_module._register_group = MagicMock()
    parallel_state_module.destroy_distributed_environment = (
        destroy_distributed_environment
    )

    shm_broadcast_module: Any = ModuleType(
        "vllm.distributed.device_communicators.shm_broadcast"
    )

    class MessageQueue:
        create_from_process_group = MagicMock(
            side_effect=lambda group, *_: SimpleNamespace(group=group)
        )

    shm_broadcast_module.MessageQueue = MessageQueue

    vllm_distributed.parallel_state = parallel_state_module
    vllm_distributed.destroy_distributed_environment = (
        destroy_distributed_environment
    )
    vllm_module.distributed = vllm_distributed

    vllm_ascend_module: Any = ModuleType("vllm_ascend")
    vllm_ascend_patch: Any = ModuleType("vllm_ascend.patch")
    vllm_ascend_patch_worker: Any = ModuleType("vllm_ascend.patch.worker")
    vllm_ascend_distributed: Any = ModuleType("vllm_ascend.distributed")
    vllm_ascend_device_communicators: Any = ModuleType(
        "vllm_ascend.distributed.device_communicators"
    )
    npu_communicator_module: Any = ModuleType(
        "vllm_ascend.distributed.device_communicators.npu_communicator"
    )
    utils_module: Any = ModuleType("vllm_ascend.utils")

    class FakeNPUCommunicator:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.destroy = MagicMock()
            self.all_to_all = MagicMock()
            communicator_instances.append(self)

    npu_communicator_module.NPUCommunicator = FakeNPUCommunicator
    utils_module.create_hccl_pg_options = MagicMock(return_value=shared_hccl_options)

    vllm_ascend_module.patch = vllm_ascend_patch
    vllm_ascend_patch.worker = vllm_ascend_patch_worker
    vllm_ascend_module.distributed = vllm_ascend_distributed
    vllm_ascend_distributed.device_communicators = vllm_ascend_device_communicators

    modules = {
        "torch": torch_module,
        "torch.distributed": torch_distributed,
        "torch.distributed.distributed_c10d": torch_distributed_c10d,
        "vllm": vllm_module,
        "vllm.distributed": vllm_distributed,
        "vllm.distributed.parallel_state": parallel_state_module,
        "vllm.distributed.device_communicators.shm_broadcast": shm_broadcast_module,
        "vllm_ascend": vllm_ascend_module,
        "vllm_ascend.patch": vllm_ascend_patch,
        "vllm_ascend.patch.worker": vllm_ascend_patch_worker,
        "vllm_ascend.distributed": vllm_ascend_distributed,
        "vllm_ascend.distributed.device_communicators": (
            vllm_ascend_device_communicators
        ),
        "vllm_ascend.distributed.device_communicators.npu_communicator": (
            npu_communicator_module
        ),
        "vllm_ascend.utils": utils_module,
    }

    previous_modules = {name: sys.modules.get(name) for name in modules}
    previous_patch_module = sys.modules.get(_PATCH_MODULE_NAME)
    previous_registry_module = sys.modules.get(_REGISTRY_MODULE_NAME)
    try:
        sys.modules.update(modules)
        sys.modules.pop(_PATCH_MODULE_NAME, None)
        sys.modules.pop(_REGISTRY_MODULE_NAME, None)
        registry_module = _load_module(_REGISTRY_MODULE_NAME, _REGISTRY_MODULE_PATH)
        patch_module = _load_module(_PATCH_MODULE_NAME, _PATCH_MODULE_PATH)
        yield SimpleNamespace(
            module=patch_module,
            registry_module=registry_module,
            torch=torch_module,
            distributed=torch_distributed,
            parallel_state_module=parallel_state_module,
            utils_module=utils_module,
            new_group_calls=new_group_calls,
            destroy_process_group=destroy_process_group,
            destroy_distributed_environment=destroy_distributed_environment,
            get_rank=get_rank,
            current_device=current_device,
            communicator_instances=communicator_instances,
            non_group_member=non_group_member,
            Backend=FakeBackend,
            vllm_distributed=vllm_distributed,
        )
    finally:
        if previous_patch_module is None:
            sys.modules.pop(_PATCH_MODULE_NAME, None)
        else:
            sys.modules[_PATCH_MODULE_NAME] = previous_patch_module
        if previous_registry_module is None:
            sys.modules.pop(_REGISTRY_MODULE_NAME, None)
        else:
            sys.modules[_REGISTRY_MODULE_NAME] = previous_registry_module
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


@pytest.fixture
def module_env():
    with _load_patch_distributed_module() as env:
        yield env


def _make_group(
    module_env,
    *,
    group_ranks: list[list[int]] | None = None,
    local_rank: int = 0,
    backend: str | FakeBackend = "hccl",
    use_device_communicator: bool = False,
    use_message_queue_broadcaster: bool = False,
    group_name: str | None = None,
):
    return module_env.module.GroupCoordinatorPatch(
        group_ranks=group_ranks or [[0, 1]],
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_device_communicator=use_device_communicator,
        use_message_queue_broadcaster=use_message_queue_broadcaster,
        group_name=group_name,
    )


def _calls_with_backend(module_env, backend: str) -> list[dict[str, object]]:
    return [call_entry for call_entry in module_env.new_group_calls if call_entry["backend"] == backend]


def test_group_coordinator_is_patched(module_env):
    assert (
        module_env.parallel_state_module.GroupCoordinator
        is module_env.module.GroupCoordinatorPatch
    )


def test_same_hccl_group_reuses_device_pg_once(module_env):
    first = _make_group(
        module_env,
        backend=module_env.Backend("hccl"),
        group_name="tp",
    )
    second = _make_group(module_env, backend="hccl", group_name="world")

    hccl_calls = _calls_with_backend(module_env, "hccl")
    gloo_calls = _calls_with_backend(module_env, "gloo")

    assert len(hccl_calls) == 1
    assert len(gloo_calls) == 2
    assert first.device_group is second.device_group


def test_same_hccl_group_reuses_with_realistic_options_object(module_env):
    module_env.utils_module.create_hccl_pg_options.return_value = (
        RealisticFakeHcclOptions(hccl_config={"hccl_buffer_size": 200})
    )

    first = _make_group(module_env, backend="hccl", group_name="tp")
    second = _make_group(module_env, backend="hccl", group_name="world")

    hccl_calls = _calls_with_backend(module_env, "hccl")
    gloo_calls = _calls_with_backend(module_env, "gloo")

    assert len(hccl_calls) == 1
    assert len(gloo_calls) == 2
    assert first.device_group is second.device_group


def test_eplb_stays_isolated_from_ep_even_when_pg_options_match(module_env):
    first = _make_group(module_env, group_name="ep")
    second = _make_group(module_env, group_name="eplb")

    hccl_calls = _calls_with_backend(module_env, "hccl")

    assert len(hccl_calls) == 2
    assert first.device_group is not second.device_group


def test_mc2_stays_isolated_from_ep_even_when_pg_options_match(module_env):
    first = _make_group(module_env, group_name="ep")
    second = _make_group(module_env, group_name="mc2")

    hccl_calls = _calls_with_backend(module_env, "hccl")

    assert len(hccl_calls) == 2
    assert first.device_group is not second.device_group


def test_dynamic_eplb_stays_separate_from_ep_when_pg_options_differ(module_env):
    default_hccl_pg_options = (
        module_env.utils_module.create_hccl_pg_options.return_value
    )

    def fake_create_hccl_pg_options(group_name: str):
        if group_name == "dynamic_eplb":
            return {"hccl_config": {"hccl_buffer_size": 512}}
        return default_hccl_pg_options

    module_env.utils_module.create_hccl_pg_options.side_effect = (
        fake_create_hccl_pg_options
    )

    first = _make_group(module_env, group_name="ep")
    second = _make_group(module_env, group_name="dynamic_eplb")

    hccl_calls = _calls_with_backend(module_env, "hccl")

    assert len(hccl_calls) == 2
    assert first.device_group is not second.device_group


def test_unknown_groups_share_by_default_when_ranks_and_options_match(module_env):
    first = _make_group(module_env, group_name="fc3_quant_x")
    second = _make_group(module_env, group_name="fc3_quant_y")

    hccl_calls = _calls_with_backend(module_env, "hccl")

    assert module_env.module._resolve_reuse_domain("fc3_quant_x:0") == "shared"
    assert len(hccl_calls) == 1
    assert first.device_group is second.device_group


def test_hccl_pg_options_are_recreated_for_each_group_ranks_entry(module_env):
    _make_group(
        module_env,
        group_ranks=[[0], [1]],
        group_name="tp",
    )

    assert module_env.utils_module.create_hccl_pg_options.call_count == 2


def test_destroy_releases_all_acquired_keys_in_reverse_order(module_env):
    group = _make_group(
        module_env,
        group_ranks=[[0, 1], [2, 3]],
        group_name="tp",
        use_device_communicator=True,
        use_message_queue_broadcaster=True,
    )
    release_mock = MagicMock(wraps=module_env.module._HCCL_PG_REGISTRY.release)
    module_env.module._HCCL_PG_REGISTRY.release = release_mock

    cpu_group = group.cpu_group
    shared_device_group = group.device_group
    acquired_keys = list(group._acquired_hccl_keys)

    group.destroy()
    group.destroy()

    assert len(acquired_keys) == 2
    assert release_mock.call_args_list == [call(acquired_keys[1]), call(acquired_keys[0])]
    assert module_env.destroy_process_group.call_args_list == [call(cpu_group), call(shared_device_group)]
    assert group.device_communicator is None
    assert group.mq_broadcaster is None
    assert not hasattr(group, "cpu_group")
    assert not hasattr(group, "device_group")
    assert group._acquired_hccl_keys == []


def test_failed_cpu_group_init_rolls_back_acquired_hccl_keys(module_env):
    original_new_group = module_env.distributed.new_group
    release_mock = MagicMock(wraps=module_env.module._HCCL_PG_REGISTRY.release)
    module_env.module._HCCL_PG_REGISTRY.release = release_mock

    def failing_new_group(ranks, backend, pg_options=None):
        if str(backend) == "gloo":
            raise RuntimeError("gloo failed")
        return original_new_group(ranks, backend, pg_options)

    module_env.distributed.new_group = failing_new_group
    hccl_key = module_env.registry_module.make_hccl_pg_key(
        [0, 1],
        "hccl",
        module_env.utils_module.create_hccl_pg_options.return_value,
        reuse_domain="shared",
    )

    with pytest.raises(RuntimeError, match="gloo failed"):
        _make_group(
            module_env,
            group_ranks=[[0, 1]],
            group_name="tp",
        )

    assert release_mock.call_args_list == [call(hccl_key)]
    assert module_env.module._HCCL_PG_REGISTRY._entries == {}


def test_failed_device_communicator_init_releases_all_keys_in_reverse_order(
    module_env,
):
    release_mock = MagicMock(wraps=module_env.module._HCCL_PG_REGISTRY.release)
    module_env.module._HCCL_PG_REGISTRY.release = release_mock
    module_env.module.NPUCommunicator = MagicMock(
        side_effect=RuntimeError("communicator failed")
    )

    key_a = module_env.registry_module.make_hccl_pg_key(
        [0, 1],
        "hccl",
        module_env.utils_module.create_hccl_pg_options.return_value,
        reuse_domain="shared",
    )
    key_b = module_env.registry_module.make_hccl_pg_key(
        [2, 3],
        "hccl",
        module_env.utils_module.create_hccl_pg_options.return_value,
        reuse_domain="shared",
    )

    with pytest.raises(RuntimeError, match="communicator failed"):
        _make_group(
            module_env,
            group_ranks=[[0, 1], [2, 3]],
            group_name="tp",
            use_device_communicator=True,
        )

    assert release_mock.call_args_list == [call(key_b), call(key_a)]
    assert module_env.module._HCCL_PG_REGISTRY._entries == {}


def test_shared_hccl_group_is_destroyed_only_after_last_coordinator(module_env):
    first = _make_group(
        module_env,
        group_ranks=[[0, 1]],
        group_name="tp",
    )
    second = _make_group(
        module_env,
        group_ranks=[[0, 1]],
        group_name="world",
    )

    cpu_group_first = first.cpu_group
    cpu_group_second = second.cpu_group
    shared_device_group = first.device_group

    assert shared_device_group is second.device_group

    first.destroy()

    assert module_env.destroy_process_group.call_args_list == [call(cpu_group_first)]

    second.destroy()

    assert module_env.destroy_process_group.call_args_list == [
        call(cpu_group_first),
        call(cpu_group_second),
        call(shared_device_group),
    ]


def test_destroy_distributed_environment_clears_registry_before_reinit(module_env):
    group = _make_group(
        module_env,
        group_ranks=[[0, 1]],
        group_name="tp",
    )
    first_device_group = group.device_group
    call_observations: list[int] = []

    def record_destroy():
        call_observations.append(len(module_env.module._HCCL_PG_REGISTRY._entries))
        return "destroyed"

    module_env.destroy_distributed_environment.side_effect = record_destroy

    assert len(_calls_with_backend(module_env, "hccl")) == 1
    assert (
        module_env.parallel_state_module.destroy_distributed_environment
        is module_env.vllm_distributed.destroy_distributed_environment
    )

    result = module_env.parallel_state_module.destroy_distributed_environment()

    assert result == "destroyed"
    assert call_observations == [1]
    assert module_env.module._HCCL_PG_REGISTRY._entries == {}

    second_group = _make_group(
        module_env,
        group_ranks=[[0, 1]],
        group_name="tp",
    )

    assert len(_calls_with_backend(module_env, "hccl")) == 2
    assert second_group.device_group is not first_device_group


def test_destroy_cleans_up_fail_closed_hccl_device_group(module_env):
    module_env.utils_module.create_hccl_pg_options.return_value = {
        "hccl_config": {"hccl_buffer_size": 200},
        "non_default_field": 7,
    }
    group = _make_group(
        module_env,
        group_ranks=[[0, 1], [2, 3]],
        group_name="tp",
    )

    cpu_group = group.cpu_group
    device_group = group.device_group

    assert group._acquired_hccl_keys == []

    group.destroy()
    group.destroy()

    assert module_env.destroy_process_group.call_args_list == [
        call(cpu_group),
        call(device_group),
    ]
    assert group._acquired_hccl_keys == []
    assert not hasattr(group, "cpu_group")
    assert not hasattr(group, "device_group")


def test_non_hccl_destroy_path_destroys_device_group_directly(module_env):
    group = _make_group(
        module_env,
        backend="nccl",
        group_name="tp",
    )

    cpu_group = group.cpu_group
    device_group = group.device_group

    group.destroy()
    group.destroy()

    assert module_env.destroy_process_group.call_args_list == [
        call(cpu_group),
        call(device_group),
    ]
    assert not hasattr(group, "cpu_group")
    assert not hasattr(group, "device_group")


def test_all_to_all_returns_input_when_world_size_is_one(module_env):
    group = _make_group(module_env)
    group.world_size = 1
    input_tensor = module_env.torch.randn(2, 3)

    assert group.all_to_all(input_tensor) is input_tensor


def test_all_to_all_raises_assertion_on_invalid_scatter_dim(module_env):
    group = _make_group(module_env)
    input_tensor = module_env.torch.randn(2, 3)

    with pytest.raises(AssertionError, match="Invalid scatter dim"):
        group.all_to_all(input_tensor, scatter_dim=2)


def test_all_to_all_raises_assertion_on_invalid_gather_dim(module_env):
    group = _make_group(module_env)
    input_tensor = module_env.torch.randn(2, 3)

    with pytest.raises(AssertionError, match="Invalid gather dim"):
        group.all_to_all(input_tensor, gather_dim=2)


def test_all_to_all_calls_device_communicator_with_correct_args(module_env):
    group = _make_group(module_env)
    communicator = MagicMock()
    communicator.all_to_all.return_value = "ok"
    group.device_communicator = communicator

    input_tensor = module_env.torch.randn(2, 3)
    output = group.all_to_all(
        input_tensor,
        scatter_dim=0,
        gather_dim=1,
        scatter_sizes=[1, 1],
        gather_sizes=[1, 1],
    )

    communicator.all_to_all.assert_called_once_with(
        input_tensor,
        0,
        1,
        [1, 1],
        [1, 1],
    )
    assert output == "ok"


def test_all_to_all_calls_device_communicator_without_sizes(module_env):
    group = _make_group(module_env)
    communicator = MagicMock()
    communicator.all_to_all.return_value = "ok"
    group.device_communicator = communicator

    input_tensor = module_env.torch.randn(2, 3)
    output = group.all_to_all(input_tensor, scatter_dim=0, gather_dim=1)

    communicator.all_to_all.assert_called_once_with(input_tensor, 0, 1, None, None)
    assert output == "ok"
