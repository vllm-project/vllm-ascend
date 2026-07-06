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
"""Mock heavy dependencies (torch, vllm, etc.) for ascend_store unit tests.

IMPORTANT: This module MUST be imported before any vllm_ascend or vllm
imports in each test file.

Usage at the top of each test file:
    import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
"""

import importlib.util
import logging
import os
import sys
import types
from typing import Any
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock torch / torch_npu
# ---------------------------------------------------------------------------
if "torch" not in sys.modules and importlib.util.find_spec("torch") is None:
    _torch = types.ModuleType("torch")
    _torch.Tensor = MagicMock  # type: ignore[attr-defined]
    _torch.bool = "bool"  # type: ignore[attr-defined]
    _torch.float16 = "float16"  # type: ignore[attr-defined]
    _torch.float32 = "float32"  # type: ignore[attr-defined]
    _torch.zeros = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]
    _torch.sum = MagicMock(return_value=0)  # type: ignore[attr-defined]
    _torch.device = MagicMock()  # type: ignore[attr-defined]
    _torch.distributed = MagicMock()  # type: ignore[attr-defined]
    _npu = MagicMock()
    _npu.Event = MagicMock
    _npu.current_device = MagicMock(return_value=0)
    _npu.set_device = MagicMock()
    _torch.npu = _npu  # type: ignore[attr-defined]
    sys.modules["torch"] = _torch
    sys.modules["torch.distributed"] = _torch.distributed  # type: ignore[attr-defined]

if "torch_npu" not in sys.modules:
    sys.modules["torch_npu"] = MagicMock()
    sys.modules["torch_npu._inductor"] = MagicMock()

# ---------------------------------------------------------------------------
# Mock vllm modules
# ---------------------------------------------------------------------------
_MOCK_VLLM_DEPS = importlib.util.find_spec("vllm") is None
_vllm_mock_modules = [
    "vllm",
    "vllm.config",
    "vllm.distributed",
    "vllm.distributed.kv_events",
    "vllm.distributed.kv_transfer",
    "vllm.distributed.kv_transfer.kv_connector",
    "vllm.distributed.kv_transfer.kv_connector.factory",
    "vllm.distributed.kv_transfer.kv_connector.v1",
    "vllm.distributed.kv_transfer.kv_connector.v1.base",
    "vllm.distributed.parallel_state",
    "vllm.envs",
    "vllm.forward_context",
    "vllm.logger",
    "vllm.model_executor",
    "vllm.model_executor.layers",
    "vllm.model_executor.layers.linear",
    "vllm.model_executor.layers.quantization",
    "vllm.platforms",
    "vllm.utils",
    "vllm.utils.hashing",
    "vllm.utils.math_utils",
    "vllm.utils.network_utils",
    "vllm.v1",
    "vllm.v1.attention",
    "vllm.v1.attention.backend",
    "vllm.v1.core",
    "vllm.v1.core.block_pool",
    "vllm.v1.core.kv_cache_manager",
    "vllm.v1.core.kv_cache_utils",
    "vllm.v1.core.sched",
    "vllm.v1.core.sched.output",
    "vllm.v1.core.single_type_kv_cache_manager",
    "vllm.v1.kv_cache_interface",
    "vllm.v1.kv_cache_spec_registry",
    "vllm.v1.outputs",
    "vllm.v1.request",
    "vllm.v1.serial_utils",
]
if _MOCK_VLLM_DEPS:
    for _mod_name in _vllm_mock_modules:
        if _mod_name not in sys.modules:
            sys.modules[_mod_name] = MagicMock()

if _MOCK_VLLM_DEPS:
    sys.modules["vllm.utils.math_utils"].cdiv = lambda a, b: -(-a // b)  # type: ignore[attr-defined]
    sys.modules["vllm.logger"].logger = logging.getLogger("vllm")  # type: ignore[attr-defined]

_base_mod: Any = (
    sys.modules["vllm.distributed.kv_transfer.kv_connector.v1.base"] if _MOCK_VLLM_DEPS else types.SimpleNamespace()
)
_base_mod.KVConnectorBase_V1 = type("KVConnectorBase_V1", (), {"__init__": lambda self, **kw: None})  # type: ignore[attr-defined]
_base_mod.KVConnectorMetadata = type("KVConnectorMetadata", (), {})  # type: ignore[attr-defined]
_base_mod.KVConnectorWorkerMetadata = type("KVConnectorWorkerMetadata", (), {})  # type: ignore[attr-defined]
_base_mod.KVConnectorRole = MagicMock()  # type: ignore[attr-defined]
_base_mod.KVConnectorRole.SCHEDULER = "SCHEDULER"
_base_mod.KVConnectorRole.WORKER = "WORKER"

_events_mod: Any = sys.modules["vllm.distributed.kv_events"] if _MOCK_VLLM_DEPS else types.SimpleNamespace()
_events_mod.KVCacheEvent = type("KVCacheEvent", (), {})  # type: ignore[attr-defined]
_events_mod.KVConnectorKVEvents = type("KVConnectorKVEvents", (), {})  # type: ignore[attr-defined]


class _FakeAggregator:
    def __init__(self, *args, **kwargs):
        self._mock = MagicMock()

    def __getattr__(self, name):
        return getattr(self._mock, name)


_events_mod.KVEventAggregator = _FakeAggregator  # type: ignore[attr-defined]
_events_mod.BlockStored = type(  # type: ignore[attr-defined]
    "BlockStored",
    (),
    {"__init__": lambda self, **kwargs: self.__dict__.update(kwargs)},
)

_kv_cache_utils_mod: Any = sys.modules["vllm.v1.core.kv_cache_utils"] if _MOCK_VLLM_DEPS else types.SimpleNamespace()
_kv_cache_utils_mod.BlockHash = bytes  # type: ignore[attr-defined]
_kv_cache_utils_mod.maybe_convert_block_hash = lambda x: x  # type: ignore[attr-defined]


class _FakeKVCacheBlock:
    def __init__(self, block_id=0, **kwargs):
        self.block_id = block_id
        self.__dict__.update(kwargs)


class _FakeKVCacheSpec:
    def __init__(self, block_size=16, **kwargs):
        self.block_size = block_size
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __eq__(self, other):
        return type(self) is type(other) and self.__dict__ == getattr(other, "__dict__", {})

    def copy_with_new_block_size(self, block_size):
        kwargs = self.__dict__.copy()
        kwargs["block_size"] = block_size
        return type(self)(**kwargs)

    @property
    def page_size_bytes(self):
        num_kv_heads = getattr(self, "num_kv_heads", 1)
        head_size = getattr(self, "head_size", 1)
        dtype = getattr(self, "dtype", None)
        dtype_size = getattr(dtype, "itemsize", None)
        if dtype_size is None and dtype is not None and hasattr(dtype, "element_size"):
            dtype_size = dtype.element_size()
        return self.block_size * num_kv_heads * head_size * int(dtype_size or 1) * 2


class _FakeFullAttentionSpec(_FakeKVCacheSpec):
    pass


class _FakeSlidingWindowSpec(_FakeKVCacheSpec):
    def __init__(self, block_size=16, sliding_window=32, **kwargs):
        super().__init__(block_size=block_size, sliding_window=sliding_window, **kwargs)


class _FakeMambaSpec(_FakeKVCacheSpec):
    def __init__(self, block_size=16, **kwargs):
        super().__init__(block_size=block_size, **kwargs)
        self.num_speculative_blocks = getattr(self, "num_speculative_blocks", 0)


class _FakeUniformTypeKVCacheSpecs(_FakeKVCacheSpec):
    def __init__(self, block_size=16, kv_cache_specs=None, **kwargs):
        super().__init__(block_size=block_size, **kwargs)
        self.kv_cache_specs = kv_cache_specs or {}

    @classmethod
    def from_specs(cls, kv_cache_specs):
        if not kv_cache_specs:
            return None
        first_spec = next(iter(kv_cache_specs.values()))
        return cls(
            block_size=getattr(first_spec, "block_size", 16),
            kv_cache_specs=kv_cache_specs,
        )


class _FakeKVCacheGroupSpec:
    def __init__(self, layer_names=None, kv_cache_spec=None, is_eagle_group=False):
        self.layer_names = layer_names or []
        self.kv_cache_spec = kv_cache_spec or _FakeFullAttentionSpec()
        self.is_eagle_group = is_eagle_group


class _FakeKVCacheConfig:
    def __init__(self, num_blocks=1, kv_cache_tensors=None, kv_cache_groups=None):
        self.num_blocks = num_blocks
        self.kv_cache_tensors = kv_cache_tensors or []
        self.kv_cache_groups = kv_cache_groups or []


_kv_cache_utils_mod.KVCacheBlock = _FakeKVCacheBlock  # type: ignore[attr-defined]
_kv_cache_utils_mod.BlockHashList = list  # type: ignore[attr-defined]


class _FakeBlockPool:
    def __init__(self, *args, **kwargs):
        self.null_block = _FakeKVCacheBlock(block_id=0)
        self._next_block_id = 1

    def get_new_blocks(self, num_blocks):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(_FakeKVCacheBlock(block_id=self._next_block_id))
            self._next_block_id += 1
        return blocks


if _MOCK_VLLM_DEPS:
    sys.modules["vllm.v1.core.block_pool"].BlockPool = _FakeBlockPool  # type: ignore[attr-defined]


class _FakeSingleTypeKVCacheManager:
    def __init__(self, *args, **kwargs):
        self._mock = MagicMock()

    def __getattr__(self, name):
        return getattr(self._mock, name)

    @classmethod
    def reachable_block_mask(
        cls,
        start_block,
        end_block,
        alignment_tokens,
        kv_cache_spec,
        use_eagle,
        retention_interval=None,
        num_prompt_tokens=None,
    ):
        return None

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes,
        max_length,
        kv_cache_group_ids,
        block_pool,
        kv_cache_spec,
        drop_eagle_block=False,
        alignment_tokens=16,
        dcp_world_size=1,
        pcp_world_size=1,
    ):
        computed: tuple[list[object], ...] = tuple([] for _ in kv_cache_group_ids)
        max_blocks = max_length // kv_cache_spec.block_size
        for block_hash in list(block_hashes)[:max_blocks]:
            cached = block_pool.get_cached_block(block_hash, kv_cache_group_ids)
            if not cached:
                break
            for blocks, block in zip(computed, cached):
                blocks.append(block)
        if drop_eagle_block and computed and computed[0]:
            for blocks in computed:
                blocks.pop()
        return computed


class _FakeSlidingWindowManager(_FakeSingleTypeKVCacheManager):
    @classmethod
    def reachable_block_mask(
        cls,
        start_block,
        end_block,
        alignment_tokens,
        kv_cache_spec,
        use_eagle,
        retention_interval=None,
        num_prompt_tokens=None,
    ):
        if alignment_tokens is None:
            return None
        per_segment = max(alignment_tokens // kv_cache_spec.block_size, 1)
        return [(idx + 1) % per_segment == 0 for idx in range(start_block, end_block)]


_single_type_mod: Any = (
    sys.modules["vllm.v1.core.single_type_kv_cache_manager"] if _MOCK_VLLM_DEPS else types.SimpleNamespace()
)
_single_type_mod.SingleTypeKVCacheManager = _FakeSingleTypeKVCacheManager  # type: ignore[attr-defined]
_single_type_mod.FullAttentionManager = _FakeSingleTypeKVCacheManager  # type: ignore[attr-defined]
_single_type_mod.SlidingWindowManager = _FakeSlidingWindowManager  # type: ignore[attr-defined]
_single_type_mod.MambaManager = _FakeSingleTypeKVCacheManager  # type: ignore[attr-defined]
_single_type_mod.spec_manager_map = {  # type: ignore[attr-defined]
    _FakeFullAttentionSpec: _FakeSingleTypeKVCacheManager,
    _FakeSlidingWindowSpec: _FakeSlidingWindowManager,
    _FakeMambaSpec: _FakeSingleTypeKVCacheManager,
}

_kv_interface_mod: Any = sys.modules["vllm.v1.kv_cache_interface"] if _MOCK_VLLM_DEPS else types.SimpleNamespace()
_kv_interface_mod.KVCacheSpec = _FakeKVCacheSpec  # type: ignore[attr-defined]
_kv_interface_mod.FullAttentionSpec = _FakeFullAttentionSpec  # type: ignore[attr-defined]
_kv_interface_mod.SlidingWindowSpec = _FakeSlidingWindowSpec  # type: ignore[attr-defined]
_kv_interface_mod.MambaSpec = _FakeMambaSpec  # type: ignore[attr-defined]
_kv_interface_mod.UniformTypeKVCacheSpecs = _FakeUniformTypeKVCacheSpecs  # type: ignore[attr-defined]
_kv_interface_mod.KVCacheGroupSpec = _FakeKVCacheGroupSpec  # type: ignore[attr-defined]
_kv_interface_mod.KVCacheConfig = _FakeKVCacheConfig  # type: ignore[attr-defined]


class _FakeKVCacheSpecRegistry:
    @classmethod
    def get_manager_class(cls, kv_cache_spec):
        if isinstance(kv_cache_spec, _FakeSlidingWindowSpec):
            return _FakeSlidingWindowManager
        return _FakeSingleTypeKVCacheManager


if _MOCK_VLLM_DEPS:
    sys.modules["vllm.v1.kv_cache_spec_registry"].KVCacheSpecRegistry = _FakeKVCacheSpecRegistry  # type: ignore[attr-defined]

_sched_output_mod: Any = sys.modules["vllm.v1.core.sched.output"] if _MOCK_VLLM_DEPS else types.SimpleNamespace()
_sched_output_mod.NewRequestData = MagicMock  # type: ignore[attr-defined]

if _MOCK_VLLM_DEPS:
    sys.modules["vllm.envs"].VLLM_RPC_BASE_PATH = "/tmp/vllm_rpc"  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Mock external backends
# ---------------------------------------------------------------------------
for _mod_name in [
    "mooncake",
    "mooncake.engine",
    "mooncake.store",
    "memcache_hybrid",
    "yr",
    "yr.datasystem",
    "yr.datasystem.hetero_client",
    "yr.datasystem.kv_client",
    "yr.datasystem.object_client",
    "zmq",
]:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

# ---------------------------------------------------------------------------
# Mock vllm_ascend transitive imports
# ---------------------------------------------------------------------------


def _make_pkg(name, path=""):
    mod = types.ModuleType(name)
    mod.__path__ = [path]  # type: ignore[attr-defined]
    mod.__package__ = name  # type: ignore[attr-defined]
    return mod


for _pkg in ["vllm_ascend", "vllm_ascend.distributed"]:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = _make_pkg(_pkg)

_distributed_utils = types.ModuleType("vllm_ascend.distributed.utils")
_distributed_utils.get_decode_context_model_parallel_rank = MagicMock(  # type: ignore[attr-defined]
    return_value=0
)
_distributed_utils.get_decode_context_model_parallel_world_size = MagicMock(  # type: ignore[attr-defined]
    return_value=1
)
sys.modules["vllm_ascend.distributed.utils"] = _distributed_utils

_kv_transfer_init = _make_pkg("vllm_ascend.distributed.kv_transfer")
_kv_transfer_init.register_connector = MagicMock()  # type: ignore[attr-defined]
sys.modules["vllm_ascend.distributed.kv_transfer"] = _kv_transfer_init

_kv_utils_pkg = _make_pkg("vllm_ascend.distributed.kv_transfer.utils")
sys.modules["vllm_ascend.distributed.kv_transfer.utils"] = _kv_utils_pkg
sys.modules["vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine"] = MagicMock()

_kv_pool_pkg = _make_pkg("vllm_ascend.distributed.kv_transfer.kv_pool")
sys.modules["vllm_ascend.distributed.kv_transfer.kv_pool"] = _kv_pool_pkg

_ascend_store_real_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "..",
    "vllm_ascend",
    "distributed",
    "kv_transfer",
    "kv_pool",
    "ascend_store",
)
_ascend_store_pkg = _make_pkg(
    "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store",
    os.path.abspath(_ascend_store_real_path),
)
sys.modules["vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store"] = _ascend_store_pkg

_backend_pkg = _make_pkg(
    "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend",
    os.path.join(os.path.abspath(_ascend_store_real_path), "backend"),
)
sys.modules["vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend"] = _backend_pkg

if "vllm_ascend.utils" not in sys.modules or not hasattr(sys.modules["vllm_ascend.utils"], "AscendDeviceType"):
    _ascend_utils = MagicMock()
    _ascend_utils.AscendDeviceType = MagicMock()
    _ascend_utils.get_ascend_device_type = MagicMock()
    sys.modules["vllm_ascend.utils"] = _ascend_utils
