# SPDX-License-Identifier: Apache-2.0
"""Checkpoint API compatibility without importing KDA's vendor kernels."""

import ast
import builtins
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import vllm
import vllm.v1.attention.backend as attention_backend
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.kv_cache_interface import (
    MambaSpec,
    get_mamba_prefill_checkpoint_position,
    is_mamba_prefill_checkpoint_valid,
)

from vllm_ascend.patch.platform.patch_kv_cache_coordinator import _effective_eagle_block_drop
from vllm_ascend.patch.platform.patch_mamba_manager import AscendMambaManager
from vllm_ascend.patch.worker import patch_mamba_checkpoint as mod

LEGACY_KDA_MODULE = "vllm.models.kimi_k3.nvidia.kda_metadata"
GENERIC_CHECKPOINT_MODULE = "vllm.model_executor.layers.mamba.checkpoint"
CHECKPOINT_BLOCK_SIZE = 384
CHECKPOINT_ROLES = [
    pytest.param(None, False, id="standalone"),
    pytest.param(SimpleNamespace(is_kv_producer=True, is_kv_consumer=False), False, id="producer"),
    pytest.param(SimpleNamespace(is_kv_producer=False, is_kv_consumer=True), True, id="consumer"),
    pytest.param(SimpleNamespace(is_kv_producer=True, is_kv_consumer=True), True, id="kv_both"),
]


def _config(role, *, use_eagle=True, mode="align"):
    speculative_config = SimpleNamespace(
        use_eagle_block_drop=lambda: use_eagle,
        use_eagle=lambda: use_eagle,
        num_speculative_tokens=5,
    )
    return SimpleNamespace(
        kv_transfer_config=role,
        speculative_config=speculative_config,
        cache_config=SimpleNamespace(mamba_cache_mode=mode, prefix_match_unit=CHECKPOINT_BLOCK_SIZE),
    )


def _lightweight_builder_type(*, module=LEGACY_KDA_MODULE, name="KimiK3KDAMetadataBuilder"):
    class LightweightAttentionMetadataBuilder:
        def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
            self.kv_cache_spec = kv_cache_spec
            self.layer_names = layer_names
            self.vllm_config = vllm_config
            self.device = device
            self.base_constructor_calls = 1

    def initialize(self, kv_cache_spec, layer_names, vllm_config, device):
        LightweightAttentionMetadataBuilder.__init__(self, kv_cache_spec, layer_names, vllm_config, device)
        # These are initialized from the original config by the real GDN/KDA
        # constructors after their shared AttentionMetadataBuilder parent.
        self.speculative_config = vllm_config.speculative_config
        self.num_spec = self.speculative_config.num_speculative_tokens if self.speculative_config is not None else 0
        self.use_spec_decode = self.speculative_config is not None
        self.use_recoverssm = True
        self.recoverssm_context = object()
        self.recoverssm_num_accepted_tokens = torch.ones(1, dtype=torch.int32)
        self.staging_state = object()

    def build(self):
        speculative_config = self.vllm_config.speculative_config
        return speculative_config is not None and speculative_config.use_eagle_block_drop()

    builder_type = type(
        name,
        (LightweightAttentionMetadataBuilder,),
        {"__module__": module, "__init__": initialize, "build": build},
    )
    return builder_type, LightweightAttentionMetadataBuilder


@pytest.mark.parametrize("role, expected_drop", CHECKPOINT_ROLES)
@pytest.mark.parametrize("derived", [False, True], ids=["upstream", "ascend-subclass"])
def test_legacy_parent_hook_changes_checkpoint_policy_only(role, expected_drop, derived):
    builder_type, builder_base = _lightweight_builder_type()
    mod.install_legacy_kda_checkpoint_policy(builder_base)
    patched_init = builder_base.__init__
    mod.install_legacy_kda_checkpoint_policy(builder_base)
    assert builder_base.__init__ is patched_init
    if derived:
        builder_type = type("AscendKDAMetadataBuilder", (builder_type,), {"__module__": __name__})
    config = _config(role)
    spec = SimpleNamespace(num_prefill_checkpoint_blocks=1)
    builder = builder_type(spec, ["kda"], config, torch.device("cpu"))
    state_objects = (builder.recoverssm_context, builder.recoverssm_num_accepted_tokens, builder.staging_state)

    for _ in range(3):
        assert builder.build() is expected_drop
        assert builder.speculative_config is config.speculative_config
        assert builder.num_spec == 5
        assert builder.use_spec_decode is True
        assert builder.use_recoverssm is True
        assert all(
            actual is original
            for actual, original in zip(
                (builder.recoverssm_context, builder.recoverssm_num_accepted_tokens, builder.staging_state),
                state_objects,
            )
        )
    assert builder.base_constructor_calls == 1
    assert config.speculative_config.use_eagle_block_drop() is True
    assert config.speculative_config.use_eagle() is True
    assert (builder.vllm_config is config) is expected_drop


@pytest.mark.parametrize(
    "config, checkpoint_blocks",
    [
        pytest.param(_config(None, use_eagle=False), 1, id="no-mtp-drop"),
        pytest.param(_config(None), 0, id="internal-disabled"),
        pytest.param(_config(None, mode="none"), 1, id="align-disabled"),
    ],
)
def test_legacy_policy_does_not_wrap_inactive_checkpoint_paths(config, checkpoint_blocks):
    builder_type, builder_base = _lightweight_builder_type()
    mod.install_legacy_kda_checkpoint_policy(builder_base)
    builder = builder_type(SimpleNamespace(num_prefill_checkpoint_blocks=checkpoint_blocks), [], config, "cpu")
    assert builder.vllm_config is config
    assert builder.speculative_config is config.speculative_config


def test_legacy_policy_without_speculation_keeps_original_config():
    config = _config(None)
    config.speculative_config = None
    builder_type, builder_base = _lightweight_builder_type()
    mod.install_legacy_kda_checkpoint_policy(builder_base)
    builder = builder_type(SimpleNamespace(num_prefill_checkpoint_blocks=1), [], config, "cpu")
    assert builder.vllm_config is config
    assert builder.build() is False


@pytest.mark.parametrize(
    "module, name",
    [("unrelated.backend", "KimiK3KDAMetadataBuilder"), (LEGACY_KDA_MODULE, "UnrelatedMetadataBuilder")],
)
def test_legacy_parent_hook_is_narrowly_scoped(module, name):
    builder_type, builder_base = _lightweight_builder_type(module=module, name=name)
    mod.install_legacy_kda_checkpoint_policy(builder_base)
    config = _config(None)
    builder = builder_type(SimpleNamespace(num_prefill_checkpoint_blocks=1), [], config, "cpu")
    assert builder.vllm_config is config
    assert builder.build() is True


def _load_patch_with_checkpoint_import(monkeypatch, checkpoint_module=None, missing_dependency=None):
    _, builder_base = _lightweight_builder_type()
    monkeypatch.setattr(attention_backend, "AttentionMetadataBuilder", builder_base)
    original_import = builtins.__import__

    def controlled_import(name, *args, **kwargs):
        if name == GENERIC_CHECKPOINT_MODULE:
            if checkpoint_module is not None:
                return checkpoint_module
            missing = missing_dependency or GENERIC_CHECKPOINT_MODULE
            raise ModuleNotFoundError(f"No module named '{missing}'", name=missing)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", controlled_import)
    spec = importlib.util.spec_from_file_location("_ascend_checkpoint_compat_test", mod.__file__)
    assert spec is not None and spec.loader is not None
    patch_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, patch_module)
    spec.loader.exec_module(patch_module)
    return patch_module, builder_base


def test_missing_generic_api_imports_without_loading_vendor_backends(monkeypatch):
    loaded_before = set(sys.modules)
    patch_module, builder_base = _load_patch_with_checkpoint_import(monkeypatch)
    assert patch_module.HAS_GENERIC_CHECKPOINT_BUILDER is False
    assert not hasattr(patch_module, "_ORIGINAL_BUILD")
    assert getattr(builder_base.__init__, "_ascend_legacy_kda_checkpoint_policy", False)
    added = set(sys.modules) - loaded_before
    assert LEGACY_KDA_MODULE not in added
    assert not any(name.startswith(("fla.", "flashinfer.", "vllm_ascend.ops.kimi_kda")) for name in added)


def test_generic_api_transitive_missing_dependency_is_not_treated_as_old_vllm(monkeypatch):
    with pytest.raises(ModuleNotFoundError) as error:
        _load_patch_with_checkpoint_import(monkeypatch, missing_dependency="checkpoint_dependency")
    assert error.value.name == "checkpoint_dependency"


def test_present_generic_api_installs_generic_policy_only(monkeypatch):
    class GenericBuilder:
        vllm_config: SimpleNamespace
        kv_cache_spec: SimpleNamespace

        def build(self, metadata, request_rows):
            return self.vllm_config.speculative_config.use_eagle_block_drop()

    generic_module = ModuleType(GENERIC_CHECKPOINT_MODULE)
    generic_module.__dict__.update(
        MambaPrefillCheckpointBuilder=GenericBuilder,
        MambaPrefillCheckpointMetadata=SimpleNamespace,
    )
    patch_module, builder_base = _load_patch_with_checkpoint_import(monkeypatch, generic_module)
    assert patch_module.HAS_GENERIC_CHECKPOINT_BUILDER is True
    assert GenericBuilder.build is patch_module.build_mamba_prefill_checkpoint
    assert not getattr(builder_base.__init__, "_ascend_legacy_kda_checkpoint_policy", False)
    builder = GenericBuilder()
    config = _config(None)
    builder.vllm_config = config
    builder.kv_cache_spec = SimpleNamespace(num_prefill_checkpoint_blocks=1)
    assert builder.build(object(), [0]) is False
    assert builder.vllm_config is config
    assert config.speculative_config.use_eagle_block_drop() is True


def _legacy_checkpoint_section():
    # Read the installed vLLM source without importing the NVIDIA KDA backend.
    # Execute only its checkpoint section, not a synthetic rewrite of its math.
    source_path = Path(vllm.__file__).parent / "models/kimi_k3/nvidia/kda_metadata.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    builder = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "KimiK3KDAMetadataBuilder"
    )
    build = next(node for node in builder.body if isinstance(node, ast.FunctionDef) and node.name == "build")
    start = next(
        index
        for index, node in enumerate(build.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "checkpoint" for target in node.targets)
    )
    section = build.body[start : start + 3]
    assert isinstance(section[-1], ast.If)
    function = ast.parse(
        "def checkpoint_section(self, m):\n"
        "    query_start_loc_cpu = m.query_start_loc_cpu\n"
        "    query_start_loc = m.query_start_loc\n"
        "    num_prefills = 1\n"
        "    spec_sequence_masks_cpu = None\n"
        "    return checkpoint\n"
    ).body[0]
    assert isinstance(function, ast.FunctionDef)
    function.body[-1:-1] = section
    namespace = {
        "torch": torch,
        "cdiv": cdiv,
        "NULL_BLOCK_ID": 0,
        "get_mamba_prefill_checkpoint_position": get_mamba_prefill_checkpoint_position,
        "is_mamba_prefill_checkpoint_valid": is_mamba_prefill_checkpoint_valid,
        "async_tensor_h2d": lambda data, *, dtype, device: torch.tensor(data, dtype=dtype),
        "KDACheckpointMetadata": lambda offsets, indices: SimpleNamespace(
            checkpoint_offsets=offsets, state_indices=indices
        ),
    }
    compiled = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(compiled, str(source_path), "exec"), namespace)
    return namespace["checkpoint_section"]


@pytest.mark.skipif(mod.HAS_GENERIC_CHECKPOINT_BUILDER, reason="Legacy KDA checkpoint section replaced by generic API")
@pytest.mark.parametrize("prompt_length", [131082, 131328], ids=["unaligned", "aligned"])
@pytest.mark.parametrize(
    "role", [None, SimpleNamespace(is_kv_producer=True, is_kv_consumer=False)], ids=["standalone", "producer"]
)
def test_legacy_allocation_export_and_hash_registration_use_same_state(role, prompt_length):
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=CHECKPOINT_BLOCK_SIZE)
    spec = MambaSpec(
        block_size=CHECKPOINT_BLOCK_SIZE,
        shapes=((1,),),
        dtypes=(torch.int64,),
        mamba_cache_mode="align",
        num_speculative_blocks=7,
        num_prefill_checkpoint_blocks=1,
        prefill_checkpoint_alignment=128,
    )
    manager = AscendMambaManager(
        kv_cache_spec=spec,
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=CHECKPOINT_BLOCK_SIZE,
    )
    manager.cache_hit_alignment_tokens = CHECKPOINT_BLOCK_SIZE
    manager.skips_eagle_block_drop = True
    manager.drop_eagle_checkpoint_block = _effective_eagle_block_drop(True, role)
    request = SimpleNamespace(
        request_id="prime",
        num_prompt_tokens=prompt_length,
        num_computed_tokens=0,
        shared_prefix_boundary=0,
        block_hashes=[i.to_bytes(32, "little") for i in range(1, prompt_length // CHECKPOINT_BLOCK_SIZE + 1)],
    )
    num_blocks = manager.get_num_blocks_to_allocate(
        request_id=request.request_id,
        num_tokens=prompt_length,
        new_computed_blocks=(),
        total_computed_tokens=0,
        num_local_computed_tokens=0,
        num_tokens_main_model=prompt_length,
        prefill_end=prompt_length,
    )
    position, column = manager._checkpoints[request.request_id]
    allocated = manager.allocate_new_blocks(request.request_id, prompt_length, prompt_length)
    assert sum(not block.is_null for block in allocated) == num_blocks
    blocks = manager.req_to_blocks[request.request_id]
    checkpoint_block = blocks[column]
    builder_type, builder_base = _lightweight_builder_type()
    mod.install_legacy_kda_checkpoint_policy(builder_base)
    config = _config(role)
    builder = builder_type(spec, ["kda"], config, torch.device("cpu"))
    query_start_loc = torch.tensor([0, prompt_length], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_reqs=1,
        query_start_loc_cpu=query_start_loc,
        query_start_loc=query_start_loc,
        seq_lens_cpu_upper_bound=torch.tensor([prompt_length]),
        block_table_tensor=torch.tensor([[block.block_id for block in blocks]], dtype=torch.int64),
    )
    exported = _legacy_checkpoint_section()(builder, metadata)
    assert exported is not None
    assert position == 130944
    assert exported.checkpoint_offsets.tolist() == [position]
    assert exported.state_indices.tolist() == [checkpoint_block.block_id]
    # Synthetic CPU states verify metadata consistency, not vendor kernels.
    state = torch.zeros(pool.num_gpu_blocks, dtype=torch.int64)
    state[exported.state_indices] = exported.checkpoint_offsets.to(torch.int64)
    manager.cache_blocks(
        request,
        prompt_length // CHECKPOINT_BLOCK_SIZE * CHECKPOINT_BLOCK_SIZE,
        retention_interval=0,
        replay_boundaries=(prompt_length - 1, prompt_length),
    )
    checkpoint_hash = request.block_hashes[position // CHECKPOINT_BLOCK_SIZE - 1]
    assert pool.get_cached_block(checkpoint_hash, [0]) == [checkpoint_block]
    assert checkpoint_block.block_hash_num_tokens == position
    assert state[checkpoint_block.block_id].item() == position
    assert builder.speculative_config is config.speculative_config
    assert builder.num_spec == 5
    assert config.speculative_config.use_eagle_block_drop() is True
