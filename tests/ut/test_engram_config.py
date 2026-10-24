# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import copy
from dataclasses import FrozenInstanceError, fields, make_dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from pydantic import TypeAdapter, model_validator

from vllm_ascend.config_utils import config as config_dataclass
from vllm_ascend.engram_config import (
    AscendEngramConfig,
    resolve_engram_config,
)


@pytest.fixture
def native(monkeypatch):
    """Replace only the external vLLM schema/validation boundary."""
    import vllm.config

    @config_dataclass
    class Native:
        cpu_offload: bool = True
        dp_shared_memory: bool = False
        embedding_across_dp: bool = False

        @model_validator(mode="after")
        def _validate_shared_memory(self):
            if self.dp_shared_memory and not self.cpu_offload:
                raise ValueError("dp_shared_memory requires cpu_offload=True")
            return self

    def verify(vc):
        shared = vc.engram_config.dp_shared_memory
        parallel = vc.parallel_config
        if shared and parallel.data_parallel_size <= 1 and parallel.prefill_context_parallel_size <= 1:
            raise ValueError("dp_shared_memory requires DP or PCP peers")
        if shared and not vc.engram_config.cpu_offload:
            raise ValueError("dp_shared_memory requires cpu_offload=True")

    verification = Mock(side_effect=verify)
    monkeypatch.setattr(vllm.config, "EngramConfig", Native, raising=False)
    monkeypatch.setattr(
        vllm.config, "VllmConfig", SimpleNamespace(_resolve_and_verify_engram_config=verification), raising=False
    )
    monkeypatch.setattr("vllm_ascend.engram_config.default_engram_overlap", lambda: True)
    return SimpleNamespace(cls=Native, verify=verification)


def vc(options=None, *, native=None, v2=True, dp=4, pcp=1):
    text = SimpleNamespace(engram_layer_ids=[1, 14])
    model = SimpleNamespace(architecture="DeepseekV41ForCausalLM", hf_text_config=text, hf_config=text)
    values = dict(
        additional_config={"engram_config": options or {}},
        engram_config=native,
        model_config=model,
        speculative_config=None,
        parallel_config=SimpleNamespace(
            enable_elastic_ep=False,
            tensor_parallel_size=1,
            data_parallel_size=dp,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=pcp,
            decode_context_parallel_size=1,
            world_size_across_dp=dp * pcp,
            local_world_size=dp * pcp,
            data_parallel_size_local=1,
            use_sequence_parallel_moe=False,
        ),
        load_config=SimpleNamespace(load_format="safetensors"),
        use_v2_model_runner=v2,
    )
    cls = make_dataclass("TestVllmConfig", [(key, object) for key in values])
    return cls(**values)


def test_default_schema_and_resolved_config_are_read_only(native):
    assert AscendEngramConfig().enabled is False
    resolved = resolve_engram_config(vc())
    with pytest.raises(FrozenInstanceError):
        resolved.enabled = True
    with pytest.raises(ValueError):
        AscendEngramConfig(group_size=4)


@pytest.mark.parametrize("options", [{}, {"enabled": False}, {"enabled": "false", "backend": "elastic"}])
def test_disabled_never_loads_upstream_or_device_policy(monkeypatch, native, options):
    monkeypatch.setattr("vllm_ascend.engram_config.default_engram_overlap", Mock(side_effect=AssertionError("device")))
    config = SimpleNamespace(additional_config={"engram_config": options}, engram_config=native.cls())
    resolved = resolve_engram_config(config)
    assert not resolved.enabled and not resolved.overlap and resolved.group_size is None
    assert config.engram_config is None
    native.verify.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        [],
        True,
        {"enabled": "bad"},
        {"backend": "bad"},
        {"overalp": True},
        {"engram_tp_size": 4},
        {"cpu_offload": False},
        {"dp_shared_memory": True},
    ],
)
def test_invalid_additional_input_is_rejected(native, options):
    config = vc()
    config.additional_config["engram_config"] = options
    with pytest.raises(ValueError):
        resolve_engram_config(config)


@pytest.mark.parametrize(
    "key,value",
    [
        ("enable_engram", False),
        ("engram_offload_backend", "uva"),
        ("multistream_engram_overlap", True),
        ("engram_tp_size", None),
    ],
)
def test_flat_engram_switches_are_rejected(native, key, value):
    config = vc({"enabled": False})
    config.additional_config[key] = value
    with pytest.raises(ValueError, match="additional_config.engram_config"):
        resolve_engram_config(config)


def test_native_config_has_no_ascend_fields_or_metadata(native):
    schema = TypeAdapter(native.cls).json_schema()
    assert set(schema["properties"]) == {"cpu_offload", "dp_shared_memory", "embedding_across_dp"}
    for extra in ("enabled", "backend", "overlap", "_ascend_requested"):
        with pytest.raises(ValueError):
            native.cls(**{extra: True})


@pytest.mark.parametrize("overlap", [None, True, False])
def test_elastic_automatically_derives_native_policy_and_group(native, overlap):
    config = vc({"enabled": True, "backend": "elastic", "overlap": overlap})
    result = resolve_engram_config(config)
    assert result.enabled and result.cpu_offload and not result.dp_shared_memory
    assert result.group_size == 4 and result.overlap == (overlap is not False)
    assert isinstance(config.engram_config, native.cls)
    assert not config.engram_config.embedding_across_dp
    native.verify.assert_called_once_with(config)


@pytest.mark.parametrize(
    "v2,dp,pcp,expected", [(False, 4, 1, False), (True, 4, 1, True), (True, 1, 2, True), (True, 1, 1, False)]
)
def test_uva_auto_sharing_follows_runner_and_topology(native, v2, dp, pcp, expected):
    config = vc({"enabled": True}, v2=v2, dp=dp, pcp=pcp)
    assert resolve_engram_config(config).dp_shared_memory is expected
    assert config.engram_config.dp_shared_memory is expected


def test_existing_native_v2_host_sharing_policy_is_preserved(native):
    supplied = native.cls(dp_shared_memory=False)
    config = vc({"enabled": True}, native=supplied)
    assert resolve_engram_config(config).dp_shared_memory
    assert supplied.dp_shared_memory is False


def test_hbm_replicas_are_not_reported_as_shared_host_storage(native):
    config = vc({"enabled": True}, native=native.cls(cpu_offload=False))
    result = resolve_engram_config(config)
    assert not result.cpu_offload and not result.dp_shared_memory and result.replicate_across_dp
    assert not config.engram_config.dp_shared_memory


@pytest.mark.parametrize("case", ["cpu_offload", "dp_shared_memory", "world", "node", "dummy", "sp", "mtp"])
def test_elastic_rejects_invalid_combinations(native, case):
    config = vc({"enabled": True, "backend": "elastic"})
    if case in ("cpu_offload", "dp_shared_memory"):
        config.engram_config = native.cls(**{case: case == "dp_shared_memory"})
    elif case == "world":
        config.parallel_config.world_size_across_dp = 1
    elif case == "node":
        config.parallel_config.local_world_size = 2
    elif case == "dummy":
        config.load_config.load_format = "dummy"
    elif case == "sp":
        config.parallel_config.use_sequence_parallel_moe = True
    else:
        config.speculative_config = SimpleNamespace(method="mtp", num_speculative_tokens=3)
    with pytest.raises(ValueError):
        resolve_engram_config(config)


@pytest.mark.parametrize(
    "field", ["pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size"]
)
def test_elastic_parallel_restrictions_remain(native, field):
    config = vc({"enabled": True, "backend": "elastic"})
    setattr(config.parallel_config, field, 2)
    with pytest.raises(ValueError, match="PP"):
        resolve_engram_config(config)


@pytest.mark.parametrize("tokens,valid", [(None, False), (0, False), (1, True), (31, True), (32, False)])
def test_elastic_speculation_retains_ring_capacity(native, tokens, valid):
    config = vc({"enabled": True, "backend": "elastic"})
    config.speculative_config = SimpleNamespace(method="dspark", num_speculative_tokens=tokens)
    if valid:
        resolve_engram_config(config)
    else:
        with pytest.raises(ValueError, match="1..31"):
            resolve_engram_config(config)


@pytest.mark.parametrize("schedule", [[(1, 16, 3), (17, 32, 0)], {1: 3, 17: 0}])
def test_elastic_dynamic_schedule_includes_zero(native, schedule):
    config = vc({"enabled": True, "backend": "elastic"})
    config.speculative_config = SimpleNamespace(
        method="dspark", num_speculative_tokens=3, num_speculative_tokens_per_batch_size=schedule
    )
    resolve_engram_config(config)
    for invalid in (-1, 32):
        config.speculative_config.num_speculative_tokens_per_batch_size = {1: invalid}
        with pytest.raises(ValueError, match="dynamic lengths"):
            resolve_engram_config(config)


@pytest.mark.parametrize("method,report", [("dspark", False), ("mtp", True), ("eagle3", True)])
def test_elastic_does_not_override_proposer_selection(native, method, report):
    config = vc({"enabled": True, "backend": "elastic"})
    config.speculative_config = SimpleNamespace(method=method, num_speculative_tokens=3, use_dspark=lambda: report)
    with pytest.raises(ValueError):
        resolve_engram_config(config)


@pytest.mark.parametrize("enabled,backend", [(False, "elastic"), (True, "uva")])
def test_speculative_constraints_are_local_to_elastic(native, enabled, backend):
    config = vc({"enabled": enabled, "backend": backend})
    config.speculative_config = SimpleNamespace(method="mtp", num_speculative_tokens=32)
    resolve_engram_config(config)


@pytest.mark.parametrize("deep", [False, True])
def test_additional_switches_survive_reentry_and_copies(native, deep):
    config = vc({"enabled": True, "backend": "elastic"})
    expected = resolve_engram_config(config)
    assert resolve_engram_config(config) == expected
    copied = copy.deepcopy(config) if deep else copy.copy(config)
    assert resolve_engram_config(copied) == expected
    # Reconstruct using only declared fields and the original native type.
    fresh = vc({"enabled": True, "backend": "elastic"}, native=config.engram_config)
    assert resolve_engram_config(fresh) == expected
    config.additional_config["engram_config"]["overlap"] = False
    assert resolve_engram_config(config).overlap is False


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_vllm_replace_accepts_normalized_config_without_dynamic_fields(native, enabled):
    from vllm.config import replace

    config = vc({"enabled": enabled, "backend": "elastic"})
    expected = resolve_engram_config(config)
    assert set(vars(config)) == {item.name for item in fields(config)}
    copied = replace(config)
    assert resolve_engram_config(copied) == expected
    assert set(vars(copied)) == {item.name for item in fields(copied)}
    if enabled:
        assert set(vars(copied.engram_config)) <= {item.name for item in fields(copied.engram_config)}
        assert replace(copied.engram_config).cpu_offload


def test_replicated_pcp_copy_does_not_affect_target(native):
    from vllm.config import replace

    target = vc({"enabled": True}, native=native.cls(dp_shared_memory=True), dp=1, pcp=2)
    resolve_engram_config(target)
    projection = replace(target.engram_config, dp_shared_memory=False)
    draft = vc({"enabled": True}, dp=1, pcp=1, native=projection)
    result = resolve_engram_config(draft)
    assert not result.dp_shared_memory
    assert target.engram_config.dp_shared_memory


def test_validation_failure_does_not_erase_explicit_native_input(native):
    supplied = native.cls(embedding_across_dp=True)
    config = vc({"enabled": True}, native=supplied)
    with pytest.raises(ValueError, match="embedding_across_dp"):
        resolve_engram_config(config)
    assert config.engram_config is supplied


def production_function(path, name, namespace, class_name=None):
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / path).read_text())
    nodes = tree.body
    if class_name:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name).body
    function = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("replicate,expected_rows", [(False, 8), (True, 32)])
def test_hbm_embedding_uses_replica_shards_and_local_lookups(replicate, expected_rows):
    """Run the production constructor/forward with CPU allocation boundaries."""
    namespace = {
        "torch": torch,
        "nn": torch.nn,
        "in_the_same_node_as": lambda group: [True],
        "get_tp_group": lambda: SimpleNamespace(cpu_group=object()),
        "get_engram_dp_group": lambda: SimpleNamespace(cpu_group=object(), world_size=4),
        "get_engram_dp_size": lambda: 4,
        "get_tensor_model_parallel_world_size": lambda: 2,
        "get_tensor_model_parallel_rank": lambda: 0,
        "engram_head_shard_rank": lambda: 0,
        "set_weight_attrs": lambda param, attrs: [setattr(param, key, value) for key, value in attrs.items()],
    }
    path = "vllm_ascend/models/deepseek_v41/engram/embedding.py"
    cls = type("CpuEmbedding", (torch.nn.Module,), {})
    for name in ("__init__", "_get_shard_info", "forward"):
        setattr(cls, name, production_function(path, name, namespace, "AscendParallelEngramEmbedding"))
    cls._weight_loader = Mock()
    cls._allocate_weights = lambda table: (torch.zeros((table.part_num_embeddings, table.dim)), None)
    table = cls(64, 4, (4,) * 16, 0, storage_dtype=torch.bfloat16, replicate_across_dp=replicate)
    assert table.part_num_embeddings == expected_rows
    assert table._shared_group is None and not table.cpu_offload
    calls = []
    namespace["gather_engram_hashes"] = lambda ids, **kw: calls.append(kw["dp_shared_memory"]) or ids
    table.embed_gathered = lambda ids, count: ids[:count]
    ids = torch.tensor([[3, 7]])
    assert torch.equal(table.forward(ids), ids)
    assert calls == [replicate]


def test_actual_pcp_adapter_preserves_additional_switches(native):
    from vllm.config import replace

    target = vc({"enabled": True}, native=native.cls(dp_shared_memory=True), dp=1, pcp=2)
    target.parallel_config.enable_eplb = False
    target.parallel_config.eplb_config = object()
    resolve_engram_config(target)

    def reconstruct(value, **updates):
        copied = replace(value, **updates) if hasattr(value, "__dataclass_fields__") else copy.copy(value)
        if not hasattr(value, "__dataclass_fields__"):
            copied.__dict__.update(updates)
        if "parallel_config" in updates:
            resolve_engram_config(copied)
        return copied

    adapter = production_function(
        "vllm_ascend/worker/v2/spec_decode/pcp_utils.py",
        "prepare_replicated_pcp_config",
        {"replace": reconstruct},
    )
    draft, replicated = adapter(target)
    assert replicated and draft.parallel_config.prefill_context_parallel_size == 1
    assert not draft.engram_config.dp_shared_memory
    assert target.engram_config.dp_shared_memory


def test_factory_injects_additional_policy_and_preserves_other_input(native):
    from vllm_ascend.engram_config import LEGACY_ENGRAM_KEYS, ResolvedEngramConfig

    config = vc({"enabled": True, "backend": "elastic"})
    config.additional_config["enable_cpu_binding"] = False
    captured = {}

    def construct(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            derive_and_validate=Mock(),
            rl_config=SimpleNamespace(apply=Mock()),
            finegrained_tp_config=SimpleNamespace(_validate_preconditions=Mock()),
            xlite_graph_config=SimpleNamespace(_validate_preconditions=Mock()),
        )

    stub = SimpleNamespace(from_additional_config=lambda *args: None, from_vllm_config=lambda *args: None)
    namespace = {
        "_ASCEND_CONFIG": None,
        "_INIT_VLLM_CONFIG": None,
        "validate_additional_config_bool": lambda value, path: bool(value),
        "resolve_engram_config": resolve_engram_config,
        "LEGACY_ENGRAM_KEYS": LEGACY_ENGRAM_KEYS,
        "SchedulerConfig": stub,
        "SparseKVOffloadConfig": stub,
        "KVPPConfig": stub,
        "AscendConfig": SimpleNamespace(
            _resolve_dump_config_path=lambda raw: None, __dataclass_fields__={"enable_cpu_binding": None}
        ),
        "_is_ascend_config_initialized": lambda obj: False,
        "importlib": SimpleNamespace(util=SimpleNamespace(find_spec=lambda name: None)),
        "logger": SimpleNamespace(warning=Mock()),
    }
    schema = namespace["AscendConfig"]
    construct._resolve_dump_config_path = schema._resolve_dump_config_path
    construct.__dataclass_fields__ = schema.__dataclass_fields__
    namespace["AscendConfig"] = construct
    factory = production_function("vllm_ascend/ascend_config.py", "init_ascend_config", namespace)
    factory(config)
    assert isinstance(captured["engram_config"], ResolvedEngramConfig)
    assert captured["engram_config"].group_size == 4
    assert captured["enable_cpu_binding"] is False
    assert not set(LEGACY_ENGRAM_KEYS).intersection(captured)
    # The platform can inject the same result without a second resolution.
    resolved = captured["engram_config"]
    namespace["resolve_engram_config"] = Mock(side_effect=AssertionError("double resolution"))
    factory(config, engram_config=resolved)
    assert captured["engram_config"] is resolved
