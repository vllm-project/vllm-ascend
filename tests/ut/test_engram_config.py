# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import copy
from dataclasses import FrozenInstanceError, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.engram_config import (
    AscendEngramConfig,
    copy_engram_projection,
    resolve_engram_config,
)


@pytest.fixture
def native(monkeypatch):
    """Replace only the external vLLM schema/validation boundary."""
    import vllm.config

    defaults = SimpleNamespace(cpu=True)

    @dataclass
    class Native:
        cpu_offload: bool = field(default_factory=lambda: defaults.cpu)
        dp_shared_memory: bool = False
        embedding_across_dp: bool = False

        def __post_init__(self):
            if self.dp_shared_memory and not self.cpu_offload:
                raise ValueError("dp_shared_memory requires cpu_offload=True")

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
    return SimpleNamespace(cls=Native, defaults=defaults, verify=verification)


def vc(options=None, *, legacy=None, native=None, v2=True, dp=4, pcp=1):
    text = SimpleNamespace(engram_layer_ids=[1, 14])
    model = SimpleNamespace(architecture="DeepseekV41ForCausalLM", hf_text_config=text, hf_config=text)
    return SimpleNamespace(
        additional_config=legacy if legacy is not None else {"engram_config": options or {}},
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
    config = SimpleNamespace(additional_config={"engram_config": options}, engram_config=None)
    resolved = resolve_engram_config(config)
    assert not resolved.enabled and not resolved.overlap and resolved.group_size is None
    assert config.engram_config is None
    native.verify.assert_not_called()


@pytest.mark.parametrize(
    "options", [None, [], True, {"enabled": "bad"}, {"backend": "bad"}, {"overalp": True}, {"engram_tp_size": 4}]
)
def test_invalid_nested_input_is_rejected(native, options):
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
def test_new_and_legacy_keys_cannot_be_mixed(native, key, value):
    config = vc({"enabled": False})
    config.additional_config[key] = value
    with pytest.raises(ValueError, match="Do not mix"):
        resolve_engram_config(config)


def test_explicit_native_config_cannot_be_mixed_even_when_disabled(native):
    with pytest.raises(ValueError, match="--engram-config"):
        resolve_engram_config(vc(native=native.cls()))


@pytest.mark.parametrize("overlap", [None, True, False])
def test_elastic_automatically_derives_native_policy_and_group(native, overlap):
    config = vc({"enabled": True, "backend": "elastic", "overlap": overlap})
    result = resolve_engram_config(config)
    assert result.enabled and result.cpu_offload and not result.dp_shared_memory
    assert result.group_size == 4 and result.overlap == (overlap is not False)
    assert isinstance(config.engram_config, native.cls)
    assert not config.engram_config.embedding_across_dp
    native.verify.assert_called_once_with(config)


@pytest.mark.parametrize("backend", ["uva", "elastic"])
def test_new_cpu_default_is_independent_of_legacy_environment(native, backend):
    native.defaults.cpu = False
    result = resolve_engram_config(vc({"enabled": True, "backend": backend}))
    assert result.cpu_offload
    legacy = resolve_engram_config(vc(legacy={"enable_engram": True}, v2=False))
    assert not legacy.cpu_offload


@pytest.mark.parametrize(
    "v2,dp,pcp,expected", [(False, 4, 1, False), (True, 4, 1, True), (True, 1, 2, True), (True, 1, 1, False)]
)
def test_uva_auto_sharing_follows_runner_and_topology(native, v2, dp, pcp, expected):
    config = vc({"enabled": True}, v2=v2, dp=dp, pcp=pcp)
    assert resolve_engram_config(config).dp_shared_memory is expected
    assert config.engram_config.dp_shared_memory is expected


def test_new_explicit_false_is_not_silently_overwritten(native):
    with pytest.raises(ValueError, match="V2.*automatic"):
        resolve_engram_config(vc({"enabled": True, "dp_shared_memory": False}))
    legacy = vc(legacy={"enable_engram": True}, native=native.cls(dp_shared_memory=False))
    assert resolve_engram_config(legacy).dp_shared_memory


def test_hbm_replicas_are_not_reported_as_shared_host_storage(native):
    config = vc({"enabled": True, "cpu_offload": False})
    result = resolve_engram_config(config)
    assert not result.cpu_offload and not result.dp_shared_memory and result.replicate_across_dp
    assert not config.engram_config.dp_shared_memory
    legacy = vc(legacy={"enable_engram": True}, native=native.cls(cpu_offload=False))
    assert resolve_engram_config(legacy).replicate_across_dp
    with pytest.raises(ValueError, match="requires cpu_offload"):
        resolve_engram_config(vc({"enabled": True, "cpu_offload": False, "dp_shared_memory": True}))


@pytest.mark.parametrize("case", ["cpu_offload", "dp_shared_memory", "world", "node", "dummy", "sp", "mtp"])
def test_elastic_rejects_invalid_combinations(native, case):
    config = vc({"enabled": True, "backend": "elastic"})
    if case in ("cpu_offload", "dp_shared_memory"):
        config.additional_config["engram_config"][case] = case == "dp_shared_memory"
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


@pytest.mark.parametrize("size", [0, 1, 2, 3, 8])
def test_legacy_group_size_is_only_a_checked_alias(native, size):
    config = vc(legacy={"enable_engram": True, "engram_offload_backend": "elastic", "engram_tp_size": size})
    with pytest.raises(ValueError, match="engram_tp_size"):
        resolve_engram_config(config)
    config.additional_config["engram_tp_size"] = 4
    assert resolve_engram_config(config).group_size == 4


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
def test_projection_provenance_survives_reentry_and_copies(native, deep):
    config = vc({"enabled": True, "backend": "elastic"})
    expected = resolve_engram_config(config)
    assert resolve_engram_config(config) == expected
    copied = copy.deepcopy(config) if deep else copy.copy(config)
    assert resolve_engram_config(copied) == expected
    # Reconstruct a VllmConfig using its declared fields and marked projection.
    fresh = vc({"enabled": True, "backend": "elastic"}, native=config.engram_config)
    assert resolve_engram_config(fresh) == expected
    config.additional_config["engram_config"]["overlap"] = False
    assert resolve_engram_config(config).overlap is False


def test_legacy_native_defaults_survive_reentry_and_toggle(native):
    config = vc(legacy={"enable_engram": False}, native=native.cls(cpu_offload=False), v2=False)
    resolve_engram_config(config)
    assert config.engram_config is None
    config.additional_config["enable_engram"] = True
    assert not resolve_engram_config(config).cpu_offload
    assert not resolve_engram_config(copy.deepcopy(config)).cpu_offload


def test_replicated_pcp_projection_retains_source_without_affecting_target(native):
    target = vc({"enabled": True, "dp_shared_memory": True}, dp=1, pcp=2)
    resolve_engram_config(target)
    projection = copy_engram_projection(target.engram_config, dp_shared_memory=False)
    draft = vc(copy.deepcopy(target.additional_config["engram_config"]), dp=1, pcp=1, native=projection)
    result = resolve_engram_config(draft)
    assert not result.dp_shared_memory
    assert target.engram_config.dp_shared_memory
    assert target.additional_config["engram_config"]["dp_shared_memory"]
    # If a copied config gains peers again, restore the declared sharing policy.
    draft.parallel_config.data_parallel_size = 2
    assert resolve_engram_config(draft).dp_shared_memory


def test_validation_failure_does_not_erase_explicit_native_input(native):
    supplied = native.cls(embedding_across_dp=True)
    config = vc(legacy={"enable_engram": True}, native=supplied)
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


def test_actual_pcp_adapter_preserves_nested_input_provenance(native):
    target = vc({"enabled": True, "dp_shared_memory": True}, dp=1, pcp=2)
    target.parallel_config.enable_eplb = False
    target.parallel_config.eplb_config = object()
    resolve_engram_config(target)

    def reconstruct(value, **updates):
        copied = copy.copy(value)
        copied.__dict__.update(updates)
        if "parallel_config" in updates:
            # Real vllm.config.replace reconstructs declared fields and reruns
            # VllmConfig.__post_init__, without arbitrary instance attributes.
            copied.__dict__.pop("_ascend_engram_state", None)
            resolve_engram_config(copied)
        return copied

    adapter = production_function(
        "vllm_ascend/worker/v2/spec_decode/pcp_utils.py",
        "prepare_replicated_pcp_config",
        {"replace": reconstruct, "copy_engram_projection": copy_engram_projection},
    )
    draft, replicated = adapter(target)
    assert replicated and draft.parallel_config.prefill_context_parallel_size == 1
    assert not draft.engram_config.dp_shared_memory
    assert target.engram_config.dp_shared_memory


@pytest.mark.parametrize("legacy", [False, True])
def test_factory_injects_normalized_policy_and_strips_all_engram_input(native, legacy):
    from vllm_ascend.engram_config import LEGACY_ENGRAM_KEYS, ResolvedEngramConfig

    config = vc({"enabled": True, "backend": "elastic"})
    if legacy:
        config.additional_config = {"enable_engram": True, "engram_offload_backend": "elastic"}
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
