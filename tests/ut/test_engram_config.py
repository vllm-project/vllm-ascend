# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import copy
import sys
from dataclasses import FrozenInstanceError, dataclass, fields, make_dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from pydantic import TypeAdapter, model_validator

from vllm_ascend.config_utils import config as config_dataclass
from vllm_ascend.engram_config import (
    AscendEngramConfig,
    resolve_engram_config,
)


@dataclass
class _ParallelConfig:
    tensor_parallel_size: int = 8
    data_parallel_size: int = 1
    data_parallel_size_local: int = 1
    data_parallel_external_lb: bool = False
    data_parallel_rank: int = 0
    pipeline_parallel_size: int = 1
    prefill_context_parallel_size: int = 1
    decode_context_parallel_size: int = 1
    enable_elastic_ep: bool = False
    use_sequence_parallel_moe: bool = False
    enable_dbo: bool = False
    ubatch_size: int = 0

    @property
    def world_size(self):
        return self.tensor_parallel_size * self.pipeline_parallel_size * self.prefill_context_parallel_size

    @property
    def world_size_across_dp(self):
        return self.world_size * self.data_parallel_size

    @property
    def local_world_size(self):
        return self.world_size

    @property
    def use_ubatching(self):
        return self.enable_dbo or self.ubatch_size > 1


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


@pytest.mark.parametrize("v2", [False, True])
def test_elastic_disables_multithread_load_without_changing_other_loader_options(native, v2, caplog):
    config = vc({"enabled": True, "backend": "elastic"}, v2=v2)
    supplied = {"enable_multithread_load": True, "num_threads": 128, "enable_weights_track": True}
    config.load_config.model_loader_extra_config = supplied
    resolve_engram_config(config)
    resolve_engram_config(config)
    assert config.load_config.model_loader_extra_config == {**supplied, "enable_multithread_load": False}
    assert supplied["enable_multithread_load"] is True
    assert caplog.text.count("Disabling multithread checkpoint loading for Elastic Engram") == 1


@pytest.mark.parametrize("options", [{}, {"enabled": False, "backend": "elastic"}, {"enabled": True}])
def test_disabled_and_uva_preserve_multithread_load(native, options):
    config = vc(options)
    supplied = {"enable_multithread_load": True, "num_threads": 128}
    config.load_config.model_loader_extra_config = supplied
    resolve_engram_config(config)
    assert config.load_config.model_loader_extra_config is supplied
    assert supplied == {"enable_multithread_load": True, "num_threads": 128}


@pytest.mark.parametrize("extra", [None, {}, {"num_threads": 128}, {"enable_multithread_load": False}])
def test_elastic_does_not_add_loader_options_when_multithread_load_is_inactive(native, extra):
    config = vc({"enabled": True, "backend": "elastic"})
    config.load_config.model_loader_extra_config = extra
    resolve_engram_config(config)
    assert config.load_config.model_loader_extra_config is extra


def test_failed_native_validation_preserves_loader_options(native):
    config = vc({"enabled": True, "backend": "elastic"})
    supplied = {"enable_multithread_load": True, "num_threads": 128}
    config.load_config.model_loader_extra_config = supplied
    native.verify.side_effect = ValueError("native validation failed")
    with pytest.raises(ValueError, match="native validation failed"):
        resolve_engram_config(config)
    assert config.load_config.model_loader_extra_config is supplied
    assert supplied["enable_multithread_load"] is True


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


@pytest.mark.parametrize("rank", range(4))
def test_elastic_external_dp_defers_locality_checks_for_each_server(native, rank):
    config = vc({"enabled": True, "backend": "elastic", "overlap": False})
    config.parallel_config = _ParallelConfig(
        tensor_parallel_size=1,
        data_parallel_size=4,
        data_parallel_size_local=1,
        data_parallel_external_lb=True,
        data_parallel_rank=rank,
    )
    config.speculative_config = SimpleNamespace(method="dspark", num_speculative_tokens=5)
    resolved = resolve_engram_config(config)
    assert resolved.group_size == 4
    assert resolved.cpu_offload and not resolved.dp_shared_memory


@pytest.mark.parametrize("local_dp", [1, 4])
def test_elastic_internal_dp_retains_configured_locality_check(native, local_dp):
    config = vc({"enabled": True, "backend": "elastic"})
    config.parallel_config = _ParallelConfig(
        tensor_parallel_size=1, data_parallel_size=4, data_parallel_size_local=local_dp
    )
    if local_dp == 4:
        assert resolve_engram_config(config).group_size == 4
    else:
        with pytest.raises(ValueError, match="world_size_across_dp=4, configured_local_ranks=1"):
            resolve_engram_config(config)


@pytest.mark.parametrize("external_dp", [False, True])
def test_elastic_requires_multiple_ranks_in_all_launch_modes(native, external_dp):
    config = vc({"enabled": True, "backend": "elastic"})
    config.parallel_config = _ParallelConfig(tensor_parallel_size=1, data_parallel_external_lb=external_dp)
    with pytest.raises(ValueError, match=">=2 participating ranks; world_size_across_dp=1"):
        resolve_engram_config(config)


def test_elastic_uses_external_dp_target_policy_when_draft_parallel_metadata_differs(native):
    config = vc({"enabled": True, "backend": "elastic", "overlap": False})
    target_parallel = _ParallelConfig(
        tensor_parallel_size=1,
        data_parallel_size=4,
        data_parallel_size_local=1,
        data_parallel_external_lb=True,
    )
    config.parallel_config = _ParallelConfig(tensor_parallel_size=1)
    config.speculative_config = SimpleNamespace(
        method="dspark", num_speculative_tokens=5, target_parallel_config=target_parallel
    )
    assert resolve_engram_config(config).group_size == 4


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


@pytest.mark.parametrize("dp", [1, 2])
@pytest.mark.parametrize("draft_model_config", [False, True])
@pytest.mark.parametrize("external_dp", [False, True])
def test_elastic_keeps_target_domain_when_dspark_reconstructs_with_tp1(dp, draft_model_config, external_dp):
    from vllm.config import replace

    target = vc({"enabled": True, "backend": "elastic", "overlap": False}, dp=dp)
    parallel = _ParallelConfig(
        data_parallel_size=dp,
        data_parallel_size_local=1 if external_dp else dp,
        data_parallel_external_lb=external_dp,
    )
    target.parallel_config = parallel
    draft_model = SimpleNamespace(
        architecture="DeepseekV41DSparkModel", hf_text_config=SimpleNamespace(engram_layer_ids=[])
    )
    target.speculative_config = SimpleNamespace(
        method="dspark",
        num_speculative_tokens=5,
        target_model_config=target.model_config,
        draft_model_config=draft_model,
        target_parallel_config=parallel,
    )
    expected = resolve_engram_config(target)
    assert expected.group_size == 8 * dp
    draft = replace(
        target,
        model_config=draft_model if draft_model_config else target.model_config,
        parallel_config=replace(parallel, tensor_parallel_size=1),
    )
    assert draft.parallel_config.world_size_across_dp == dp
    assert resolve_engram_config(draft).group_size == expected.group_size
    assert parallel.tensor_parallel_size == 8


def test_elastic_rejects_invalid_target_domain_even_if_draft_world_is_larger():
    value = vc({"enabled": True, "backend": "elastic", "overlap": False}, dp=1)
    value.parallel_config = _ParallelConfig()
    value.speculative_config = SimpleNamespace(
        method="dspark",
        num_speculative_tokens=5,
        target_model_config=value.model_config,
        draft_model_config=SimpleNamespace(architecture="DeepseekV41DSparkModel"),
        target_parallel_config=_ParallelConfig(tensor_parallel_size=1),
    )
    with pytest.raises(ValueError, match=">=2"):
        resolve_engram_config(value)


def _platform_config_update(monkeypatch):
    from vllm.logger import logger

    # Isolate logging, quantization and unrelated platform validators, while
    # executing the full production update and parameter-normalization methods.
    noop = Mock()
    for name, exports in (
        ("vllm_ascend.logger", {"configure_ascend_file_logging": noop, "configure_ascend_logging": noop}),
        ("vllm_ascend.quantization.utils", {"maybe_auto_detect_quantization": noop}),
        ("vllm_ascend.quantization.methods.kv_cache.turboquant.config", {"validate_turboquant": noop}),
    ):
        module = ModuleType(name)
        module.__dict__.update(exports)
        monkeypatch.setitem(sys.modules, name, module)
    namespace = {
        name: noop
        for name in (
            "_validate_model_runner_config",
            "_validate_routing_replay_config",
            "_validate_pcp_dcp_config",
            "_validate_draft_decode_context_parallel_config",
            "_validate_parallel_config",
            "_validate_eplb_config",
            "_configure_minimax_m3_a5_mixed_kv_cache",
        )
    }
    namespace.update(
        logger=logger,
        envs_vllm=SimpleNamespace(VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000),
        _validate_engram_config=resolve_engram_config,
        init_ascend_config=lambda value, *, engram_config: engram_config,
    )
    path = "vllm_ascend/platform.py"
    production_function(path, "_fix_incompatible_config", namespace)
    update = production_function(path, "check_and_update_config", namespace, "NPUPlatform")
    platform = SimpleNamespace(device_type="npu", _validate_indexer_pp_config=noop)
    return update, platform, namespace


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("dbo,ubatch", [(True, 0), (False, 2), (True, 2)])
def test_platform_normalizes_microbatch_options_before_native_engram_validation(monkeypatch, enabled, dbo, ubatch):
    from vllm.config import EngramConfig, VllmConfig

    update, platform, namespace = _platform_config_update(monkeypatch)
    value = vc({"enabled": enabled, "overlap": False}, native=EngramConfig(cpu_offload=True), dp=1)
    value.parallel_config = _ParallelConfig(tensor_parallel_size=1, enable_dbo=dbo, ubatch_size=ubatch)
    value.cache_config = SimpleNamespace(cache_dtype="auto")
    value.attention_config = SimpleNamespace(indexer_kv_dtype="auto")
    value.scheduler_config = value.observability_config = value.kv_transfer_config = value.compilation_config = None

    class ConfigurationVerified(Exception):
        pass

    def check(config, resolved):
        assert resolved.enabled is enabled
        assert not config.parallel_config.use_ubatching
        raise ConfigurationVerified

    # Stop after actual configuration validation, before compiler/worker setup.
    namespace["_check_ascend_config"] = check
    with pytest.raises(ConfigurationVerified):
        update(platform, value)
    assert value.parallel_config.enable_dbo is False
    assert value.parallel_config.ubatch_size == 0
    # VllmConfig.__post_init__ performs this upstream check after the platform
    # hook returns. The normalized values must also pass that second check.
    VllmConfig._resolve_and_verify_engram_config(value)
    assert (value.engram_config is not None) is enabled


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
