# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the V4.1 PCP runtime boundary without importing NPU libraries."""

import ast
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def validate():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/platform.py"
    names = {"_validate_v41_pcp_config", "_validate_parallel_config"}
    nodes = [
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "VllmConfig": object,
        "is_deepseek_v41": lambda config: config.model_type == "deepseek_v41",
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def config(pcp=2, model_type="deepseek_v41"):
    return SimpleNamespace(
        use_v2_model_runner=True,
        model_config=SimpleNamespace(enforce_eager=True, hf_config=SimpleNamespace(model_type=model_type)),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            data_parallel_size=1,
        ),
        speculative_config=None,
        kv_transfer_config=None,
        additional_config={},
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )


def test_pcp_normalizes_auto_cache_to_supported_bf16(validate):
    runtime = config()
    validate["_validate_v41_pcp_config"](runtime)
    assert runtime.cache_config.cache_dtype == "bfloat16"


@pytest.mark.parametrize("pcp,model_type", [(1, "deepseek_v41"), (2, "deepseek_v4")])
def test_existing_runner_modes_are_unchanged(validate, pcp, model_type):
    runtime = config(pcp, model_type)
    runtime.model_config.enforce_eager = False
    runtime.speculative_config = SimpleNamespace()
    runtime.cache_config.cache_dtype = "fp8"
    validate["_validate_v41_pcp_config"](runtime)
    assert runtime.cache_config.cache_dtype == "fp8"


@pytest.mark.parametrize(
    "feature,message",
    [
        ("mrv1", "requires model runner V2"),
        ("graph", "requires enforce_eager"),
        ("pp", "PP=DCP=1"),
        ("dcp", "PP=DCP=1"),
        ("pd", "colocated prefill and decode"),
        ("spec", "speculative decoding to be disabled"),
        ("dsacp", "cannot be enabled at the same time"),
        ("cache", "BF16 KV cache"),
    ],
)
def test_unsupported_combinations_fail_before_model_execution(validate, feature, message):
    runtime = config()
    if feature == "mrv1":
        runtime.use_v2_model_runner = False
    elif feature == "graph":
        runtime.model_config.enforce_eager = False
    elif feature == "pp":
        runtime.parallel_config.pipeline_parallel_size = 2
    elif feature == "dcp":
        runtime.parallel_config.decode_context_parallel_size = 2
    elif feature == "pd":
        runtime.kv_transfer_config = SimpleNamespace()
    elif feature == "spec":
        runtime.speculative_config = SimpleNamespace()
    elif feature == "dsacp":
        runtime.additional_config["enable_dsa_cp"] = True
    else:
        runtime.cache_config.cache_dtype = "fp8"
    with pytest.raises(ValueError, match=message):
        validate["_validate_v41_pcp_config"](runtime)


def test_platform_parallel_validation_calls_the_v41_guard(validate):
    runtime = config()
    runtime.model_config.enforce_eager = False
    with pytest.raises(ValueError, match="V4.1 PCP requires enforce_eager"):
        validate["_validate_parallel_config"](runtime)


@pytest.mark.parametrize("dp,pcp", [(4, 1), (2, 2), (4, 2)])
def test_platform_allows_dp_with_pcp(validate, dp, pcp):
    runtime = config(pcp=pcp)
    runtime.parallel_config.data_parallel_size = dp
    validate["_validate_v41_pcp_config"](runtime)
    assert runtime.cache_config.cache_dtype == ("auto" if pcp == 1 else "bfloat16")


@pytest.mark.parametrize("component", ["lmhead", "embedding", "oproj", "mlp"])
def test_pcp_rejects_finegrained_groups_with_missing_pcp_dimension(validate, component):
    runtime = config()
    runtime.additional_config["finegrained_tp_config"] = {f"{component}_tensor_parallel_size": 2}
    with pytest.raises(ValueError, match="fine-grained TP"):
        validate["_validate_v41_pcp_config"](runtime)


@pytest.fixture
def engram_config_type():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/patch/platform/patch_engram_config.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "verify_parallel_config"
    )

    class UpstreamConfig:
        embedding_across_dp = False
        dp_shared_memory = False

        def verify_parallel_config(self, parallel_config):
            if self.dp_shared_memory:
                if parallel_config.data_parallel_size <= 1:
                    raise ValueError("dp_shared_memory requires data_parallel_size > 1.")
                if parallel_config.enable_elastic_ep:
                    raise ValueError("dp_shared_memory is not supported with elastic EP.")
            self.parent_checked = True
            self.parent_storage_size = parallel_config.data_parallel_size

    scope = {"_verify_parallel_config": UpstreamConfig.verify_parallel_config, "copy": copy}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), scope)
    return type("NativeEngramConfig", (UpstreamConfig,), {"verify_parallel_config": scope["verify_parallel_config"]})


def engram_topology(tp=8, pcp=2, dp=1, **overrides):
    values = dict(
        tensor_parallel_size=tp,
        prefill_context_parallel_size=pcp,
        data_parallel_size=dp,
        data_parallel_size_local=dp,
        data_parallel_external_lb=False,
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        enable_elastic_ep=False,
        nnodes=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("tp,dp,pcp", [(8, 1, 2), (4, 1, 4), (2, 1, 8), (1, 1, 16), (4, 2, 2), (2, 4, 2)])
def test_engram_accepts_single_node_pcp_and_keeps_parent_validation(engram_config_type, tp, dp, pcp):
    config = engram_config_type()
    config.verify_parallel_config(engram_topology(tp, pcp, dp))
    assert config.parent_checked


@pytest.mark.parametrize("external", [False, True])
def test_engram_pcp1_retains_dp_shared_memory(engram_config_type, external):
    config = engram_config_type()
    config.dp_shared_memory = True
    config.verify_parallel_config(
        engram_topology(4, 1, 4, data_parallel_external_lb=external, data_parallel_size_local=1 if external else 4)
    )


def test_engram_pcp1_retains_multinode_node_local_dp(engram_config_type):
    config = engram_config_type()
    config.dp_shared_memory = True
    topology = engram_topology(8, 1, 4, nnodes=2, data_parallel_size_local=2)
    config.verify_parallel_config(topology)
    assert config.parent_storage_size == 4
    assert topology.data_parallel_size == 4


@pytest.mark.parametrize("external", [False, True])
def test_engram_shared_memory_accepts_pcp_and_dp(engram_config_type, external):
    config = engram_config_type()
    config.dp_shared_memory = True
    config.verify_parallel_config(
        engram_topology(4, 2, 2, data_parallel_external_lb=external, data_parallel_size_local=1 if external else 2)
    )


@pytest.mark.parametrize("tp,pcp", [(1, 16), (2, 8), (4, 4), (8, 2)])
def test_engram_shared_memory_accepts_pcp_without_dp(engram_config_type, tp, pcp):
    config = engram_config_type()
    config.dp_shared_memory = True
    topology = engram_topology(tp=tp, pcp=pcp, dp=1)
    config.verify_parallel_config(topology)
    assert config.parent_checked
    assert config.parent_storage_size == pcp
    assert topology.data_parallel_size == 1


def test_engram_shared_memory_rejects_single_member(engram_config_type):
    config = engram_config_type()
    config.dp_shared_memory = True
    with pytest.raises(ValueError, match="data_parallel_size > 1"):
        config.verify_parallel_config(engram_topology(tp=8, pcp=1, dp=1))


def test_engram_pcp_shared_memory_keeps_parent_elastic_validation(engram_config_type):
    config = engram_config_type()
    config.dp_shared_memory = True
    with pytest.raises(ValueError, match="not supported with elastic EP"):
        config.verify_parallel_config(engram_topology(tp=2, pcp=8, dp=1, enable_elastic_ep=True))


@pytest.mark.parametrize(
    "topology,message",
    [
        (engram_topology(8, 4), "at most 16 ranks"),
        (engram_topology(4, 2, 4), "at most 16 ranks"),
        (engram_topology(nnodes=2), "single-node"),
        (engram_topology(decode_context_parallel_size=2), "PP=DCP=1"),
        (engram_topology(pipeline_parallel_size=2), "PP=DCP=1"),
    ],
)
def test_engram_rejects_unsupported_pcp_topologies(engram_config_type, topology, message):
    with pytest.raises(ValueError, match=message):
        engram_config_type().verify_parallel_config(topology)
