# SPDX-License-Identifier: Apache-2.0
"""Run draft config construction and the Ascend scheduler checks on CPU."""

import ast
import copy
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_definition(path, name, namespace, methods=None):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend" / path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(node for node in tree.body if getattr(node, "name", None) == name)
    if methods is not None:
        node.body = [method for method in node.body if getattr(method, "name", None) in methods]
    # Dependency injection replaces the scheduler implementation, while all
    # platform policy checks execute verbatim from the production source.
    if name == "_check_ascend_config":
        node.body = [statement for statement in node.body if not isinstance(statement, ast.ImportFrom)]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace[name]


def _config_factory():
    namespace = {
        "RecomputeSchedulerConfig": SimpleNamespace(
            initialize_from_config=lambda config: copy.copy(config.scheduler_config)
        ),
        "_get_recompute_scheduler_cls": lambda **kwargs: "recompute",
        "_validate_kv_load_failure_policy": lambda config: None,
        "logger": SimpleNamespace(warning=lambda *args: None),
    }
    validate = _load_definition("platform.py", "_check_ascend_config", namespace)

    @dataclass
    class Config:
        parallel_config: object
        cache_config: object
        speculative_config: object
        kv_transfer_config: object
        additional_config: dict
        scheduler_config: object

        def __post_init__(self):
            scheduler = self.additional_config["scheduler_config"]
            extension = SimpleNamespace(
                enable_balance_scheduling=False,
                short_request_first_config=SimpleNamespace(enabled=False),
                dyntra_lb_config=SimpleNamespace(enabled=scheduler["dyntra"]),
                profiling_chunk_config=SimpleNamespace(enabled=False),
                recompute_scheduler_enable=scheduler["recompute_scheduler_enable"],
            )
            validate(self, SimpleNamespace(scheduler_config=extension))

    return Config


@pytest.mark.parametrize("runner", ["mrv1", "mrv2"])
@pytest.mark.parametrize(
    "role,target_dcp,dp,recompute,dyntra",
    [("kv_producer", 1, 1, False, False), ("kv_consumer", 8, 4, True, True), (None, 8, 4, False, False)],
)
def test_gqa_draft_preserves_pd_scheduler_and_target_parallel_identity(runner, role, target_dcp, dp, recompute, dyntra):
    draft_model = SimpleNamespace(hf_config=SimpleNamespace(), hf_text_config=SimpleNamespace(), use_mla=False)
    config = _config_factory()(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            pipeline_parallel_size=1,
            decode_context_parallel_size=target_dcp,
            data_parallel_size=dp,
            data_parallel_rank=dp - 1,
            rank=7,
            nnodes_within_dp=1,
        ),
        cache_config=SimpleNamespace(block_size=768),
        speculative_config=SimpleNamespace(
            draft_parallel_config=SimpleNamespace(tensor_parallel_size=8, pipeline_parallel_size=1),
            draft_model_config=draft_model,
        ),
        kv_transfer_config=SimpleNamespace(kv_role=role) if role else None,
        additional_config={"scheduler_config": {"recompute_scheduler_enable": recompute, "dyntra": dyntra}},
        scheduler_config=SimpleNamespace(async_scheduling=True),
    )

    class Parent:
        def __init__(self, config, device):
            self.vllm_config = config
            self.draft_model_config = draft_model

        def _create_draft_vllm_config(self):
            return copy.copy(self.vllm_config)

    namespace = {
        "copy": copy,
        "replace": replace,
        "AscendDflashProposer": Parent,
        "DSparkSpeculator": Parent,
        "ascend_envs": SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
        "is_kimi_k3_gqa_dspark": lambda config: True,
        "prepare_replicated_pcp_config": lambda config: (config, False),
        "DeviceMetadataExecutor": object,
    }
    if runner == "mrv1":
        cls = _load_definition(
            "spec_decode/dspark_proposer.py", "AscendDSparkProposer", namespace, {"_create_draft_vllm_config"}
        )
        proposer = cls(config, "cpu")
        proposer.speculative_config = config.speculative_config
        proposer._uses_dcp_replicated_draft_kv = lambda: True
        draft = proposer._create_draft_vllm_config()
    else:
        cls = _load_definition(
            "worker/v2/spec_decode/dspark/speculator.py", "AscendDSparkSpeculator", namespace, {"__init__"}
        )
        draft = cls(config, "cpu").vllm_config
        # The upstream DSpark model loader reconstructs VllmConfig using
        # replace(), including the Ascend scheduler validation.
        draft = replace(draft)

    assert draft.kv_transfer_config is config.kv_transfer_config
    assert draft.additional_config is config.additional_config
    assert draft.parallel_config.tensor_parallel_size == 8
    assert draft.parallel_config.decode_context_parallel_size == 1
    assert draft.parallel_config.data_parallel_size == dp
    assert draft.parallel_config.data_parallel_rank == dp - 1
    assert draft.parallel_config.rank == 7
    assert config.parallel_config.decode_context_parallel_size == target_dcp
    draft.cache_config.block_size = 128
    assert config.cache_config.block_size == 768
    if recompute:
        assert config.additional_config["scheduler_config"]["recompute_scheduler_enable"]
        assert draft.scheduler_config.scheduler_cls == "recompute"
