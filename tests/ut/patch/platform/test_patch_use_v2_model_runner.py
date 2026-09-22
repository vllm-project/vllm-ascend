import pytest
from vllm.config import parallel as parallel_module
from vllm.config import replace
from vllm.config.parallel import ParallelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.config.vllm import VllmConfig

from vllm_ascend.patch.platform import patch_parallel_config, patch_use_v2_model_runner
from vllm_ascend.utils import vllm_version_is


def test_ascend_v1_supported_features_are_not_rejected(monkeypatch):
    monkeypatch.setattr(
        patch_use_v2_model_runner,
        "_original_get_v1_model_runner_unsupported_features",
        lambda _: [
            "prefill context parallel",
            "dspark speculative decoding",
            "dflash2 drafts",
            "diffusion models",
        ],
    )

    unsupported = patch_use_v2_model_runner._patched_get_v1_model_runner_unsupported_features(object())

    assert unsupported == ["prefill context parallel", "diffusion models"]


def test_upstream_pcp_unsupported_feature_is_preserved(monkeypatch):
    monkeypatch.setattr(
        patch_use_v2_model_runner,
        "_original_get_unsupported_features",
        lambda _: ["prefill context parallelism", "diffusion models"],
    )
    monkeypatch.setattr(
        patch_use_v2_model_runner,
        "resolve_spec_pp_support",
        lambda _: None,
    )

    unsupported = patch_use_v2_model_runner._patched_get_unsupported_features(object())

    # Both supported pins delegate PCP checks to the manager (#53853).
    # The Ascend wrapper must preserve any remaining upstream restriction.
    assert unsupported == ["prefill context parallelism", "diffusion models"]


@pytest.fixture
def pcp_dp(monkeypatch):
    # Run on both supported lanes: patched v0.29 and native verified main.
    monkeypatch.setattr(parallel_module.current_platform, "device_name", "npu")
    monkeypatch.setattr(parallel_module, "get_open_ports_list", lambda count: list(range(29000, 29000 + count)))
    return dict(
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=2,
        decode_context_parallel_size=1,
        data_parallel_size=2,
        data_parallel_size_local=2,
        distributed_executor_backend="mp",
        is_moe_model=True,
    )


def test_parallel_validator_has_single_owner():
    validator = ParallelConfig._validate_parallel_config
    registered = ParallelConfig.__pydantic_decorators__.model_validators["_validate_parallel_config"].func
    assert registered is validator
    if vllm_version_is("0.29.0"):
        assert validator is patch_parallel_config._validate_parallel_config
    else:
        assert validator.__module__ == parallel_module.__name__


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("dcp", [1, 2, 4])
@pytest.mark.parametrize("use_v2", [False, None, True])
def test_pcp_dp_construction_and_replace(pcp_dp, monkeypatch, nested, dcp, use_v2):
    # Exercise real Pydantic schemas without model initialization/downloads.
    # Config validity does not establish model-runner or backend support.
    monkeypatch.setattr(patch_use_v2_model_runner.envs, "VLLM_USE_V2_MODEL_RUNNER", use_v2)
    monkeypatch.setattr(VllmConfig, "__post_init__", lambda self: None)
    monkeypatch.setattr(SpeculativeConfig, "__post_init__", lambda self: None)
    values = pcp_dp | {"decode_context_parallel_size": dcp}
    if nested:
        config = VllmConfig(
            parallel_config=values,
            speculative_config={"method": "eagle3", "num_speculative_tokens": 1},
        ).parallel_config
    else:
        config = ParallelConfig(**values)
    copied = replace(config)
    wrapped = VllmConfig(parallel_config=copied).parallel_config
    for actual in (config, copied, wrapped):
        assert actual.prefill_context_parallel_size == 2
        assert actual.decode_context_parallel_size == dcp
        assert actual.data_parallel_size == actual.data_parallel_size_local == 2
        assert actual.world_size == 4
        assert actual.world_size_across_dp == 8
    assert wrapped is copied


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"data_parallel_size_local": 3}, "data_parallel_size_local"),
        (
            {"data_parallel_external_lb": True, "data_parallel_size": 1, "data_parallel_size_local": 1},
            "data_parallel_external_lb",
        ),
        ({"numa_bind": False, "numa_bind_cpus": ["0-1"]}, "numa_bind_nodes and numa_bind_cpus"),
        ({"decode_context_parallel_size": 3}, "When PCP is enabled, DCP must"),
        (
            {"prefill_context_parallel_size": 1, "decode_context_parallel_size": 3},
            "must be divisible by dcp_size",
        ),
    ],
)
def test_pcp_dp_preserves_other_validation(pcp_dp, overrides, error):
    with pytest.raises(ValueError, match=error):
        ParallelConfig(**(pcp_dp | overrides))


def test_invalid_dcp_does_not_mutate_pcp(pcp_dp):
    config = ParallelConfig(**pcp_dp)
    config.decode_context_parallel_size = 3
    with pytest.raises(ValueError, match="When PCP is enabled, DCP must"):
        config._validate_parallel_config()
    assert config.prefill_context_parallel_size == 2
    assert config.world_size == 4
    assert config.world_size_across_dp == 8
    config.decode_context_parallel_size = 2
    assert config._validate_parallel_config() is config


@pytest.mark.parametrize("pcp,dp", [(1, 2), (2, 1)])
def test_without_combined_pcp_dp(pcp_dp, pcp, dp):
    config = ParallelConfig(
        **(
            pcp_dp
            | {
                "prefill_context_parallel_size": pcp,
                "data_parallel_size": dp,
                "data_parallel_size_local": dp,
            }
        )
    )
    assert config.prefill_context_parallel_size == pcp
    assert config.data_parallel_size == dp
    assert config.world_size == 2 * pcp
