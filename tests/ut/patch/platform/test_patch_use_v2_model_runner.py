import pytest
from vllm.config import parallel as parallel_module
from vllm.config import replace
from vllm.config.parallel import ParallelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.config.vllm import VllmConfig

from vllm_ascend.patch.platform import patch_use_v2_model_runner
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


def test_release_pcp_is_not_rejected_as_v2_unsupported_feature(monkeypatch):
    monkeypatch.setattr(patch_use_v2_model_runner, "vllm_version_is", lambda version: version == "0.28.0")
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

    assert unsupported == ["diffusion models"]


@pytest.fixture
def release_pcp_dp(monkeypatch):
    if not vllm_version_is("0.28.0"):
        pytest.skip("The PCP+DP validation workaround is release-only")
    monkeypatch.setattr(patch_use_v2_model_runner.envs, "VLLM_USE_V2_MODEL_RUNNER", True)
    monkeypatch.setattr(patch_use_v2_model_runner.current_platform, "device_name", "npu")
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


@pytest.mark.parametrize("nested", [False, True])
def test_release_pcp_dp_construction_and_replace(release_pcp_dp, monkeypatch, nested):
    # Exercise real Pydantic schemas without model initialization or downloads.
    monkeypatch.setattr(VllmConfig, "__post_init__", lambda self: None)
    monkeypatch.setattr(SpeculativeConfig, "__post_init__", lambda self: None)
    if nested:
        config = VllmConfig(
            parallel_config=release_pcp_dp,
            speculative_config={"method": "eagle3", "num_speculative_tokens": 1},
        ).parallel_config
    else:
        config = ParallelConfig(**release_pcp_dp)
    copied = replace(config)
    wrapped = VllmConfig(parallel_config=copied).parallel_config
    for actual in (config, copied, wrapped):
        assert actual.prefill_context_parallel_size == 2
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
        ({"dcp_comm_backend": "a2a"}, "requires decode_context_parallel_size > 1"),
    ],
)
def test_release_pcp_dp_preserves_other_validation(release_pcp_dp, overrides, error):
    with pytest.raises(ValueError, match=error):
        ParallelConfig(**(release_pcp_dp | overrides))


def test_release_pcp_dp_restores_pcp_after_validation_error(release_pcp_dp):
    config = ParallelConfig(**release_pcp_dp)
    config.dcp_comm_backend = "a2a"
    with pytest.raises(ValueError, match="requires decode_context_parallel_size > 1"):
        config._validate_parallel_config()
    assert config.prefill_context_parallel_size == 2
    assert config.world_size == 4


@pytest.mark.parametrize(
    "device,use_v2,dcp", [("npu", False, 1), ("npu", None, 1), ("cuda", True, 1), ("npu", True, 2)]
)
def test_release_pcp_dp_keeps_unsupported_combinations(release_pcp_dp, monkeypatch, device, use_v2, dcp):
    monkeypatch.setattr(patch_use_v2_model_runner.current_platform, "device_name", device)
    monkeypatch.setattr(patch_use_v2_model_runner.envs, "VLLM_USE_V2_MODEL_RUNNER", use_v2)
    with pytest.raises(ValueError, match="PCP does not support data parallelism yet"):
        ParallelConfig(**(release_pcp_dp | {"decode_context_parallel_size": dcp}))


@pytest.mark.parametrize("pcp,dp", [(1, 2), (2, 1)])
def test_release_without_combined_pcp_dp(release_pcp_dp, pcp, dp):
    config = ParallelConfig(
        **(
            release_pcp_dp
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
