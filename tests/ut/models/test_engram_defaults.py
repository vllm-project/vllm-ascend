# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.models.deepseek_v41 import model as model_mod


@pytest.mark.parametrize("device", [AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5])
@pytest.mark.parametrize("explicit", [None, False, True])
def test_engram_overlap_hardware_default_and_explicit_override(monkeypatch, device, explicit):
    monkeypatch.setattr("vllm_ascend.ascend_config.get_current_hardware_profile", lambda: get_hardware_profile(device))
    kwargs = {} if explicit is None else {"multistream_engram_overlap": explicit}
    config = AscendConfig(sparse_kv_offload_config=SimpleNamespace(enabled=False), **kwargs)
    expected = device == AscendDeviceType.A5 if explicit is None else explicit
    assert config.multistream_engram_overlap is expected


@pytest.mark.parametrize(
    "change,prepare,pipeline",
    [
        ({}, True, True),
        ({"backend": "uva"}, False, False),
        ({"cpu": False}, False, False),
        ({"shm": False}, False, False),
        ({"v2": False}, False, False),
        ({"layers": ()}, False, False),
        ({"layers": (1,)}, True, False),
        ({"layers": (2, 14)}, True, False),
        ({"tp": 2}, True, False),
        ({"pp": 2}, True, False),
        ({"sp": True}, True, False),
        ({"overlap": False}, True, False),
    ],
)
def test_aicpu_pipeline_is_automatic_only_for_supported_layout(monkeypatch, change, prepare, pipeline):
    options = dict(
        backend="aicpu_urma_cube_hbm", cpu=True, shm=True, v2=True, layers=(1, 14), tp=1, pp=1, sp=False, overlap=True
    )
    options.update(change)
    config = AscendConfig(
        sparse_kv_offload_config=SimpleNamespace(enabled=False),
        engram_lookup_backend=options["backend"],
        multistream_engram_overlap=options["overlap"],
    )
    monkeypatch.setattr(model_mod, "get_ascend_config", lambda: config)
    monkeypatch.setattr(model_mod.envs, "VLLM_USE_V2_MODEL_RUNNER", options["v2"])
    monkeypatch.setattr(model_mod, "get_tensor_model_parallel_world_size", lambda: options["tp"])
    monkeypatch.setattr(model_mod, "get_pp_group", lambda: SimpleNamespace(world_size=options["pp"]))
    model = SimpleNamespace(
        has_engram=bool(options["layers"]),
        engram_dp_shared_memory=options["shm"],
        config=SimpleNamespace(engram_layer_ids=options["layers"]),
        use_sequence_parallel=options["sp"],
    )
    model_mod.DeepseekV41Model._configure_engram_pipeline(model, options["cpu"])
    assert model._engram_layer_major_hashes is prepare
    assert model._engram_late_lookup_enabled is pipeline
