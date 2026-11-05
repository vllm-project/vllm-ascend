# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from vllm_ascend.ascend_config import AscendConfig, FastEEPConfig
from vllm_ascend.platform import _validate_eplb_config


def test_additional_config_accepts_fasteep_options():
    config = AscendConfig(
        sparse_kv_offload_config=SimpleNamespace(enabled=False),
        fasteep={"communication": "alltoall", "verify_migration": True},
    )
    assert config.fasteep.communication == "alltoall"
    assert config.fasteep.verify_migration is True
    assert FastEEPConfig().communication == "allgather"


@pytest.mark.parametrize("options", [{"communication": "unknown"}, {"verify_migration": "invalid"}, {"typo": True}])
def test_invalid_fasteep_options_fail_at_configuration(options):
    with pytest.raises(ValidationError):
        FastEEPConfig(**options)


def test_v2_preserves_fasteep_checkpoint_placement():
    placement = {"dynamic_eplb": False, "expert_map_path": "placement.json", "num_redundant_experts": 128}
    config = SimpleNamespace(
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(
            enable_eplb=False,
            worker_extension_cls="vllm_ascend.distributed.fasteep.runtime.FastEEPWorkerExtension",
        ),
        additional_config={"eplb_config": dict(placement)},
    )
    _validate_eplb_config(config)
    assert config.additional_config["eplb_config"] == placement
