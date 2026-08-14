# SPDX-License-Identifier: Apache-2.0

import argparse

import msgspec
import pytest
from vllm import SamplingParams

from vllm_ascend.patch.platform.patch_reasoning_eos import (
    _parse_reasoning_config,
    update_reasoning_config_cli,
)


@pytest.mark.parametrize("ignore_eos", [False, True])
def test_model_eos_ids_are_independent_from_user_stops(ignore_eos: bool):
    params = SamplingParams(ignore_eos=ignore_eos, stop_token_ids=[7])
    params.update_from_generation_config({"eos_token_id": [2, 3]}, 2)

    assert params.model_eos_token_ids == {2, 3}
    assert 7 not in params.model_eos_token_ids
    assert params._eos_token_id == (None if ignore_eos else 2)
    assert params.clone().model_eos_token_ids == {2, 3}
    restored = msgspec.msgpack.decode(
        msgspec.msgpack.encode(params), type=SamplingParams
    )
    assert restored.model_eos_token_ids == {2, 3}


def test_reasoning_config_parser_accepts_phase_aware_policy():
    config = _parse_reasoning_config('{"reasoning_parser":"deepseek_r1","premature_eos_policy":"mask_in_reasoning"}')

    assert config is not None
    assert config.reasoning_parser == "deepseek_r1"
    assert config.premature_eos_policy == "mask_in_reasoning"


def test_reasoning_config_parser_rejects_invalid_policy():
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_reasoning_config('{"premature_eos_policy":"invalid"}')


def test_reasoning_config_cli_action_is_replaced():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning-config", type=str)

    update_reasoning_config_cli(parser)
    args = parser.parse_args(
        [
            "--reasoning-config",
            '{"premature_eos_policy":"mask_in_reasoning"}',
        ]
    )

    assert args.reasoning_config.premature_eos_policy == "mask_in_reasoning"
