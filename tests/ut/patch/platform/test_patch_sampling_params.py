# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import runpy
from pathlib import Path

import msgspec
import pytest
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest


@pytest.fixture
def patched_validator(monkeypatch):
    original = SamplingParams._verify_extra_args
    validated_args = []

    def record_validation(self):
        validated_args.append(self.extra_args)
        return original(self)

    monkeypatch.setattr(SamplingParams, "_verify_extra_args", record_validation)
    patch_path = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/platform/patch_sampling_params.py"
    runpy.run_path(str(patch_path))
    return validated_args


def transfer_params(blocks_per_group=8):
    return {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_block_ids": [list(range(blocks_per_group)) for _ in range(4)],
        "remote_engine_id": "prefill-engine",
        "remote_request_id": "prefill-request",
        "remote_host": "127.0.0.1",
        "remote_port": 36011,
        "tp_size": 4,
    }


def test_transfer_only_decode_skips_deep_validation(patched_validator):
    # Typed MessagePack decode runs __post_init__, as on EngineCore ingress.
    extra_args = {"kv_transfer_params": transfer_params(blocks_per_group=16384)}
    params = SamplingParams(extra_args=extra_args)
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(params), type=SamplingParams)

    assert params.extra_args is extra_args
    assert decoded.extra_args == extra_args
    assert patched_validator == []


def test_engine_core_request_decode_skips_deep_validation(patched_validator):
    params = SamplingParams(extra_args={"kv_transfer_params": transfer_params()})
    request = EngineCoreRequest(
        request_id="decode-request",
        prompt_token_ids=[1],
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(request), type=EngineCoreRequest)

    assert decoded.sampling_params.extra_args == params.extra_args
    assert patched_validator == []


@pytest.mark.parametrize("include_transfer", [False, True])
def test_other_extensions_keep_upstream_validation(patched_validator, include_transfer):
    extra_args = {"custom": {"nested": [1, 2**64 - 1]}}
    if include_transfer:
        extra_args["kv_transfer_params"] = transfer_params()
    params = SamplingParams(extra_args=extra_args)
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(params), type=SamplingParams)

    assert decoded.extra_args == extra_args
    assert patched_validator == [extra_args, extra_args]


@pytest.mark.parametrize("extension", ["custom", "ec_transfer_params"])
def test_other_extensions_still_reject_transport_overflow(patched_validator, extension):
    extra_args = {
        "kv_transfer_params": transfer_params(),
        extension: {"nested": [2**64]},
    }
    with pytest.raises(VLLMValidationError, match="extra_args integers"):
        SamplingParams(extra_args=extra_args)
    assert patched_validator == [extra_args]


def test_transfer_only_retains_other_sampling_checks(patched_validator):
    with pytest.raises(VLLMValidationError, match="temperature"):
        SamplingParams(temperature=-1, extra_args={"kv_transfer_params": transfer_params()})
