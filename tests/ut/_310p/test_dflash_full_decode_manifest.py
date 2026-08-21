# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import importlib
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def _manifest_module():
    try:
        return importlib.import_module("vllm_ascend._310p.dflash_full_decode_manifest")
    except ModuleNotFoundError:
        pytest.fail("310P DFlash FULL_DECODE_ONLY capture manifest is missing")


def _config(method: str = "dflash"):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method=method,
            num_speculative_tokens=15,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            cudagraph_capture_sizes=[64, 32, 16],
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=16),
    )


def _record_complete_manifest(manifest, rank: int = 0) -> None:
    for component in ("target", "draft"):
        for tokens in (16, 32, 64):
            manifest.record_full_decode_capture(
                component=component,
                local_rank=rank,
                runtime_mode=CUDAGraphMode.FULL,
                descriptor=BatchDescriptor(
                    num_tokens=tokens,
                    num_reqs=tokens // 16,
                    uniform=True,
                ),
                capture_count=1,
                warmup_replay_count=1,
                output_bound=True,
                contract_validated=True,
            )


def test_complete_manifest_contains_target_and_draft_for_every_descriptor():
    manifest = _manifest_module()
    manifest.reset_full_decode_capture_manifest()
    _record_complete_manifest(manifest)

    with patch(
        "vllm_ascend._310p.dflash_full_decode_only.is_310p",
        return_value=True,
    ):
        records = manifest.validate_full_decode_capture_manifest(
            _config(),
            local_rank=0,
        )

    assert len(records) == 6
    assert {key.component for key in records} == {"target", "draft"}
    assert {key.descriptor.num_tokens for key in records} == {16, 32, 64}
    assert all(key.local_rank == 0 for key in records)
    assert all(key.graph_mode is CUDAGraphMode.FULL for key in records)


def test_manifest_validation_fails_if_a_component_descriptor_is_missing():
    manifest = _manifest_module()
    manifest.reset_full_decode_capture_manifest()
    _record_complete_manifest(manifest)
    missing_key = next(
        key
        for key in manifest.get_full_decode_capture_manifest()
        if key.component == "draft" and key.descriptor.num_tokens == 32
    )
    manifest.remove_full_decode_capture_for_test(missing_key)

    with (
        patch(
            "vllm_ascend._310p.dflash_full_decode_only.is_310p",
            return_value=True,
        ),
        pytest.raises(
            manifest.DFlashFullDecodeManifestError,
            match="missing.*draft.*32",
        ),
    ):
        manifest.validate_full_decode_capture_manifest(_config(), local_rank=0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("capture_count", 0),
        ("warmup_replay_count", 0),
        ("output_bound", False),
        ("contract_validated", False),
    ],
)
def test_manifest_rejects_incomplete_capture(field: str, value: object):
    manifest = _manifest_module()
    manifest.reset_full_decode_capture_manifest()
    values = {
        "capture_count": 1,
        "warmup_replay_count": 1,
        "output_bound": True,
        "contract_validated": True,
    }
    values[field] = value

    with pytest.raises(
        manifest.DFlashFullDecodeManifestError,
        match=field,
    ):
        manifest.record_full_decode_capture(
            component="target",
            local_rank=0,
            runtime_mode=CUDAGraphMode.FULL,
            descriptor=BatchDescriptor(
                num_tokens=16,
                num_reqs=1,
                uniform=True,
            ),
            **values,
        )


def test_manifest_validation_is_inactive_outside_exact_scope():
    manifest = _manifest_module()
    manifest.reset_full_decode_capture_manifest()

    with patch(
        "vllm_ascend._310p.dflash_full_decode_only.is_310p",
        return_value=True,
    ):
        assert (
            manifest.validate_full_decode_capture_manifest(
                _config(method="mtp"),
                local_rank=0,
            )
            == {}
        )


def test_model_capture_validates_local_manifest_before_returning():
    runner = object.__new__(NPUModelRunner)
    runner.vllm_config = _config()
    runner.encoder_cudagraph_manager = None

    with (
        patch(
            "vllm_ascend.worker.model_runner_v1._get_gpu_model_runner_module_name",
            return_value="vllm.v1.worker.gpu_model_runner",
        ),
        patch(
            "vllm_ascend.worker.model_runner_v1._torch_cuda_wrapper",
            return_value=nullcontext(),
        ),
        patch(
            "vllm_ascend.worker.model_runner_v1._replace_gpu_model_runner_function_wrapper",
            return_value=nullcontext(),
        ),
        patch.object(GPUModelRunner, "capture_model", return_value=123),
        patch(
            "vllm_ascend.worker.model_runner_v1.get_full_decode_local_rank",
            return_value=2,
            create=True,
        ),
        patch(
            "vllm_ascend.worker.model_runner_v1.validate_full_decode_capture_manifest",
            create=True,
        ) as validate_manifest,
    ):
        captured_bytes = runner.capture_model()

    assert captured_bytes == 123
    validate_manifest.assert_called_once_with(
        runner.vllm_config,
        local_rank=2,
    )


def test_model_capture_resets_manifest_between_capture_sessions():
    manifest = _manifest_module()
    manifest.reset_full_decode_capture_manifest()
    runner = object.__new__(NPUModelRunner)
    runner.vllm_config = _config()
    runner.vllm_config.compilation_config.cudagraph_capture_sizes = [16]
    runner.encoder_cudagraph_manager = None

    def capture_current_descriptors(model_runner):
        for component in ("target", "draft"):
            for tokens in model_runner.vllm_config.compilation_config.cudagraph_capture_sizes:
                manifest.record_full_decode_capture(
                    component=component,
                    local_rank=0,
                    runtime_mode=CUDAGraphMode.FULL,
                    descriptor=BatchDescriptor(
                        num_tokens=tokens,
                        num_reqs=tokens // 16,
                        uniform=True,
                    ),
                    capture_count=1,
                    warmup_replay_count=1,
                    output_bound=True,
                    contract_validated=True,
                )
        return 123

    with (
        patch(
            "vllm_ascend.worker.model_runner_v1._get_gpu_model_runner_module_name",
            return_value="vllm.v1.worker.gpu_model_runner",
        ),
        patch(
            "vllm_ascend.worker.model_runner_v1._torch_cuda_wrapper",
            return_value=nullcontext(),
        ),
        patch(
            "vllm_ascend.worker.model_runner_v1._replace_gpu_model_runner_function_wrapper",
            return_value=nullcontext(),
        ),
        patch.object(
            GPUModelRunner,
            "capture_model",
            side_effect=capture_current_descriptors,
        ),
        patch(
            "vllm_ascend._310p.dflash_full_decode_only.is_310p",
            return_value=True,
        ),
    ):
        assert runner.capture_model() == 123
        runner.vllm_config.compilation_config.cudagraph_capture_sizes = [32]
        assert runner.capture_model() == 123

    records = manifest.get_full_decode_capture_manifest()
    assert len(records) == 2
    assert {key.component for key in records} == {"target", "draft"}
    assert {key.descriptor.num_tokens for key in records} == {32}
