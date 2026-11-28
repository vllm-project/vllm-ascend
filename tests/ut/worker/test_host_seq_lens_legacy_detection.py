#
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
#
"""UTs for the legacy exact host seq-lens detection (device-native default).

Covers the backend registry classification and the init-time detection
guards: PP count exchange, PCP/DCP sharding, the force flag, and the
target/draft backend scan.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
from vllm_ascend.attention.dsa_v1 import AscendDSABackend
from vllm_ascend.attention.fa_v1 import AscendFlashAttentionBackend
from vllm_ascend.attention.host_seq_lens import backend_needs_exact_host_seq_lens
from vllm_ascend.attention.mla_v1 import AscendMLABackend
from vllm_ascend.attention.sfa_v1 import AscendSFABackend
from vllm_ascend.models.qwen4_exp.qsa import AscendQSABackend
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


def _group(backend: type) -> Any:
    return SimpleNamespace(backend=backend)


def _parallel_config(pcp: int = 1, dcp: int = 1, pp: int = 1) -> Any:
    return SimpleNamespace(
        prefill_context_parallel_size=pcp,
        decode_context_parallel_size=dcp,
        pipeline_parallel_size=pp,
    )


def _runner(
    backends: list[type],
    draft_backends: list[type] | None = None,
    dspark_backends: dict[str, type] | None = None,
    *,
    sync_spec_pp_cpu_counts: bool = False,
    force_exact_host_seq_lens: bool = False,
    parallel_config: Any | None = None,
) -> Any:
    speculator = SimpleNamespace(
        attn_groups=[[_group(b) for b in (draft_backends or [])]],
        attn_backends=dspark_backends or {},
    )
    return SimpleNamespace(
        attn_groups=[[_group(b) for b in backends]],
        speculator=speculator,
        sync_spec_pp_cpu_counts=sync_spec_pp_cpu_counts,
        parallel_config=parallel_config or _parallel_config(),
        ascend_config=SimpleNamespace(force_exact_host_seq_lens=force_exact_host_seq_lens),
    )


@pytest.mark.parametrize(
    "backend_cls,expected",
    [
        # Legacy families.
        (AscendAttentionBackend, True),
        (AscendMLABackend, True),
        # Device-native override inside the FIA family.
        (AscendFlashAttentionBackend, False),
        # Device-native backends outside the legacy families.
        (AscendDSABackend, False),
        (AscendSFABackend, False),
        (AscendQSABackend, False),
    ],
)
def test_backend_registry_classification(backend_cls: type, expected: bool):
    assert backend_needs_exact_host_seq_lens(backend_cls) is expected


def test_future_fia_subclass_stays_legacy():
    """A new FIA-family subclass without its own build() must stay legacy."""

    class FutureFIABackend(AscendAttentionBackend):
        pass

    assert backend_needs_exact_host_seq_lens(FutureFIABackend)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sync_spec_pp_cpu_counts", True),
        ("force_exact_host_seq_lens", True),
    ],
)
def test_detection_guards_force_legacy(field: str, value: bool):
    runner = _runner([AscendDSABackend], **{field: value})
    assert NPUModelRunner._detect_legacy_exact_host_seq_lens(runner) is True


@pytest.mark.parametrize(
    "pcp,dcp",
    [
        (2, 1),
        (1, 2),
    ],
)
def test_detection_context_parallelism_forces_legacy(pcp: int, dcp: int):
    runner = _runner([AscendDSABackend], parallel_config=_parallel_config(pcp=pcp, dcp=dcp))
    assert NPUModelRunner._detect_legacy_exact_host_seq_lens(runner) is True


def test_detection_target_legacy_backend_forces_legacy():
    runner = _runner([AscendDSABackend, AscendAttentionBackend])
    assert NPUModelRunner._detect_legacy_exact_host_seq_lens(runner) is True


def test_detection_tda_draft_legacy_backend_forces_legacy():
    runner = _runner([AscendDSABackend], draft_backends=[AscendMLABackend])
    assert NPUModelRunner._detect_legacy_exact_host_seq_lens(runner) is True


def test_detection_dspark_draft_legacy_backend_forces_legacy():
    runner = _runner([AscendDSABackend], dspark_backends={"layer": AscendMLABackend})
    assert NPUModelRunner._detect_legacy_exact_host_seq_lens(runner) is True


def test_detection_device_native_stack_skips_sync():
    runner = _runner(
        [AscendDSABackend],
        draft_backends=[AscendFlashAttentionBackend],
        dspark_backends={"layer": AscendFlashAttentionBackend},
    )
    assert NPUModelRunner._detect_legacy_exact_host_seq_lens(runner) is False


def test_detection_no_speculator_device_native_target_skips_sync():
    runner = _runner([AscendDSABackend])
    runner.speculator = None
    assert NPUModelRunner._detect_legacy_exact_host_seq_lens(runner) is False
