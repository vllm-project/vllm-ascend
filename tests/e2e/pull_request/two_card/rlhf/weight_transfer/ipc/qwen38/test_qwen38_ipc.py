# SPDX-License-Identifier: Apache-2.0
"""Four-layer PLE/MRoPE Qwen3.8: TP2, generated weights, unpacked IPC."""

import json
from pathlib import Path

import pytest


def test_qwen38_ipc_unpacked(request):
    if not request.config.getoption("--qwen38-enable-npu"):
        pytest.skip("NPU execution is disabled; enable explicitly after allocating two idle cards")
    model_dir = request.config.getoption("--qwen38-model-dir")
    if not model_dir:
        pytest.fail("Prepare a descriptor and supply --qwen38-model-dir")
    model_dir = Path(model_dir).resolve()

    # Keep hardware-aware imports behind the explicit execution gate.
    from tests.e2e.common.rlhf.weight_transfer.case import WeightUpdateModelCase
    from tests.e2e.common.rlhf.weight_transfer.npu_ipc import run_npu_ipc_weight_transfer_transaction
    from tests.e2e.common.rlhf.weight_transfer.qwen38_manifest import validate_ci_config
    from tests.e2e.common.rlhf.weight_transfer.qwen38_weight_source import Qwen38RandomWeightSource

    validate_ci_config(json.loads((model_dir / "config.json").read_text()))
    case = WeightUpdateModelCase(
        id="qwen3.8-flash-next-4layers-small-ple-mrope-head128",
        model=str(model_dir),
        hf_overrides={},
        # Preserve the multimodal architecture/MRoPE, skip unused tower weights.
        extra_server_args=(
            "--max-num-seqs",
            "4",
            "--limit-mm-per-prompt",
            '{"image":0,"video":0}',
            "--engram-config",
            '{"cpu_offload":false}',
        ),
    )
    run_npu_ipc_weight_transfer_transaction(case, packed=False, source_factory=Qwen38RandomWeightSource)
