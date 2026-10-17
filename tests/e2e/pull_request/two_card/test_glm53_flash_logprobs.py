# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A3 TP2/EP2, MRV1 + hybrid: real five-layer 0916 checkpoint vs fixed golden."""

import tempfile
from pathlib import Path

import pytest

from tests.e2e.glm53_flash.checkpoint import object_hash, prepare_checkpoint, read_json, write_json
from tests.e2e.glm53_flash.logprobs import validate_golden
from tests.e2e.glm53_flash.runtime import FIXTURES, contract, load_prompts, run_suite
from vllm_ascend import envs


@pytest.mark.e2e_model("GLM-5.3-Flash-W8A8-0916")
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="logprobs,sfa_dsa,chunked_prefill,mixed_lengths",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="eager",
)
def test_glm53_flash_logprobs(pytestconfig: pytest.Config) -> None:
    source_dir = envs.GLM53_FLASH_SOURCE_DIR
    assert source_dir, "Provision the reviewed 0916 checkpoint and set GLM53_FLASH_SOURCE_DIR (no dummy fallback)"
    golden_path = FIXTURES / "golden.json"
    assert golden_path.is_file(), "Missing reviewed golden; run the explicit offline capture/acceptance procedure"
    golden = read_json(golden_path)
    prompts = load_prompts()
    source = read_json(FIXTURES / "source.json")
    assert prompts["tokenizer_sha256"] == source["files"]["tokenizer.json"]["sha256"]
    cache = pytestconfig.cache.mkdir("glm53_flash")
    output = cache / source["source_id"]
    projection = prepare_checkpoint(Path(source_dir), output, source)
    expected_contract = contract(source, projection, prompts)
    validate_golden(golden, expected_contract, {case["name"] for case in prompts["cases"]})
    artifacts_root = Path("tests/outputs/glm53_flash")
    artifacts_root.mkdir(parents=True, exist_ok=True)
    # Unique artifacts survive failures and are uploaded by the selected-tests workflow.
    artifacts = Path(tempfile.mkdtemp(prefix="run-", dir=artifacts_root))
    write_json(artifacts / "contract.json", {**expected_contract, "contract_sha256": object_hash(expected_contract)})
    run_suite(output, prompts, artifacts / "server", golden)
