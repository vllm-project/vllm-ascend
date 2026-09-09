# SPDX-License-Identifier: Apache-2.0
import pytest

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.kvpp_utils import run_basic_comparison


@pytest.mark.e2e_model("vllm-ascend/DeepSeek-R1-0528-W8A8")
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="kvpp,mtp",
    parallel="TP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="eager",
)
@wait_until_npu_memory_free()
def test_kvpp_mtp(monkeypatch):
    run_basic_comparison(monkeypatch, "vllm-ascend/DeepSeek-R1-0528-W8A8", tp=16, mtp=True, quantization="ascend")


@pytest.mark.e2e_model("Eco-Tech/GLM-5.2-w4a8c8")
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="kvpp,sfa_dsa",
    parallel="TP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W4A8",
    graph_mode="eager",
)
@wait_until_npu_memory_free()
def test_kvpp_indexer_c8(monkeypatch):
    run_basic_comparison(
        monkeypatch,
        "Eco-Tech/GLM-5.2-w4a8c8",
        tp=16,
        indexer_c8=True,
        quantization="ascend",
        block_size=128,
        additional_config={"enable_sparse_sfa_c8": False, "enable_sparse_li_c8": True},
    )


@pytest.mark.e2e_model("vllm-ascend/DeepSeek-V3.2-W8A8-Pruning")
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="kvpp",
    parallel="TP,PP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="eager",
)
@wait_until_npu_memory_free()
def test_kvpp_pp(monkeypatch):
    run_basic_comparison(monkeypatch, "vllm-ascend/DeepSeek-V3.2-W8A8-Pruning", tp=2, pp=2, quantization="ascend")
