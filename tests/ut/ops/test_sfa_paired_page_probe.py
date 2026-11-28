# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops import sfa_paired_page as sfa


def _reference(**args):
    pages = args["key"]
    assert pages is args["value"] and pages is args["key_rope"]
    assert args["layout_kv"] == "PA_KR_BSND" and args["return_softmax_lse"]
    page_count, block, _, _ = pages.shape
    storage = pages.view(page_count, -1)
    latent = storage[:, : block * 512].reshape(page_count, block, 512)
    rope = storage[:, block * 512 :].reshape(page_count, block, 64)
    indices = args["sparse_indices"].flatten()
    indices = indices[indices >= 0].long()
    physical = args["block_table"][0, indices // block].long()
    within = indices % block
    values = latent[physical, within].float()
    positional = rope[physical, within].float()
    scores = args["query"].float() @ values.T + args["query_rope"].float() @ positional.T
    maximum = scores.amax(dim=-1, keepdim=True)
    exps = (scores - maximum).exp()
    total = exps.sum(dim=-1, keepdim=True)
    return ((exps / total) @ values).bfloat16(), maximum.transpose(1, 2), total.transpose(1, 2)


@pytest.fixture
def native(monkeypatch):
    op = Mock(side_effect=_reference)
    op.default.has_kernel_for_dispatch_key.return_value = True
    ops = SimpleNamespace(is_sfa_paired_page_available=Mock(return_value=True), npu_sparse_flash_attention=op)
    monkeypatch.setattr(sfa, "import_module", Mock(return_value=SimpleNamespace(__file__="/wheel/vllm_ascend_C.so")))
    monkeypatch.setattr(sfa, "bootstrap_custom_op_env", Mock())
    monkeypatch.setattr(torch.ops, "_C_ascend", ops)
    sfa.paired_page_sfa_available.cache_clear()
    yield ops
    sfa.paired_page_sfa_available.cache_clear()


def test_probe_checks_kernel_once_per_device(native):
    for device in ("cpu:0", "cpu:0", "cpu:1"):
        assert sfa.paired_page_sfa_available(torch.device(device))
    assert native.is_sfa_paired_page_available.call_count == 2
    assert [call.kwargs["query"].size(1) for call in native.npu_sparse_flash_attention.call_args_list] == [
        128,
        32,
        128,
        32,
    ]


@pytest.mark.parametrize("mismatch", ["binding", "symbols", "tiler", "output", "maximum", "sum"])
def test_probe_disables_incompatible_installation(native, mismatch):
    op = native.npu_sparse_flash_attention
    if mismatch == "binding":
        del native.is_sfa_paired_page_available
    elif mismatch == "symbols":
        native.is_sfa_paired_page_available.return_value = False
    elif mismatch == "tiler":
        op.side_effect = RuntimeError("call aclnnVllmSfaPairedPageGetWorkspaceSize failed with status 561002")
    else:

        def corrupt(**args):
            outputs = _reference(**args)
            # A defective split-G kernel may reuse the first 64 heads' row or LSE.
            tensor = outputs[{"output": 0, "maximum": 1, "sum": 2}[mismatch]]
            by_head = tensor.reshape(128, -1)
            by_head[64].copy_(by_head[0])
            return outputs

        op.side_effect = corrupt
    assert not sfa.paired_page_sfa_available(torch.device("cpu"))
    assert op.call_count == (0 if mismatch in ("binding", "symbols") else 1)


def test_probe_checks_unsplit_variant_after_split_g_succeeds(native):
    def fail_unsplit(**args):
        outputs = _reference(**args)
        if args["query"].size(1) == 32:
            outputs[0].zero_()
        return outputs

    native.npu_sparse_flash_attention.side_effect = fail_unsplit
    assert not sfa.paired_page_sfa_available(torch.device("cpu"))
    assert native.npu_sparse_flash_attention.call_count == 2


@pytest.mark.parametrize(
    "message",
    [
        "call aclnnVllmSfaPairedPage failed, detail: device",
        "call aclnnVllmSfaPairedPageGetWorkspaceSize failed with status 361001, detail: device",
    ],
)
def test_probe_propagates_execution_failure(native, message):
    native.npu_sparse_flash_attention.side_effect = RuntimeError(message)
    with pytest.raises(RuntimeError, match="detail: device"):
        sfa.paired_page_sfa_available(torch.device("cpu"))
