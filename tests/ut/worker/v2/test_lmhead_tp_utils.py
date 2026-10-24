# SPDX-License-Identifier: Apache-2.0

import ast
import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.worker.v2.spec_decode import lmhead_tp_utils
from vllm_ascend.worker.v2.spec_decode.target_dependent_ar import speculator as ar_module


@pytest.fixture(params=[False, True], ids=["main", "v0.32.0"])
def draft_mixin(request):
    """Load both versioned sampling signatures without a second vLLM install."""
    spec = importlib.util.spec_from_file_location("isolated_lmhead_tp_utils", lmhead_tp_utils.__file__)
    assert spec is not None and spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    with patch("vllm_ascend.utils.vllm_version_is", return_value=request.param):
        spec.loader.exec_module(module)
    module.lmhead_tp_enable = lambda: True
    return module.LmheadTPDraftSamplingMixin, request.param


def make_drafter(mixin, *, dcp=1, method="mtp"):
    class Parent:
        sample_rows: list[int]

        def sample_draft(self, hidden, *args, **kwargs):
            self.sample_rows.append(hidden.shape[0])
            return torch.arange(hidden.shape[0])

    Drafter = type("Drafter", (mixin, Parent), {})
    draft = Drafter()
    draft.method = method
    draft.max_num_reqs = 8
    draft.num_speculative_steps = 3
    draft.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(decode_context_parallel_size=dcp))
    draft.sample_rows = []
    return draft


@pytest.mark.parametrize("local_reqs", [0, 2, 5])
@pytest.mark.parametrize("steps", [1, 3, 5])
def test_draft_samples_dp_wide_request_rows_not_target_tokens(draft_mixin, local_reqs, steps):
    mixin, _ = draft_mixin
    draft = make_drafter(mixin)
    draft.num_speculative_steps = steps
    state = SimpleNamespace(num_reqs=5, num_tokens_across_dp=MagicMock())
    with draft._lmhead_pad_size_context(state):
        assert draft._lmhead_pad_size == 5
        out = draft.sample_draft(torch.ones(local_reqs, 4), None, None, None, None, None, None)
        assert draft.sample_rows == [5]
        torch.testing.assert_close(out, torch.arange(local_reqs))
        # Inner capture/prefill/decode scopes must retain the shared DP bound.
        with draft._lmhead_pad_size_context(capture_num_reqs=local_reqs):
            assert draft._lmhead_pad_size == 5
    assert draft._lmhead_pad_size is None
    state.num_tokens_across_dp.max.assert_not_called()


@pytest.mark.parametrize("requests,expected", [(None, 8), (0, 1), (1, 1), (2, 2), (5, 5), (8, 8), (64, 8)])
def test_capture_request_buckets_and_conservative_fallback(draft_mixin, requests, expected):
    mixin, _ = draft_mixin
    draft = make_drafter(mixin)
    with draft._lmhead_pad_size_context(capture_num_reqs=requests):
        assert draft._lmhead_pad_size == expected
    assert draft._lmhead_pad_size is None


def test_missing_shared_request_bound_does_not_use_local_count(draft_mixin):
    mixin, _ = draft_mixin
    draft = make_drafter(mixin)
    state = SimpleNamespace(num_tokens_across_dp=torch.tensor([4, 24]))
    with draft._lmhead_pad_size_context(state, capture_num_reqs=2):
        assert draft._lmhead_pad_size == 8


def test_dcp_greater_than_one_keeps_original_capacity(draft_mixin):
    mixin, _ = draft_mixin
    draft = make_drafter(mixin, dcp=2)
    with draft._lmhead_pad_size_context(SimpleNamespace(num_reqs=2), capture_num_reqs=2):
        assert draft._lmhead_pad_size == 32


def test_context_restores_padding_after_failure(draft_mixin):
    mixin, _ = draft_mixin
    draft = make_drafter(mixin)
    with pytest.raises(ValueError, match="failed"), draft._lmhead_pad_size_context(capture_num_reqs=2):
        raise ValueError("failed")
    assert draft._lmhead_pad_size is None
    draft._lmhead_pad_size = 6
    with draft._lmhead_pad_size_context(capture_num_reqs=2):
        assert draft._lmhead_pad_size == 6
    assert draft._lmhead_pad_size == 6


@pytest.mark.parametrize(
    "dp_size,replicated,shared_reqs,expected",
    [(1, False, None, 2), (2, False, 5, 5), (2, False, None, 8), (2, True, 5, 8)],
)
def test_propose_uses_request_bounds(draft_mixin, dp_size, replicated, shared_reqs, expected):
    mixin, _ = draft_mixin
    tree = ast.parse(Path(ar_module.__file__).read_text())
    ar_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendTargetDependentARSpeculator"
    )
    propose = next(node for node in ar_class.body if isinstance(node, ast.FunctionDef) and node.name == "propose")

    class Parent:
        _lmhead_pad_size: int | None

        def propose(self, *args, **kwargs):
            assert self._lmhead_pad_size == expected
            return "drafted"

    # Exercise the current propose signature with both supported sampling lanes.
    # Parsing supplies version-specific AST fields, including 3.12's type_params.
    host = ast.parse("class Host(Mixin, Parent): pass").body[0]
    assert isinstance(host, ast.ClassDef)
    host.body = [propose]
    namespace = {
        "Mixin": mixin,
        "Parent": Parent,
        "torch": torch,
        "Any": Any,
        "InputBatch": object,
        "disable_target_pcp_for_replicated_draft": lambda _: nullcontext(),
        "build_attn_metadata_wrapper": nullcontext,
        "torch_gather_wrapper": nullcontext,
    }
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[host], type_ignores=[])), ar_module.__file__, "exec"),
        namespace,
    )
    draft = namespace["Host"]()
    draft.__dict__.update(make_drafter(mixin).__dict__)
    draft.dp_size = dp_size
    draft.replicated_pcp = replicated
    batch = SimpleNamespace(num_reqs=2, num_tokens_after_padding=24)
    state = None if shared_reqs is None else SimpleNamespace(num_reqs=shared_reqs)
    assert draft.propose(batch, *([None] * 10), dp_sync_state=state) == "drafted"
    assert draft._lmhead_pad_size is None
