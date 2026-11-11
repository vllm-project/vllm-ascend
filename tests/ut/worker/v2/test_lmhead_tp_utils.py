# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression tests for draft LM-head TP alignment and sampling."""

import ast
import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, create_autospec

import pytest
import torch
from vllm.v1.worker.gpu.spec_decode import speculator as upstream_speculator

from vllm_ascend.worker.v2.spec_decode import lmhead_tp_utils
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import AscendDFlash2Speculator
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator
from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator
from vllm_ascend.worker.v2.spec_decode.gemma4.speculator import AscendGemma4Speculator
from vllm_ascend.worker.v2.spec_decode.lilicorr.speculator import AscendLiLiCorrSpeculator
from vllm_ascend.worker.v2.spec_decode.mtp.speculator import AscendMTPSpeculator
from vllm_ascend.worker.v2.spec_decode.multi_module_mtp.speculator import AscendMultiModuleMTPSpeculator
from vllm_ascend.worker.v2.spec_decode.standalone_ar.speculator import AscendStandaloneARSpeculator
from vllm_ascend.worker.v2.spec_decode.target_dependent_ar import speculator as ar_module

MAX_REQS = 8
NUM_STEPS = 3
HIDDEN_SIZE = 4
VOCAB_SIZE = 5
CACHE_FILL = -37.0


def _speculator(cls=AscendMTPSpeculator):
    # Exercise production methods without loading weights or initializing NPUs.
    spec = cls.__new__(cls)
    spec.max_num_reqs = MAX_REQS
    spec.num_speculative_steps = NUM_STEPS
    spec.use_local_argmax_reduction = False
    spec.enable_adaptive_verification = False
    spec.use_fp64_gumbel = False
    spec.draft_watermarker = None
    spec.acceptance_estimator = None
    weight = torch.arange(HIDDEN_SIZE * VOCAB_SIZE, dtype=torch.float32).reshape(HIDDEN_SIZE, VOCAB_SIZE) / 10
    spec.model = SimpleNamespace(compute_logits=MagicMock(side_effect=lambda hidden: hidden @ weight))
    return spec, weight


def _draft_cache():
    cache_cls = getattr(upstream_speculator, "DraftLogitsCache", None)
    if cache_cls is None:
        values = torch.full((MAX_REQS, NUM_STEPS, VOCAB_SIZE), CACHE_FILL)
        return values, values
    # Use the cache API supplied by the installed upstream, not a release number.
    cache = cache_cls(MAX_REQS, NUM_STEPS, VOCAB_SIZE, dtype=torch.float32, fill=CACHE_FILL, device="cpu")
    return cache, cache.values


@pytest.mark.parametrize(
    "speculator_cls,rows_per_req,num_reqs,graph_padding,shared_reqs",
    [
        (AscendMTPSpeculator, 1, 4, 0, None),
        (AscendEagleSpeculator, 1, 4, 0, None),
        (AscendGemma4Speculator, 1, 4, 0, None),
        (AscendDFlashSpeculator, NUM_STEPS, 4, 0, None),
        (AscendDFlashSpeculator, NUM_STEPS, MAX_REQS, 0, None),
        # Original graph-padding rows must survive the LM-head-only crop.
        (AscendMTPSpeculator, 1, 4, 2, None),
        # Preserve main's smaller DP-wide MTP bound without leaking TP rows.
        (AscendMTPSpeculator, 1, 4, 0, 6),
    ],
)
@pytest.mark.parametrize("probabilistic", [False, True])
@pytest.mark.parametrize("lmhead_enabled", [False, True])
def test_sample_draft_crops_before_sampling(
    monkeypatch, speculator_cls, rows_per_req, num_reqs, graph_padding, shared_reqs, probabilistic, lmhead_enabled
):
    spec, weight = _speculator(speculator_cls)
    num_rows = (num_reqs + graph_padding) * rows_per_req
    hidden = torch.arange(num_rows * HIDDEN_SIZE, dtype=torch.float32).reshape(num_rows, HIDDEN_SIZE) / 31 + 1
    expected_logits = hidden @ weight
    slots = torch.arange(MAX_REQS - num_reqs, MAX_REQS)
    idx_mapping = torch.cat((slots, torch.full((graph_padding,), -1))).repeat_interleave(rows_per_req)
    positions = torch.arange(num_rows) + 100
    temperature = torch.linspace(0.5, 1.2, MAX_REQS)
    seeds = torch.arange(MAX_REQS) + 1024
    step = torch.arange(NUM_STEPS, dtype=torch.int32).repeat(num_reqs) if rows_per_req > 1 else torch.tensor(1)
    cache, values = _draft_cache()
    expected_cache = values.clone()

    def sample(logits, mapping, temp, rng_seeds, sample_positions, **kwargs):
        # Fail at the consumer boundary if a regression leaks even one TP row.
        assert logits.shape == (num_rows, VOCAB_SIZE)
        torch.testing.assert_close(logits, expected_logits)
        assert mapping is idx_mapping
        assert sample_positions is positions
        assert temp is temperature
        assert rng_seeds is seeds
        assert kwargs["logits_cache"] is values
        assert kwargs["logits_cache_col"] is step
        assert kwargs["apply_temperature"] is True
        assert kwargs["is_drafting"] is True
        assert kwargs["use_fp64"] is False
        if isinstance(cache, torch.Tensor):
            assert "logits_cache_row_mapping" not in kwargs
        else:
            assert kwargs["logits_cache_row_mapping"] is None
        # Minimal CPU consumer contract, not a replacement probability sampler.
        # Check the full row count above before indexing; zip/truncation would
        # hide extra rows and let the old pad -> sample -> crop behavior pass.
        for row in range(num_rows):
            slot = int(mapping[row])
            if slot >= 0:
                col = int(step[row] if step.ndim else step)
                values[slot, col].copy_(logits[row])
        return logits.argmax(dim=-1)

    sampler = create_autospec(upstream_speculator.gumbel_sample, side_effect=sample)
    monkeypatch.setattr(upstream_speculator, "gumbel_sample", sampler)
    monkeypatch.setattr(lmhead_tp_utils, "lmhead_tp_enable", lambda: lmhead_enabled)

    # Keep both the real upstream sample_draft and the real Ascend logits hook.
    context = nullcontext()
    if shared_reqs is not None:
        spec.method = "mtp"
        spec.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(decode_context_parallel_size=1))
        context = spec._lmhead_pad_size_context(SimpleNamespace(num_reqs=shared_reqs))
    with context:
        tokens = spec.sample_draft(
            hidden, positions, idx_mapping, temperature, seeds, step, cache if probabilistic else None
        )

    compute_input = spec.model.compute_logits.call_args.args[0]
    capacity = shared_reqs if shared_reqs is not None else MAX_REQS * (NUM_STEPS + 1)
    if lmhead_enabled:
        assert compute_input.shape == (capacity, HIDDEN_SIZE)
        torch.testing.assert_close(compute_input[:num_rows], hidden)
        assert torch.count_nonzero(compute_input[num_rows:]) == 0
    else:
        assert compute_input is hidden
    torch.testing.assert_close(tokens, expected_logits.argmax(dim=-1))
    if probabilistic:
        sampler.assert_called_once()
        # Independently specify the request/step destinations; unused slots,
        # other steps and graph-padding rows must not change the sentinel cache.
        for req, slot in enumerate(slots.tolist()):
            for offset in range(rows_per_req):
                row = req * rows_per_req + offset
                expected_cache[slot, offset if rows_per_req > 1 else 1] = expected_logits[row]
    else:
        sampler.assert_not_called()
    torch.testing.assert_close(values, expected_cache)


@pytest.mark.parametrize("num_rows", [0, MAX_REQS * (NUM_STEPS + 1), MAX_REQS * (NUM_STEPS + 1) + 1])
def test_compute_draft_logits_capacity_boundaries(monkeypatch, num_rows):
    spec, weight = _speculator()
    hidden = torch.arange(num_rows * HIDDEN_SIZE, dtype=torch.float32).reshape(num_rows, HIDDEN_SIZE)
    monkeypatch.setattr(lmhead_tp_utils, "lmhead_tp_enable", lambda: True)
    # Capacity boundaries concern the logits hook, not 32 independent requests.
    if num_rows > MAX_REQS * (NUM_STEPS + 1):
        with pytest.raises(ValueError, match="group-agreed capacity"):
            spec.compute_draft_logits(hidden, spec_step_idx=0)
        spec.model.compute_logits.assert_not_called()
    else:
        logits = spec.compute_draft_logits(hidden, spec_step_idx=0)
        torch.testing.assert_close(logits, hidden @ weight)
        assert spec.model.compute_logits.call_args.args[0].shape == (MAX_REQS * (NUM_STEPS + 1), HIDDEN_SIZE)


@pytest.mark.parametrize("lmhead_enabled", [False, True])
def test_compute_draft_logits_forwards_module_index(monkeypatch, lmhead_enabled):
    spec, _ = _speculator()
    hidden = torch.ones(4, HIDDEN_SIZE)
    rows = MAX_REQS * (NUM_STEPS + 1) if lmhead_enabled else hidden.shape[0]
    parent_logits = torch.arange(rows * VOCAB_SIZE).reshape(rows, VOCAB_SIZE)
    compute = create_autospec(upstream_speculator.DraftModelSpeculator.compute_draft_logits, return_value=parent_logits)
    monkeypatch.setattr(upstream_speculator.DraftModelSpeculator, "compute_draft_logits", compute)
    monkeypatch.setattr(lmhead_tp_utils, "lmhead_tp_enable", lambda: lmhead_enabled)
    logits = spec.compute_draft_logits(hidden, spec_step_idx=2)
    compute.assert_called_once()
    assert compute.call_args.args[0] is spec
    assert compute.call_args.args[1].shape == (rows, HIDDEN_SIZE)
    assert compute.call_args.args[2] == 2
    torch.testing.assert_close(logits, parent_logits[: hidden.shape[0]])
    if not lmhead_enabled:
        assert compute.call_args.args[1] is hidden
        assert logits is parent_logits


@pytest.mark.parametrize(
    "speculator_cls,supported",
    [
        (AscendMTPSpeculator, True),
        (AscendEagleSpeculator, True),
        (AscendGemma4Speculator, True),
        (AscendDFlashSpeculator, True),
        (AscendDSparkSpeculator, True),
        (AscendDFlash2Speculator, False),
        (AscendLiLiCorrSpeculator, False),
    ],
)
@pytest.mark.parametrize("draft_method", ["greedy", "probabilistic"])
def test_draft_sampling_support_scope(monkeypatch, speculator_cls, supported, draft_method):
    spec, _ = _speculator(speculator_cls)
    spec.speculative_config = SimpleNamespace(draft_sample_method=draft_method)
    monkeypatch.setattr(lmhead_tp_utils, "lmhead_tp_enable", lambda: True)
    if supported:
        spec._lmhead_tp_validate_draft_sampling()
    else:
        with pytest.raises(NotImplementedError, match=speculator_cls.__name__):
            spec._lmhead_tp_validate_draft_sampling()


@pytest.mark.parametrize("speculator_cls", [AscendStandaloneARSpeculator, AscendMultiModuleMTPSpeculator])
def test_unrelated_speculators_do_not_inherit_sampling_hook(speculator_cls):
    assert not issubclass(speculator_cls, lmhead_tp_utils.LmheadTPDraftSamplingMixin)


@pytest.fixture
def draft_mixin():
    """Load the sampling mixin in isolation."""
    spec = importlib.util.spec_from_file_location("isolated_lmhead_tp_utils", lmhead_tp_utils.__file__)
    assert spec is not None and spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.lmhead_tp_enable = lambda: True
    return module.LmheadTPDraftSamplingMixin, None


def make_drafter(mixin, *, dcp=1, method="mtp"):
    class Parent:
        sample_rows: list[int]

        def compute_draft_logits(self, hidden, spec_step_idx):
            self.sample_rows.append(hidden.shape[0])
            return torch.arange(hidden.shape[0]).reshape(-1, 1)

        def sample_draft(self, hidden, *args, **kwargs):
            return self.compute_draft_logits(hidden, spec_step_idx=0).squeeze(-1)

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
    "dp_size,pcp_enabled,shared_reqs,expected",
    [(1, False, None, 2), (2, False, 5, 5), (2, False, None, 8), (2, True, 5, 8)],
)
def test_propose_uses_request_bounds(draft_mixin, dp_size, pcp_enabled, shared_reqs, expected):
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

    # Exercise the current propose signature with the upstream sampling interface.
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
    draft.pcp_manager = object() if pcp_enabled else None
    draft._get_pcp_draft_prefill_metadata = lambda metadata: metadata
    batch = SimpleNamespace(num_reqs=2, num_tokens_after_padding=24)
    state = None if shared_reqs is None else SimpleNamespace(num_reqs=shared_reqs)
    assert draft.propose(batch, *([None] * 10), dp_sync_state=state) == "drafted"
    assert draft._lmhead_pad_size is None
