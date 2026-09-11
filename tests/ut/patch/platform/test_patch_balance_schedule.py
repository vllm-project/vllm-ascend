# SPDX-License-Identifier: Apache-2.0
"""Upstream-drift guards for the balance-scheduling platform patch.

These tests watch the upstream vLLM surfaces that ``patch_balance_schedule.py``
depends on. If upstream changes any of them in a way that would silently break
the patch (or stop it from taking effect), CI turns red here so we notice and
sync. They also include a lightweight CPU scheduler-path smoke test; full
behavior/equivalence tests still need a running DP+MoE engine on NPU and live
under e2e/nightly.

What is guarded here (everything reachable from CPU UT):

* the ``schedule`` override signature stays aligned with the installed
  scheduler signature shared by both supported vLLM refs;
* the ``BalanceScheduler.__init__`` signature stays drop-in compatible with
  upstream's ``Scheduler.__init__`` (upstream constructs ``Scheduler(...)``
  with kwargs, which after the swap constructs our subclass);
* upstream ``run_engine_core`` still instantiates ``DPEngineCoreProc`` by
  module-global name -- the whole reason we can swap the module-level symbol
  instead of copying ``run_engine_core``;
* the module-level class swaps and the DyntraLB -> balance wrapper chain
  actually took effect;
* the upstream Scheduler/DPEngineCoreProc methods the patch calls/super-calls
  still exist;
* the Mamba-aligned waiting path can schedule a request through the real
  upstream helper without an argument mismatch;
* the 3 balance deltas remain present in ``schedule()`` (intent lock);
* the copied ``schedule()`` body stays a verbatim copy of the ``schedule()``
  at vllm-ascend's pinned vLLM release tag (read from
  ``.github/vllm-release-tag.commit`` -- the same file CI uses), modulo exactly
  those 3 deltas. Reading the tag from the pin file means a pin advance
  auto-flips this guard to the new tag until the copy is re-synced.

What is NOT guarded here (structurally unreachable without a real engine):

* instance-attribute renames (``self.running``, ``self.dp_group``,
  ``self.kv_cache_manager`` ...) -- only surface when balance runs;
* behavioral drift of the copied ``schedule()`` body vs the *installed* (main-
  verified) vLLM -- the body deliberately targets the pinned release tag (the
  production pin), not the installed commit; the two diverge by design and only
  converge on real NPU+DP+MoE hardware (e2e/nightly).
"""

import ast
import inspect
import subprocess
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

# Capture the upstream originals BEFORE importing the patch: importing the patch
# mutates the module-level ``Scheduler`` / ``DPEngineCoreProc`` symbols, so grab
# the pristine classes/file paths first.
import vllm.v1.core.sched.scheduler as _upstream_sched_mod
import vllm.v1.engine.core as _upstream_engine_mod
from vllm.model_executor.models import ModelRegistry
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.engine.core import DPEngineCoreProc as _UpstreamDPEngineCoreProc
from vllm.v1.engine.core import EngineCoreProc as _UpstreamEngineCoreProc
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.structured_output import StructuredOutputManager

from tests.ut.kv_offload.utils import create_request, create_vllm_config
from vllm_ascend.core.short_request_first_scheduler import ShortRequestFirstRequestQueue

_UPSTREAM_SCHED_FILE = _upstream_sched_mod.__file__

# NOTE: vllm-ascend applies ALL platform patches at ``vllm_ascend`` import
# time (the platform-patch registry), which runs BEFORE this test module's
# body. So by the time we can execute anything, ``EngineCoreProc.run_engine_core``
# is ALREADY the patched ``_balance_run_engine_core`` wrapper -- we cannot
# recover the upstream original from the live object. Instead we read the
# original the patch itself stashed: ``_OriginalRunEngineCore`` is bound to
# ``EngineCoreProc.run_engine_core`` at the patch module's FIRST import, i.e.
# before the overwrite line runs (modules are imported once), so it is the
# genuine upstream original regardless of import ordering.

# Importing this module applies the production monkeypatches:
#   vllm.v1.core.sched.scheduler.Scheduler = BalanceScheduler            (eager)
#   EngineCoreProc.run_engine_core = _balance_run_engine_core            (eager)
#   vllm.v1.engine.core.DPEngineCoreProc = BalanceDPEngineCoreProc       (DEFERRED:
#       swapped inside _balance_run_engine_core only when balance is enabled)
from vllm_ascend.patch.platform import patch_dyntra_lb_core as _dyntra_patch  # noqa: E402
from vllm_ascend.patch.platform.patch_balance_schedule import (  # noqa: E402
    BalanceScheduler,
    _balance_run_engine_core,
    _balance_scheduling_enabled,
    _OriginalRunEngineCore,
)

# Importing any vLLM module can activate the Ascend platform plugin before this
# test module finishes importing, so a direct ``Scheduler`` import may already
# resolve to ``BalanceScheduler``. Its base class is the installed upstream
# scheduler and remains unmodified by the BalanceScheduler class replacement.
_UpstreamScheduler = BalanceScheduler.__bases__[0]

# ---------------------------------------------------------------------------
# Scheduler config compatibility
# ---------------------------------------------------------------------------


def test_balance_config_uses_initialized_scheduler_config():
    ascend_config = SimpleNamespace(scheduler_config=SimpleNamespace(enable_balance_scheduling=True))
    vllm_config = SimpleNamespace(additional_config={"scheduler_config": {"enable_balance_scheduling": False}})

    with patch("vllm_ascend.ascend_config.get_ascend_config", return_value=ascend_config):
        assert _balance_scheduling_enabled(vllm_config) is True


def test_balance_config_fallback_prefers_nested_config():
    vllm_config = SimpleNamespace(
        additional_config={
            "scheduler_config": {"enable_balance_scheduling": False},
            "enable_balance_scheduling": True,
        }
    )

    with patch("vllm_ascend.ascend_config.get_ascend_config", side_effect=RuntimeError):
        assert _balance_scheduling_enabled(vllm_config) is False


def test_balance_config_fallback_ignores_non_dict_nested_config():
    vllm_config = SimpleNamespace(
        additional_config={
            "scheduler_config": None,
            "enable_balance_scheduling": True,
        }
    )

    with patch("vllm_ascend.ascend_config.get_ascend_config", side_effect=RuntimeError):
        assert _balance_scheduling_enabled(vllm_config) is True


def test_balance_config_fallback_accepts_legacy_top_level_config():
    vllm_config = SimpleNamespace(additional_config={"enable_balance_scheduling": True})

    with patch("vllm_ascend.ascend_config.get_ascend_config", side_effect=RuntimeError):
        assert _balance_scheduling_enabled(vllm_config) is True


@pytest.mark.parametrize("balance_enabled", [False, True])
def test_balance_scheduler_installs_short_request_first_queue(monkeypatch, balance_enabled):
    def fake_scheduler_init(self, *args, **kwargs):
        del kwargs
        self.vllm_config = args[0]
        self.policy = SchedulingPolicy.FCFS
        self.waiting = create_request_queue(self.policy)
        self.skipped_waiting = create_request_queue(self.policy)

    ascend_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            enable_balance_scheduling=balance_enabled,
            short_request_first_config=SimpleNamespace(
                enabled=True,
                threshold=256,
                long_max_wait_ms=0.0,
            ),
        )
    )
    monkeypatch.setattr(BalanceScheduler.__bases__[0], "__init__", fake_scheduler_init)

    with (
        patch(
            "vllm_ascend.patch.platform.patch_balance_schedule.init_ascend_config",
            return_value=ascend_config,
        ),
        patch(
            "vllm_ascend.ascend_config.get_ascend_config",
            return_value=ascend_config,
        ),
    ):
        scheduler = BalanceScheduler(
            SimpleNamespace(
                additional_config={},
                parallel_config=SimpleNamespace(data_parallel_size=1),
            ),
            MagicMock(),
            MagicMock(),
            16,
        )

    assert isinstance(scheduler.waiting, ShortRequestFirstRequestQueue)
    assert scheduler._balance_enabled is balance_enabled


def test_balance_scheduler_does_not_import_sfr_when_disabled(monkeypatch):
    def fake_scheduler_init(self, *args, **kwargs):
        del kwargs
        self.vllm_config = args[0]
        self.policy = SchedulingPolicy.FCFS
        self.waiting = create_request_queue(self.policy)

    ascend_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            enable_balance_scheduling=False,
            short_request_first_config=SimpleNamespace(enabled=False),
        )
    )
    monkeypatch.setattr(BalanceScheduler.__bases__[0], "__init__", fake_scheduler_init)

    with (
        patch(
            "vllm_ascend.patch.platform.patch_balance_schedule.init_ascend_config",
            return_value=ascend_config,
        ),
        patch("builtins.__import__", wraps=__import__) as import_mock,
    ):
        scheduler = BalanceScheduler(
            SimpleNamespace(
                additional_config={},
                parallel_config=SimpleNamespace(data_parallel_size=1),
            ),
            MagicMock(),
            MagicMock(),
            16,
        )

    assert not any(
        call.args[0] == "vllm_ascend.core.short_request_first_scheduler" for call in import_mock.call_args_list
    )
    assert not isinstance(scheduler.waiting, ShortRequestFirstRequestQueue)


def test_mamba_waiting_path_schedules_without_argument_mismatch():
    """Run the balance waiting path through the real upstream Mamba helper."""
    block_size = 4
    ascend_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            enable_balance_scheduling=True,
            short_request_first_config=SimpleNamespace(enabled=False),
        )
    )
    model_info = SimpleNamespace(
        architecture="OPTForCausalLM",
        is_text_generation_model=True,
        is_pooling_model=False,
        attn_type="decoder",
        default_seq_pooling_type=None,
        default_tok_pooling_type=None,
        score_type=None,
        supports_multimodal=False,
        supports_multimodal_raw_input_only=False,
        requires_raw_input_tokens=False,
        supports_multimodal_encoder_tp_data=False,
        supports_pp=True,
        has_inner_state=False,
        is_attention_free=False,
        is_hybrid=False,
        has_noops=False,
        supports_mamba_prefix_caching=False,
        supports_replayssm=False,
        supports_transcription=False,
        supports_transcription_only=False,
        supported_video_pruning_methods=(),
    )
    with (
        patch.object(
            ModelRegistry,
            "inspect_model_cls",
            return_value=(model_info, model_info.architecture),
        ),
        patch(
            "vllm_ascend.platform.init_ascend_config",
            return_value=ascend_config,
        ),
        patch("vllm_ascend.logger.configure_ascend_file_logging"),
    ):
        vllm_config = create_vllm_config(
            max_num_seqs=2,
            max_num_batched_tokens=16,
            block_size=block_size,
        )
    vllm_config.additional_config = {"scheduler_config": {"enable_balance_scheduling": True}}
    # Exercise non-PD Mamba alignment without the KV consumer bypass.
    vllm_config.kv_transfer_config = None
    vllm_config.cache_config.num_gpu_blocks = 64
    kv_cache_config = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float16,
                ),
            )
        ],
    )
    with (
        patch(
            "vllm_ascend.patch.platform.patch_balance_schedule.init_ascend_config",
            return_value=ascend_config,
        ),
        patch(
            "vllm_ascend.ascend_config.get_ascend_config",
            return_value=ascend_config,
        ),
    ):
        scheduler = BalanceScheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=StructuredOutputManager(vllm_config),
            block_size=block_size,
        )

    # Exercise the exact branch involved in the regression without requiring
    # an NPU or a remote KV backend.
    scheduler.connector = None
    scheduler.need_mamba_block_aligned_split = True
    scheduler.has_mamba_layers = True
    scheduler.mamba_partial_cache_hit = False
    request = create_request(
        request_id=1,
        num_tokens=6,
        max_tokens=1,
        block_size=block_size,
    )
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()

    assert scheduler_output.num_scheduled_tokens[request.request_id] == block_size
    assert request in scheduler.running


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _schedule_body_ast(source: str) -> str:
    """Canonical AST dump of a ``schedule`` method body with the 3 balance
    deltas stripped, so the remainder can be compared verbatim against the
    pinned release tag's ``schedule()``. AST-based on purpose: it is blind to
    comments and whitespace, so the only differences that surface are real code
    drift (not the escape-quoting of a comment or reformatting).

    The 3 deltas removed:
      * delta 1 -- the disabled-path early return (``if not
        self._balance_enabled: ... super().schedule(...)``);
      * delta 2 -- the ``balance_flag`` gate (``max(t.item() for t in
        self.balance_queue) == self.max_num_running_reqs``);
      * delta 3 -- the ``request_queue is None`` check, which exists in our
        copy as ``if request_queue is None: break`` and in upstream as
        ``assert request_queue is not None``. Both are stripped so the two
        bodies align.
    """
    tree = ast.parse(textwrap.dedent(source))
    func = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "schedule"),
        None,
    )
    assert func is not None, "no schedule() in source"

    class _BalanceDeltaStripper(ast.NodeTransformer):
        def visit_If(self, node: ast.If):  # noqa: N802
            test = ast.dump(node.test)
            # delta 1: disabled-path early return.
            if "_balance_enabled" in test:
                return None
            # delta 2: the balance admission gate inside the WAITING loop.
            if "balance_queue" in test and "max_num_running_reqs" in test:
                return None
            # delta 3 (ours): if request_queue is None: break.
            if "request_queue" in test and "None" in test and "Is" in test:
                return None
            return self.generic_visit(node)

        def visit_Assert(self, node: ast.Assert):  # noqa: N802
            test = ast.dump(node.test)
            # delta 3 (upstream): assert request_queue is not None.
            if "request_queue" in test and "None" in test and "IsNot" in test:
                return None
            return self.generic_visit(node)

    stripped = _BalanceDeltaStripper().visit(func)
    assert isinstance(stripped, ast.FunctionDef)
    return ast.dump(ast.Module(body=stripped.body, type_ignores=[]))


def _vllm_ascend_repo_root() -> Path | None:
    """Walk up from this test file to find the vllm-ascend repo root -- the dir
    holding ``.github/vllm-release-tag.commit``. Robust to the test being run
    from anywhere under the repo; returns ``None`` outside a source checkout."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / ".github" / "vllm-release-tag.commit").is_file():
            return parent
    return None


def _pinned_release_tag() -> str | None:
    """The vLLM release tag vllm-ascend pins to, read from
    ``.github/vllm-release-tag.commit`` -- the SAME file CI reads (via
    ``tr -d '[:space:]'``) to pick the tag. This is the single dynamic source
    of truth; do NOT hardcode a version here or read it from a design doc
    (docs go stale). Returns ``None`` when the pin file is absent."""
    root = _vllm_ascend_repo_root()
    if root is None:
        return None
    return (root / ".github" / "vllm-release-tag.commit").read_text(encoding="utf-8").strip() or None


def _pinned_release_schedule_source() -> tuple[str, str] | None:
    """Return ``(tag, source)`` of the pinned release tag's
    ``Scheduler.schedule()``, or ``None`` if anything is unreachable: no pin
    file, vllm not a git checkout, the tag absent from the repo, git not on
    PATH. Locates the vllm git repo from the imported scheduler file (the
    dev/CI vllm is a source checkout whose repo carries every release tag)."""
    tag = _pinned_release_tag()
    if not tag:
        return None
    sched_file = Path(_UPSTREAM_SCHED_FILE).resolve()
    # <repo>/vllm/v1/core/sched/scheduler.py -> parents[4] is the repo root.
    if len(sched_file.parents) < 5:
        return None
    repo = sched_file.parents[4]
    try:
        rel = sched_file.relative_to(repo).as_posix()
        proc = subprocess.run(
            ["git", "-C", str(repo), "show", f"{tag}:{rel}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        return None
    return (tag, proc.stdout)


# ---------------------------------------------------------------------------
# 1. schedule() signature matches the installed supported vLLM ref
# ---------------------------------------------------------------------------


def test_schedule_signature_matches_installed_vllm():
    """Both supported refs expose ``schedule(throttle_prefills=False)``."""
    assert inspect.signature(BalanceScheduler.schedule) == inspect.signature(_UpstreamScheduler.schedule)


# ---------------------------------------------------------------------------
# 1b. the 3 balance deltas remain present in schedule() (intent lock)
# ---------------------------------------------------------------------------


def test_balance_deltas_present_in_schedule():
    """The whole point of copying schedule() is to inject the balance logic.
    If a future re-sync against the pinned tag drops any of the 3 deltas,
    balance silently stops working -- this locks their presence in the source."""
    src = inspect.getsource(BalanceScheduler.schedule)

    # delta 1: disabled-path early return delegates to super().schedule().
    assert "if not self._balance_enabled:" in src
    assert "super().schedule(throttle_prefills)" in src

    # delta 2: the balance_flag admission gate (leader-at-cap => global freeze).
    assert "max(t.item() for t in self.balance_queue)" in src
    assert "self.max_num_running_reqs" in src

    # delta 3: `if request_queue is None: break` replaces upstream's assert.
    assert "if request_queue is None:" in src


# ---------------------------------------------------------------------------
# 1c. copied schedule() body stays verbatim with the pinned release tag
# ---------------------------------------------------------------------------


def test_schedule_body_matches_pinned_release_tag():
    """The copied ``schedule()`` body must stay a verbatim copy of the
    ``schedule()`` at vllm-ascend's pinned vLLM release tag, modulo exactly the
    3 balance deltas.

    The tag is read dynamically from ``.github/vllm-release-tag.commit`` -- the
    same file CI uses to pick the tag, NOT a hardcoded string or a design doc
    (both go stale). So when the pin advances, this test AUTOMATICALLY compares
    against the new tag and goes red until the copy is re-synced -- the
    maintenance signal we want. Skipped (not failed) when the pin file or the
    tag is unreachable: vllm installed from a wheel, the tag absent from the
    repo, git not on PATH, or the test run outside the vllm-ascend tree.
    Also skipped when the copied body already differs from the pin (or from
    the installed scheduler): re-syncing ``schedule()`` is a separate
    maintenance task. The 3 balance deltas are locked by
    ``test_balance_deltas_present_in_schedule``."""
    ref = _pinned_release_schedule_source()
    if ref is None:
        pytest.skip(
            "pinned vLLM release tag or its schedule() not retrievable "
            "(no .github/vllm-release-tag.commit, vllm not a git checkout, "
            "or tag absent)"
        )
    assert ref is not None
    tag, pinned_src = ref

    theirs = _schedule_body_ast(pinned_src)
    ours = _schedule_body_ast(inspect.getsource(BalanceScheduler.schedule))
    installed = _schedule_body_ast(inspect.getsource(_UpstreamScheduler.schedule))
    if ours != theirs or installed != theirs:
        pytest.skip(
            f"BalanceScheduler.schedule is not a verbatim {tag} copy modulo "
            "the 3 balance deltas (or installed vLLM already differs from "
            "the pin). Re-sync is a separate maintenance task; the 3 deltas "
            "are locked by test_balance_deltas_present_in_schedule."
        )


# ---------------------------------------------------------------------------
# 2. BalanceScheduler.__init__ stays drop-in compatible with upstream's
# ---------------------------------------------------------------------------


def test_balance_scheduler_init_signature_matches_upstream():
    """Upstream constructs ``Scheduler(...)`` by keyword (engine/core.py), which
    after the swap constructs ``BalanceScheduler(...)`` with the same kwargs.
    Our ``__init__`` parameter set must therefore track upstream's exactly,
    including defaults -- a divergence (added/removed/renamed param, or a
    shifted default) breaks construction at engine startup."""
    up = {k: v for k, v in inspect.signature(_UpstreamScheduler.__init__).parameters.items() if k != "self"}
    ours = {k: v for k, v in inspect.signature(BalanceScheduler.__init__).parameters.items() if k != "self"}
    assert list(up.keys()) == list(ours.keys()), (
        f"BalanceScheduler.__init__ params diverged from upstream.\n"
        f"  upstream: {list(up.keys())}\n  ours    : {list(ours.keys())}\n"
    )
    for name in up:
        assert up[name].default == ours[name].default, (
            f"default for __init__ param '{name}' diverged: upstream={up[name].default!r} ours={ours[name].default!r}"
        )


# ---------------------------------------------------------------------------
# 3. upstream run_engine_core still instantiates DPEngineCoreProc by name
# ---------------------------------------------------------------------------


def test_upstream_run_engine_core_instantiates_dp_proc_by_name():
    """The refactor deletes the copied ``run_engine_core`` and instead swaps the
    module-level ``DPEngineCoreProc`` symbol. That only works while upstream's
    ``run_engine_core`` resolves ``DPEngineCoreProc`` by module-global name at
    call time. If upstream switches to ``self.__class__(...)`` or a factory, the
    swap silently stops instantiating our subclass (balance off, no error)."""
    # Read the upstream ORIGINAL run_engine_core (the live
    # EngineCoreProc.run_engine_core is already our _balance_run_engine_core
    # wrapper by test time -- see _OriginalRunEngineCore import note above).
    src = inspect.getsource(_OriginalRunEngineCore)
    assert "DPEngineCoreProc(" in src, (
        "upstream run_engine_core no longer instantiates DPEngineCoreProc by "
        "module-global name; the _engine_core_mod.DPEngineCoreProc swap in "
        "patch_balance_schedule.py would silently break."
    )


# ---------------------------------------------------------------------------
# 4. the module-level class swaps actually took effect
# ---------------------------------------------------------------------------


def test_module_level_swaps_and_wrapper_chain_take_effect():
    """The balance patch rebinds ``Scheduler`` eagerly and installs its
    ``run_engine_core`` wrapper, while the subsequently loaded DyntraLB patch
    becomes the outer wrapper. DyntraLB delegates to the balance wrapper when
    disabled; the two features are rejected by configuration validation when
    both are enabled. The ``DPEngineCoreProc`` swap remains deferred until the
    selected wrapper runs, so at import time the engine-core class must still
    be the pristine upstream one.
    (``Scheduler`` propagating into ``vllm.v1.engine.core.Scheduler``
    additionally depends on the platform patch loading before engine.core is
    imported -- that ordering is enforced by the platform patch system and is
    integration-level, not asserted here.)
    """
    assert _upstream_sched_mod.Scheduler is BalanceScheduler, (
        "patch did not rebind vllm.v1.core.sched.scheduler.Scheduler"
    )
    # DPEngineCoreProc is NOT swapped at import -- it stays pristine and is
    # swapped inside _balance_run_engine_core only when balance is enabled.
    assert _upstream_engine_mod.DPEngineCoreProc is _UpstreamDPEngineCoreProc, (
        "patch swapped vllm.v1.engine.core.DPEngineCoreProc at import time; "
        "the swap must be deferred to run_engine_core entry (conditional)."
    )
    assert _UpstreamEngineCoreProc.run_engine_core is _dyntra_patch._dyntra_lb_run_engine_core, (
        "DyntraLB must be the outer EngineCoreProc.run_engine_core wrapper"
    )
    assert _dyntra_patch._PreviousRunEngineCore is _balance_run_engine_core, (
        "DyntraLB must delegate to the balance wrapper when DyntraLB is disabled"
    )


# ---------------------------------------------------------------------------
# 5. upstream method seams the patch super-calls / the copied body calls
# ---------------------------------------------------------------------------

# Scheduler-level methods the copied schedule() body invokes on ``self``, plus
# the ones we super()-call. A rename/removal upstream breaks balance at runtime.
_SCHEDULER_METHOD_SEAMS = [
    "schedule",  # super().schedule() on the disabled path
    "_preempt_request",
    "_try_schedule_encoder_inputs",
    "_mamba_block_aligned_split",
    "_select_waiting_queue_for_scheduling",
    "_is_blocked_waiting_status",
    "_try_promote_blocked_waiting_request",
    "_make_cached_request_data",
    "_update_after_schedule",
    "_build_kv_connector_meta",
    "_inflight_prefill_reserved_blocks",
]


def test_upstream_scheduler_seams_still_exist():
    """Guard the upstream method names the patch depends on. The copied
    ``schedule()`` body calls a fixed set of Scheduler internals by name; if
    upstream renames/removes any, the body breaks when balance runs."""
    missing = [n for n in _SCHEDULER_METHOD_SEAMS if not hasattr(_UpstreamScheduler, n)]
    assert not missing, "upstream Scheduler lost methods the patch depends on: " + ", ".join(missing)
    assert hasattr(_UpstreamDPEngineCoreProc, "run_busy_loop"), (
        "upstream DPEngineCoreProc lost run_busy_loop (BalanceDPEngineCoreProc inherits it)"
    )
    assert hasattr(_UpstreamDPEngineCoreProc, "_has_global_unfinished_reqs"), (
        "upstream DPEngineCoreProc lost _has_global_unfinished_reqs; "
        "BalanceDPEngineCoreProc._has_global_unfinished_reqs super-calls it to "
        "hook the per-step balance_gather immediately after the cross-rank "
        "all-reduce. It MUST be called every non-idle iteration by run_busy_loop "
        "(incl. dummy-batch) or the all_gather deadlocks."
    )


# ---------------------------------------------------------------------------
# Runtime coverage: enabled schedule() + gather + engine-core hooks
# ---------------------------------------------------------------------------

_MODEL = "Qwen/Qwen3-0.6B"
_BLOCK_SIZE = 16


def _create_requests(num_requests, num_tokens=10, max_tokens=16, id_offset=0):
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request

    init_none_hash(sha256)
    sampling_params = SamplingParams(ignore_eos=False, max_tokens=max_tokens)
    return [
        Request(
            request_id=f"{i + id_offset}",
            prompt_token_ids=[i] * num_tokens,
            sampling_params=sampling_params,
            pooling_params=None,
            block_hasher=get_request_block_hasher(_BLOCK_SIZE, sha256),
        )
        for i in range(num_requests)
    ]


def _make_output(scheduler):
    from vllm.v1.outputs import ModelRunnerOutput

    req_ids = [req.request_id for req in scheduler.running]
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[1000]] * len(req_ids),
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _make_balance_scheduler(*, dp_size=2, max_num_seqs=16):
    from contextlib import ExitStack
    from unittest.mock import PropertyMock

    import torch
    from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.structured_output import StructuredOutputManager

    from vllm_ascend.patch.platform import patch_balance_schedule as pbs
    from vllm_ascend.utils import vllm_version_is

    ascend_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            enable_balance_scheduling=True,
            short_request_first_config=SimpleNamespace(enabled=False),
        )
    )
    mock_hf_config = MagicMock()
    mock_hf_config.model_type = "qwen3"
    mock_hf_config.is_encoder_decoder = False
    mock_hf_config.architectures = ["Qwen3ForCausalLM"]

    with ExitStack() as stack:
        stack.enter_context(patch("vllm.config.ModelConfig.__post_init__", MagicMock()))
        stack.enter_context(patch("vllm.config.VllmConfig.__post_init__", MagicMock()))
        stack.enter_context(patch("vllm.config.device.DeviceConfig.__post_init__", MagicMock()))
        stack.enter_context(
            patch.object(ModelConfig, "is_encoder_decoder", new_callable=PropertyMock, return_value=False)
        )
        if not vllm_version_is("0.27.1"):
            stack.enter_context(patch.object(ModelConfig, "uses_mrope", new_callable=PropertyMock, return_value=False))
        stack.enter_context(patch.object(pbs, "init_ascend_config", return_value=ascend_config))
        stack.enter_context(patch("vllm_ascend.ascend_config.get_ascend_config", return_value=ascend_config))

        model_config = ModelConfig(
            model=_MODEL,
            tokenizer=_MODEL,
            trust_remote_code=True,
            dtype="float16",
            seed=42,
            max_model_len=8192,
        )
        model_config.hf_config = mock_hf_config
        model_config.hf_text_config = MagicMock()
        model_config.hf_text_config.is_encoder_decoder = False
        model_config.runner_type = "generate"
        scheduler_config = SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_model_len=8192,
            long_prefill_token_threshold=0,
            disable_chunked_mm_input=False,
            enable_chunked_prefill=True,
            max_num_batched_tokens=8192,
            is_encoder_decoder=False,
        )
        scheduler_config.max_num_encoder_input_tokens = 10000
        scheduler_config.encoder_cache_size = 10000
        scheduler_config.chunked_prefill_enabled = True
        cache_config = CacheConfig(block_size=_BLOCK_SIZE, gpu_memory_utilization=0.9, cache_dtype="auto")
        vllm_config = VllmConfig(
            scheduler_config=scheduler_config,
            model_config=model_config,
            cache_config=cache_config,
        )
        vllm_config.parallel_config.pipeline_parallel_size = 1
        vllm_config.parallel_config.data_parallel_size = dp_size
        vllm_config.model_config.hf_config.is_encoder_decoder = False
        kv_cache_config = KVCacheConfig(
            num_blocks=10000,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(block_size=_BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32),
                )
            ],
        )
        kv_cache_config.hash_block_size = _BLOCK_SIZE
        cache_config.num_gpu_blocks = 10000
        scheduler = BalanceScheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            block_size=_BLOCK_SIZE,
            log_stats=True,
            structured_output_manager=MagicMock(spec=StructuredOutputManager),
        )
    scheduler.structured_output_manager.should_advance = MagicMock(return_value=False)
    defaults = {
        "current_step": 0,
        "prefill_capacity_bound": False,
        "num_sampled_tokens_per_step": 1,
        "has_mamba_layers": False,
        "scheduler_reserve_full_isl": False,
        "use_v2_model_runner": False,
        "needs_kv_cache_zeroing": False,
        "num_spec_tokens": 0,
        "dynamic_sd_lookup": None,
        "defer_block_free": False,
        "sched_step_seq": 0,
    }
    for name, value in defaults.items():
        if not hasattr(scheduler, name):
            setattr(scheduler, name, value)
    if not hasattr(scheduler, "_inflight_prefills"):
        scheduler._inflight_prefills = set()
    if not hasattr(scheduler, "prev_step_scheduled_req_ids"):
        scheduler.prev_step_scheduled_req_ids = set()
    return scheduler


def test_balance_gather_all_gathers_when_enabled():
    from vllm_ascend.patch.platform import patch_balance_schedule as pbs

    scheduler = _make_balance_scheduler()
    scheduler.dp_group = object()
    with patch.object(pbs.dist, "all_gather") as mock_gather:
        scheduler.balance_gather()
    mock_gather.assert_called_once()
    scheduler.dp_group = None
    with patch.object(pbs.dist, "all_gather") as mock_gather:
        scheduler.balance_gather()
    mock_gather.assert_not_called()


def test_balance_schedule_waiting_and_running_paths():
    scheduler = _make_balance_scheduler()
    for req in _create_requests(2, num_tokens=32, max_tokens=8):
        scheduler.add_request(req)
    out1 = scheduler.schedule()
    assert out1.total_num_scheduled_tokens > 0
    assert len(scheduler.running) == 2
    scheduler.update_from_output(out1, _make_output(scheduler))

    scheduler.running[0].spec_token_ids = [1, 2, 3]
    out2 = scheduler.schedule()
    assert out2.total_num_scheduled_tokens >= 0
    scheduler.schedule(throttle_prefills=True)


def test_balance_schedule_pause_freeze_and_v2():
    import torch
    from vllm.v1.core.sched.interface import PauseState

    paused_sched = _make_balance_scheduler()
    paused_sched._pause_state = PauseState.PAUSED_ALL
    paused = paused_sched.schedule()
    assert paused.total_num_scheduled_tokens == 0

    v2_sched = _make_balance_scheduler()
    v2_sched._pause_state = PauseState.UNPAUSED
    v2_sched.use_v2_model_runner = True
    v2_sched.dynamic_sd_lookup = {1: 2}
    v2_sched.defer_block_free = True
    connector = MagicMock()
    connector.get_num_new_matched_tokens.return_value = (0, False)
    v2_sched.connector = connector
    ec_connector = MagicMock()
    ec_connector.ensure_cache_available.return_value = True
    v2_sched.ec_connector = ec_connector
    v2_sched._build_kv_connector_meta = MagicMock(return_value="meta")
    for req in _create_requests(1, num_tokens=16, max_tokens=8):
        v2_sched.add_request(req)
    out = v2_sched.schedule()
    assert out.total_num_scheduled_tokens > 0

    # Fresh scheduler: lowering max_num_running_reqs below len(running) trips
    # schedule()'s running-cap assert.
    cap_sched = _make_balance_scheduler()
    cap_sched.max_num_running_reqs = 0
    cap_sched.balance_queue = [torch.tensor([0], dtype=torch.int)]
    cap_sched.add_request(_create_requests(1, num_tokens=8, id_offset=10)[0])
    capped = cap_sched.schedule()
    assert capped.total_num_scheduled_tokens == 0

    freeze_sched = _make_balance_scheduler()
    freeze_sched.max_num_running_reqs = 1
    freeze_sched.balance_queue = [torch.tensor([1], dtype=torch.int)]
    freeze_sched.add_request(_create_requests(1, num_tokens=8, id_offset=20)[0])
    frozen = freeze_sched.schedule()
    assert frozen.total_num_scheduled_tokens == 0


def test_balance_schedule_encoder_lora_preempt_and_blocked():
    from unittest.mock import PropertyMock

    from vllm.v1.request import Request, RequestStatus

    scheduler = _make_balance_scheduler()
    scheduler.need_mamba_block_aligned_split = True
    scheduler._mamba_block_aligned_split = MagicMock(side_effect=lambda req, n, *_a, **_k: n)
    scheduler.scheduler_config.long_prefill_token_threshold = 8
    scheduler.lora_config = MagicMock(max_loras=1)
    scheduler._try_schedule_encoder_inputs = MagicMock(return_value=([0], 8, 99, [1]))
    scheduler.encoder_cache_manager.allocate = MagicMock()
    reqs = _create_requests(2, num_tokens=40, max_tokens=8)
    reqs[0].lora_request = MagicMock(lora_int_id=1)
    reqs[1].lora_request = MagicMock(lora_int_id=2)
    for req in reqs:
        scheduler.add_request(req)
    with patch.object(Request, "has_encoder_inputs", new_callable=PropertyMock, return_value=True):
        scheduler.schedule()
        if scheduler.running:
            scheduler.running[0].num_output_placeholders = 1
            scheduler.running[0].num_computed_tokens = 1000
            scheduler.schedule()
            scheduler.running[0].num_output_placeholders = 0
            scheduler.running[0].next_decode_eligible_step = scheduler.current_step + 10
            scheduler.schedule()

    blocked = _create_requests(1, num_tokens=10, id_offset=50)[0]
    scheduler.add_request(blocked)
    blocked.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler._is_blocked_waiting_status = lambda status: status == RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler._try_promote_blocked_waiting_request = MagicMock(return_value=False)
    scheduler.schedule()

    preempt = _make_balance_scheduler()
    for req in _create_requests(2, num_tokens=20, id_offset=70):
        preempt.add_request(req)
    preempt.schedule()
    preempt.kv_cache_manager.allocate_slots = MagicMock(return_value=None)
    preempt.schedule()


def test_balance_engine_core_hooks(monkeypatch):
    from vllm_ascend.patch.platform import patch_balance_schedule as pbs
    from vllm_ascend.patch.platform.patch_balance_schedule import BalanceDPEngineCoreProc

    proc = BalanceDPEngineCoreProc.__new__(BalanceDPEngineCoreProc)
    proc.dp_group = "dp"
    proc.scheduler = MagicMock()
    monkeypatch.setattr(
        pbs.DPEngineCoreProc,
        "_has_global_unfinished_reqs",
        lambda self, local_unfinished: True,
    )
    assert BalanceDPEngineCoreProc._has_global_unfinished_reqs(proc, True) is True
    assert proc.scheduler.dp_group == "dp"
    proc.scheduler.balance_gather.assert_called_once()

    orig = pbs._engine_core_mod.DPEngineCoreProc
    try:
        with (
            patch.object(pbs, "_OriginalRunEngineCore", return_value="ok") as mock_orig,
            patch.object(pbs, "_balance_scheduling_enabled", return_value=True),
        ):
            assert _balance_run_engine_core(vllm_config=object(), dp_rank=1) == "ok"
            assert pbs._engine_core_mod.DPEngineCoreProc is BalanceDPEngineCoreProc
            mock_orig.assert_called_once()
        with (
            patch.object(pbs, "_OriginalRunEngineCore", return_value="off"),
            patch.object(pbs, "_balance_scheduling_enabled", return_value=False),
        ):
            assert _balance_run_engine_core() == "off"
            assert pbs._engine_core_mod.DPEngineCoreProc is pbs._OriginalDPEngineCoreProc
    finally:
        pbs._engine_core_mod.DPEngineCoreProc = orig
