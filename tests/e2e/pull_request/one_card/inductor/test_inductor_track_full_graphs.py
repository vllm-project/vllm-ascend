"""Stage3 e2e: inductor track with the full-graph capture family (U6, three cg values)
plus the DEFAULT tier added by the debt-2 refactor (no explicit cg — the -O2 preset
must fill FULL_AND_PIECEWISE; refactor1 M1).

Mirrors the T0''-1 probe (stage design/stage3/_notes/t0_probe/RESULTS.md): in-process
engine + NPUGraph init/replay spies grouped by forward_context.cudagraph_runtime_mode.
Acceptance formulas are symbolic (P/S/N taken from runtime counters, R3-3):

  FULL_AND_PIECEWISE : init = P*S_pw + 1*S_f   ; replay = P*N_m + 1*N_f
  FULL               : init = 1*S_f            ; replay = 1*N_all
  FULL_DECODE_ONLY   : init = 1*S_f            ; replay = 1*N_f   (mixed steps ungraphed)

Invariants: every FULL-size group captures exactly one graph (inner increment during
the FULL capture window is zero — no PIECEWISE-group counts for pure FULL/FDO), and
num_cudagraph_captured closes exactly against the grouped spy counts.
"""

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from unittest.mock import patch  # noqa: E402

import pytest  # noqa: E402

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free  # noqa: E402

PROMPTS = [
    "The capital of France is",
    "给我一句话介绍长城。",
]
MAX_TOKENS = 16
EXPECTED_TEXT_PREFIX_CHARS = 77  # greedy 16-token prefix, probe-calibrated


def _mode_of():
    try:
        from vllm.forward_context import get_forward_context

        fc = get_forward_context()
        mode = str(getattr(fc, "cudagraph_runtime_mode", None))
        nt = getattr(getattr(fc, "batch_descriptor", None), "num_tokens", None)
        return f"{mode}@{nt}"
    except Exception:
        return "no-ctx"


def _run_track(cg_mode_name: str):
    import torch
    from vllm import SamplingParams
    from vllm.compilation.counter import compilation_counter
    from vllm.config.compilation import CUDAGraphMode, CompilationConfig

    # compilation_counter is a process-global cumulative singleton: with several
    # engines in one pytest process the assertions must use deltas, not absolutes
    # (probe processes saw the absolutes directly).
    before_pieces = compilation_counter.num_piecewise_capturable_graphs_seen
    before_captured = compilation_counter.num_cudagraph_captured

    runner_kwargs = {}
    if cg_mode_name == "DEFAULT":
        # Debt 2 (ledger 13): default journey — no explicit cudagraph_mode; the
        # -O2 preset must fill FULL_AND_PIECEWISE (upstream semantics). The
        # front door still selects the track.
        runner_kwargs["compilation_config"] = CompilationConfig(backend="inductor")
    else:
        cg_mode = {
            "FULL_AND_PIECEWISE": CUDAGraphMode.FULL_AND_PIECEWISE,
            "FULL": CUDAGraphMode.FULL,
            "FULL_DECODE_ONLY": CUDAGraphMode.FULL_DECODE_ONLY,
        }[cg_mode_name]
        runner_kwargs["compilation_config"] = CompilationConfig(backend="inductor", cudagraph_mode=cg_mode)

    counts = {"init": 0, "replay": 0}
    by_mode = {"init": {}, "replay": {}}

    def bump(kind):
        key = _mode_of()
        counts[kind] += 1
        by_mode[kind][key] = by_mode[kind].get(key, 0) + 1

    orig_init = torch.npu.NPUGraph.__init__

    def init_spy(self, *args, **kwargs):
        bump("init")
        return orig_init(self, *args, **kwargs)

    orig_replay = torch.npu.NPUGraph.replay

    def replay_spy(self, *args, **kwargs):
        bump("replay")
        return orig_replay(self, *args, **kwargs)

    final_cg = None
    with patch.object(torch.npu.NPUGraph, "__init__", init_spy), patch.object(
        torch.npu.NPUGraph, "replay", replay_spy
    ):
        # VllmRunner: repo-conventional context manager whose __exit__ releases
        # the engine deterministically — raw LLM() objects leak HBM across
        # parametrized engines in one process (2026-09-11 runbook).
        with VllmRunner(
            model_name="Qwen/Qwen3-0.6B",
            dtype="bfloat16",
            max_model_len=4096,
            # accommodate slow driver reclaim between engines on a shared device
            # (residual ~25 GiB is common right after an engine exits): 0.55 of
            # 60.96 GiB = 33.5 GiB fits the 0.6B model + KV + graphs comfortably
            gpu_memory_utilization=0.55,
            max_num_seqs=16,
            **runner_kwargs,
        ) as runner:
            try:
                final_cg = runner.model.llm_engine.vllm_config.compilation_config.cudagraph_mode
            except Exception:
                final_cg = None
            outs = runner.model.generate(
                PROMPTS, SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)
            )

    return {
        "counts": counts,
        "by_mode": by_mode,
        "pieces_delta": compilation_counter.num_piecewise_capturable_graphs_seen - before_pieces,
        "captured_delta": compilation_counter.num_cudagraph_captured - before_captured,
        "final_cg": final_cg,
    }, outs


def _full_group(by_mode, kind) -> dict:
    # str(CUDAGraphMode.FULL) is the bare name ("FULL@N"); runtime modes are
    # only NONE/PIECEWISE/FULL so the bare prefixes are unambiguous.
    return {k: v for k, v in by_mode[kind].items() if k.startswith("FULL@")}


def _piecewise_group(by_mode, kind) -> dict:
    return {k: v for k, v in by_mode[kind].items() if k.startswith("PIECEWISE@")}


@pytest.mark.parametrize("cg_mode_name", ["DEFAULT", "FULL_AND_PIECEWISE", "FULL", "FULL_DECODE_ONLY"])
@wait_until_npu_memory_free(max_wait_seconds=240)  # in-process engines release HBM slowly
def test_inductor_track_full_graph_family(cg_mode_name):
    result, outs = _run_track(cg_mode_name)
    counts = result["counts"]
    by_mode = result["by_mode"]
    # DEFAULT tier exercises the debt-2 default journey: effective mode must be
    # the -O2 preset value (FULL_AND_PIECEWISE), asserted on the final config.
    effective = "FULL_AND_PIECEWISE" if cg_mode_name == "DEFAULT" else cg_mode_name
    if cg_mode_name == "DEFAULT":
        assert result["final_cg"] is not None, "could not read final cudagraph_mode"
        assert str(result["final_cg"]).endswith("FULL_AND_PIECEWISE"), (
            f"preset-sourced default expected FULL_AND_PIECEWISE, got {result['final_cg']}"
        )

    texts = [o.outputs[0].text for o in outs]
    for text in texts:
        assert text.strip(), "empty generation"

    full_init = _full_group(by_mode, "init")
    full_replay = _full_group(by_mode, "replay")

    # invariant: each FULL size captures exactly one graph
    for key, n in full_init.items():
        assert n == 1, f"FULL size {key} captured {n} graphs, expected exactly 1"

    # formula closure: capture counter delta == spy init total (replay has no core counter)
    assert result["captured_delta"] == counts["init"], (
        f"counter delta {result['captured_delta']} != spy init {counts['init']}"
    )

    if effective == "FULL_AND_PIECEWISE":
        # P*S_pw (PIECEWISE leg: every piece per size) + 1*S_f (FULL leg)
        assert _piecewise_group(by_mode, "init"), "PIECEWISE leg inner captures missing"
        piecewise_sizes = {k.split("@")[1] for k in _piecewise_group(by_mode, "init")}
        full_sizes = {k.split("@")[1] for k in full_init}
        assert piecewise_sizes == full_sizes, "both legs should cover the same size set"
        assert result["pieces_delta"] > 1, "expected piecewise pieces"
    else:
        # upstream-aligned single-graph shape: no inner (PIECEWISE-group) captures at all
        assert not _piecewise_group(by_mode, "init"), (
            "pure FULL/FDO must not build inner piecewise graphs (splitting_ops=[])"
        )
        assert result["pieces_delta"] == 1, (
            f"single compiled graph expected (splitting_ops=[]), pieces delta {result['pieces_delta']}"
        )
        total_replays = sum(full_replay.values())
        if effective == "FULL":
            # every decode step dispatches a FULL graph, mixed steps included
            assert total_replays >= MAX_TOKENS, (
                f"FULL: expected >= {MAX_TOKENS} full-graph replays, got {total_replays}"
            )
        else:  # FULL_DECODE_ONLY: only uniform decode steps; mixed step runs ungraphed
            assert total_replays > 0
    assert full_replay, "no FULL graph replay observed"
