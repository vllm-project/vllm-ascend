# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
from types import SimpleNamespace

PROFILE_PATH = Path(__file__).resolve().parents[1] / "draft_profile.py"
DFLASH_PATH = (
    Path(__file__).resolve().parents[2]
    / "vllm_ascend"
    / "spec_decode"
    / "dflash_proposer.py"
)


class _Mode:
    FULL = "full"
    PIECEWISE = "piecewise"
    NONE = "none"


class _Event:
    def record(self) -> None:
        pass

    def elapsed_time(self, other) -> float:
        return 8.0


class _Drafter:
    method = "dflash"
    use_cuda_graph = True
    num_speculative_tokens = 15

    def __init__(self) -> None:
        self.calls = []

    def dummy_run(self, **kwargs) -> None:
        self.calls.append(kwargs)


class _Dispatcher:
    def __init__(self) -> None:
        self.calls = []

    def dispatch(self, **kwargs):
        self.calls.append(kwargs)
        return _Mode.PIECEWISE, SimpleNamespace(num_tokens=64)


class _Runner:
    def __init__(self) -> None:
        self.drafter = _Drafter()
        self.cudagraph_dispatcher = _Dispatcher()
        self.input_batch = SimpleNamespace(lora_id_to_lora_request={})

    @staticmethod
    def _sync_metadata_across_dp(num_tokens, is_draft_model):
        assert is_draft_model
        return num_tokens, None, None


def _load_profile_function():
    tree = ast.parse(PROFILE_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_adaptive_profile_draft_run"
    )
    function.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "CUDAGraphMode": _Mode,
        "ENV_PROFILE_FORCE_EAGER": "VLLM_DCUT_PROFILE_FORCE_EAGER",
        "os": SimpleNamespace(
            environ={"VLLM_DCUT_PROFILE_FORCE_EAGER": "0"},
        ),
        "_npu_event": lambda enable_timing: _Event(),
        "logger": SimpleNamespace(warning=lambda *args, **kwargs: None),
        "torch": SimpleNamespace(
            npu=SimpleNamespace(synchronize=lambda: None),
        ),
    }
    exec(compile(module, str(PROFILE_PATH), "exec"), namespace)
    return namespace["_adaptive_profile_draft_run"]


def test_draft_profile_keeps_full_query_shape_and_separate_context_q() -> None:
    profile = _load_profile_function()
    runner = _Runner()

    mode, avg_ms, padded_tokens = profile(
        runner,
        batch_size=2,
        context_tokens=6,
        n_warmup=1,
        n_measure=2,
    )

    assert mode == "PCG"
    assert avg_ms == 4.0
    assert padded_tokens == 64
    assert runner.cudagraph_dispatcher.calls[0]["num_tokens"] == 32
    assert len(runner.drafter.calls) == 3
    for call in runner.drafter.calls:
        assert call["num_tokens"] == 64
        assert call["num_reqs"] == 2
        assert call["context_num_tokens"] == 6


def test_dflash_dummy_uses_separate_context_length() -> None:
    source = DFLASH_PATH.read_text(encoding="utf-8")
    assert "context_num_tokens: int | None = None" in source
    assert "self._dflash_num_context = num_context_tokens" in source
    assert "self._context_positions_buffer[:num_context_tokens]" in source
