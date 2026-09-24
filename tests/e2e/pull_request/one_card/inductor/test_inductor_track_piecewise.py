"""Stage2 e2e: inductor compile-backend track with piecewise graph capture.

Mirrors docs/vllm/env/scripts/smoke_qwen3_inductor_pw_spy.py (workspace repo,
stage design/stage2/04 §T2'): in-process engine + NPUGraph capture/replay
spy counters + vllm compilation counter.
"""

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from unittest.mock import patch  # noqa: E402


def test_inductor_track_piecewise_capture_replay():
    import torch
    from vllm import LLM, SamplingParams
    from vllm.compilation.counter import compilation_counter
    from vllm.config.compilation import CUDAGraphMode, CompilationConfig

    counts = {"npugraph_init": 0, "npugraph_replay": 0}

    orig_init = torch.npu.NPUGraph.__init__

    def init_spy(self, *args, **kwargs):
        counts["npugraph_init"] += 1
        return orig_init(self, *args, **kwargs)

    orig_replay = torch.npu.NPUGraph.replay

    def replay_spy(self, *args, **kwargs):
        counts["npugraph_replay"] += 1
        return orig_replay(self, *args, **kwargs)

    with patch.object(torch.npu.NPUGraph, "__init__", init_spy), patch.object(
        torch.npu.NPUGraph, "replay", replay_spy
    ):
        # Explicit PIECEWISE: since the debt-2 refactor the track default
        # follows the -O presets (O2 -> FULL_AND_PIECEWISE), so this test pins
        # the mode it exists to guard.
        llm = LLM(
            model="Qwen/Qwen3-0.6B",
            dtype="bfloat16",
            max_model_len=4096,
            max_num_seqs=16,
            # front door (config refactor M1b): compilation_config.backend
            # selects the track
            compilation_config=CompilationConfig(backend="inductor", cudagraph_mode=CUDAGraphMode.PIECEWISE),
        )
        outs = llm.generate(
            ["The capital of France is"],
            SamplingParams(max_tokens=16, temperature=0.0),
        )

    assert counts["npugraph_init"] > 0, "no ACLGraph capture happened"
    assert counts["npugraph_replay"] > 0, "no ACLGraph replay happened"
    assert compilation_counter.num_cudagraph_captured > 0
    assert outs[0].outputs[0].text.strip()
