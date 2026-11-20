# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend"


def _load_class(path, name, **scope):
    """Exercise metadata logic without importing NPU kernels."""
    node = next(
        node
        for node in ast.parse((SOURCE / path).read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == name
    )
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(SOURCE / path), "exec"), scope)
    return scope[name]


def test_diffusion_phase_mask_is_updated_in_place_for_graph_replay():
    class GraphMode(Enum):
        FULL = "FULL"

    def build_attn_metadata(**kwargs):
        return {"layer": SimpleNamespace(seq_lens=kwargs["seq_lens_np"])}

    state_cls = _load_class(
        "worker/v2/model_states/diffusion_gemma.py",
        "AscendDiffusionGemmaModelState",
        DiffusionGemmaModelState=object,
        CUDAGraphMode=GraphMode,
        AscendAttentionState=SimpleNamespace(ChunkedPrefill=0),
        build_attn_metadata=build_attn_metadata,
        torch=torch,
    )
    state = state_cls.__new__(state_cls)
    state._token_positions = torch.arange(8, dtype=torch.int32)
    state._token_causal = torch.empty(8, dtype=torch.bool)
    state.max_model_len = 32
    state.diffusion_states = SimpleNamespace(is_encoder_phase=torch.tensor([False, True]))
    batch = SimpleNamespace(
        num_reqs=2,
        num_reqs_after_padding=2,
        num_tokens=5,
        num_tokens_after_padding=5,
        query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        query_start_loc_np=np.array([0, 2, 5], dtype=np.int32),
        idx_mapping=torch.tensor([1, 0]),
        seq_lens=torch.tensor([12, 23]),
        num_scheduled_tokens=np.array([2, 3]),
    )

    first = state.prepare_attn(batch, GraphMode.FULL, [], None, [], None)["layer"]
    assert first.diffusion_causal.tolist() == [True, True, False, False, False]
    assert first.seq_lens.tolist() == [12, 23]

    state.diffusion_states.is_encoder_phase.logical_not_()
    second = state.prepare_attn(batch, GraphMode.FULL, [], None, [], None)["layer"]
    assert second.diffusion_causal.tolist() == [False, False, True, True, True]
    assert first.diffusion_causal.data_ptr() == second.diffusion_causal.data_ptr()
    assert state.attn_metadata["layer"] is second


def test_diffusion_fia_tasks_have_distinct_replay_providers():
    provider = _load_class("attention/attention_v1.py", "FIAParamProvider", dataclass=dataclass, Any=Any)
    causal = provider("layer", None, diffusion_causal=True)
    bidirectional = provider("layer", None, diffusion_causal=False)
    assert len({causal, bidirectional}) == 2

    context = {"layer": SimpleNamespace(actual_seq_lengths_q=[2, 5], seq_lens_list=[12, 23], block_tables=None)}
    assert causal.resolve(context) == bidirectional.resolve(context)
