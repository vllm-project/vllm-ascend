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


def _load_class(path, name, method=None, **scope):
    """Exercise metadata logic without importing NPU kernels."""
    class_node = next(
        node
        for node in ast.parse((SOURCE / path).read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == name
    )
    node: ast.stmt = class_node
    if method is not None:
        node = next(item for item in class_node.body if isinstance(item, ast.FunctionDef) and item.name == method)
        name = method
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


def test_diffusion_full_replay_pads_fia_without_changing_upstream_offsets():
    class GraphMode(Enum):
        FULL = "FULL"

    def build_attn_metadata(**kwargs):
        return {
            "layer": SimpleNamespace(
                actual_seq_lengths_q=kwargs["query_start_loc_cpu"][1:].tolist(),
                seq_lens_list=kwargs["seq_lens_np"].tolist(),
                block_tables=kwargs["block_tables"][0],
                num_actual_tokens=kwargs["num_actual_tokens"],
            )
        }

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
    state._token_positions = torch.arange(256, dtype=torch.int32)
    state._token_causal = torch.empty(256, dtype=torch.bool)
    state.max_model_len = 1024
    state.diffusion_states = SimpleNamespace(is_encoder_phase=torch.tensor([False, True, False]))
    offsets = [0, 64, 128, 192, 192]
    batch = SimpleNamespace(
        num_reqs=3,
        num_reqs_after_padding=4,
        num_tokens=192,
        num_tokens_after_padding=256,
        query_start_loc=torch.tensor(offsets, dtype=torch.int32),
        query_start_loc_np=np.array(offsets, dtype=np.int32),
        idx_mapping=torch.tensor([1, 2, 0]),
        seq_lens=torch.tensor([80, 96, 112, 0], dtype=torch.int32),
        num_scheduled_tokens=np.array([64, 64, 64]),
    )
    table = torch.arange(4, dtype=torch.int32).view(4, 1)
    metadata = state.prepare_attn(batch, GraphMode.FULL, [table], None, [], None)
    provider = _load_class("attention/attention_v1.py", "FIAParamProvider", dataclass=dataclass, Any=Any)
    for causal in (True, False):
        params = provider("layer", None, diffusion_causal=causal).resolve(metadata)
        assert params["actual_seq_lengths"] == [64, 128, 192, 256]
        assert params["actual_seq_lengths_kv"] == [80, 96, 112, 0]
        assert params["block_table"] is table
    assert metadata["layer"].num_actual_tokens == 192
    assert batch.query_start_loc.tolist() == batch.query_start_loc_np.tolist() == offsets


def test_diffusion_causal_fia_window_matches_upstream_boundary():
    mask_params = _load_class(
        "attention/attention_v1.py",
        "AscendAttentionBackendImpl",
        method="_get_fia_mask_params",
        AscendMetadata=object,
        SWA_INT_MAX=2147483647,
        torch=torch,
    )
    impl = SimpleNamespace(sliding_window=1024)
    metadata = SimpleNamespace(causal=True, attn_mask=None, diffusion_causal=torch.tensor([True]))
    _, sparse_mode, left, right = mask_params(impl, metadata, torch.device("cpu"))
    assert (sparse_mode, left, right) == (4, 1023, 0)
    assert 1024 - left == 1
    metadata.diffusion_causal = None
    assert mask_params(impl, metadata, torch.device("cpu"))[2] == 1024
