# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the constructor's HC allocation without constructing decoder weights."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from vllm.config import SpeculativeConfig


@pytest.mark.parametrize("method", [None, "dspark", "mtp", "eagle", "eagle3", "dflash", "draft_model"])
@pytest.mark.parametrize("last_pp_rank", [False, True])
def test_target_hc_buffer_allocation(method, last_pp_rank):
    # Execute the production allocation block with real CPU tensors. Importing
    # the full model would require unrelated native operators and model weights.
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/model.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    model = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "DeepseekV41Model")
    initializer = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    names = {"spec_config", "needs_mtp_hidden_states", "self._mtp_hidden_buffer"}
    allocation: list[ast.stmt] = [
        node
        for node in initializer.body
        if isinstance(node, ast.Assign) and any(ast.unparse(target) in names for target in node.targets)
    ]
    spec = None
    if method is not None:
        spec = SpeculativeConfig.__new__(SpeculativeConfig)
        object.__setattr__(spec, "method", method)
    config = SimpleNamespace(hidden_size=8)
    state = SimpleNamespace(hc_mult=4, device="cpu")
    namespace = {
        "self": state,
        "config": config,
        "torch": torch,
        "get_pp_group": lambda: SimpleNamespace(is_last_rank=last_pp_rank),
        "vllm_config": SimpleNamespace(
            speculative_config=spec,
            scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
            model_config=SimpleNamespace(dtype=torch.bfloat16),
        ),
    }
    exec(compile(ast.Module(body=allocation, type_ignores=[]), str(source), "exec"), namespace)
    buffer = state._mtp_hidden_buffer
    if method in (None, "dspark") or not last_pp_rank:
        assert buffer is None
    else:
        assert buffer.shape == (16, 32)
        assert buffer.dtype == torch.bfloat16
        assert buffer.numel() * buffer.element_size() == 1024


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("num_tokens", [0, 1, 3, 8, 17, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_sp", [False, True])
@pytest.mark.parametrize("has_mtp_buffer", [False, True])
def test_target_final_norm_before_gather(tp_size, num_tokens, dtype, use_sp, has_mtp_buffer):
    # Execute the production forward epilogue, including HC-buffer writes and
    # auxiliary returns, without importing unrelated NPU operators.
    root = Path(__file__).resolve().parents[3]
    source = root / "vllm_ascend/models/deepseek_v41/model.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    model = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "DeepseekV41Model")
    forward = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    start = next(
        index
        for index, node in enumerate(forward.body)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "self._mtp_hidden_buffer is not None"
    )
    epilogue = ast.FunctionDef(
        name="epilogue",
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg=name)
                for name in (
                    "self",
                    "last_layer",
                    "hidden_states",
                    "pre_mix",
                    "use_sequence_parallel",
                    "full_num_tokens",
                    "aux_hidden_states",
                )
            ],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=forward.body[start:],
        decorator_list=[],
    )
    decoder = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "DeepseekV41DecoderLayer"
    )
    collapse = next(node for node in decoder.body if isinstance(node, ast.FunctionDef) and node.name == "hc_collapse")
    collapse.decorator_list = []
    sp_source = root / "vllm_ascend/models/common/ops/sequence_parallel.py"
    shard = next(
        node
        for node in ast.parse(sp_source.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name == "_ascend_sp_shard_impl"
    )
    namespace = {"torch": torch, "F": F, "get_tensor_model_parallel_world_size": lambda: tp_size}
    code = ast.fix_missing_locations(ast.Module(body=[collapse, shard, epilogue], type_ignores=[]))
    exec(compile(code, str(source), "exec"), namespace)

    hidden_size, hc_mult = 16, 4
    generator = torch.Generator().manual_seed(42)
    hidden = torch.randn(num_tokens, hc_mult, hidden_size, generator=generator).to(dtype)
    mix = torch.randn(num_tokens, hc_mult, generator=generator)
    weight = torch.randn(hidden_size, generator=generator).to(dtype)
    bias = torch.randn(hidden_size, generator=generator).to(dtype)

    def norm(x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + 1e-6)
        return (normalized.to(dtype) * weight).add(bias)

    collapsed = namespace["hc_collapse"](hidden, mix)
    expected = norm(collapsed)
    local_rows = (num_tokens + tp_size - 1) // tp_size
    padded_rows = local_rows * tp_size
    gathered_hidden = F.pad(hidden, (0, 0, 0, 0, 0, padded_rows - num_tokens))
    gathered_mix = F.pad(mix, (0, 0, 0, padded_rows - num_tokens))
    # Poison padding to verify row-local normalization cannot contaminate
    # valid tokens, even when a rank owns no valid rows.
    gathered_hidden[num_tokens:] = float("nan")
    gathered_mix[num_tokens:] = float("nan")
    gathered_output = norm(namespace["hc_collapse"](gathered_hidden, gathered_mix))

    for rank in range(tp_size):
        calls = []
        namespace["get_tensor_model_parallel_rank"] = lambda rank=rank: rank
        local_hidden = namespace["_ascend_sp_shard_impl"](hidden) if use_sp else hidden.clone()
        local_mix = namespace["_ascend_sp_shard_impl"](mix) if use_sp else mix.clone()
        if use_sp:
            local_hidden.copy_(gathered_hidden[rank * local_rows : (rank + 1) * local_rows])
            local_mix.copy_(gathered_mix[rank * local_rows : (rank + 1) * local_rows])

        def recording_norm(x, calls=calls):
            calls.append(("norm", x.shape[0]))
            return norm(x)

        def gather(x, calls=calls, rank=rank):
            calls.append(("gather", x.shape[0]))
            if has_mtp_buffer:
                full = gathered_hidden if x.ndim == 3 else gathered_mix
            else:
                assert calls == [("norm", local_rows), ("gather", local_rows)]
                full = gathered_output
            torch.testing.assert_close(
                x,
                full[rank * local_rows : (rank + 1) * local_rows],
                equal_nan=True,
            )
            return full.clone()

        namespace["sp_all_gather"] = gather
        buffer = torch.full((num_tokens + 1, hc_mult * hidden_size), -123, dtype=dtype) if has_mtp_buffer else None
        state = SimpleNamespace(norm=recording_norm, _mtp_hidden_buffer=buffer)
        aux = [hidden.mean(dim=1)]
        actual, returned_aux = namespace["epilogue"](
            state,
            SimpleNamespace(hc_collapse=namespace["hc_collapse"]),
            local_hidden,
            local_mix,
            use_sp,
            num_tokens,
            aux,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.shape == (num_tokens, hidden_size)
        assert actual.dtype == dtype
        assert returned_aux is aux
        if has_mtp_buffer:
            torch.testing.assert_close(buffer[:num_tokens], hidden.flatten(1), rtol=0, atol=0)
            assert torch.all(buffer[num_tokens:] == -123)
        expected_calls = [("norm", num_tokens)]
        if use_sp:
            expected_calls = (
                [("gather", local_rows), ("gather", local_rows), ("norm", num_tokens)]
                if has_mtp_buffer
                else [("norm", local_rows), ("gather", local_rows)]
            )
        assert calls == expected_calls
