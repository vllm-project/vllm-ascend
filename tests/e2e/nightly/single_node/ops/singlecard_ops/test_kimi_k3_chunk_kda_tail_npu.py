#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

"""Regression coverage for Kimi-K3 BF16 chunk-KDA tail execution."""

import math
from dataclasses import dataclass

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

torch_npu.npu.config.allow_internal_format = True
enable_custom_op()

_CHUNK_SIZE = 64
_HEADS = 6
_HEAD_DIM = 128
_LOWER_BOUND = -5.0
_MAX_REASONABLE_ABS = 1.0e6
_TAIL_TEST_TOKENS = (
    128,
    129,
    130,
    131,
    143,
    144,
    145,
    159,
    160,
    161,
    191,
    192,
    193,
)
_OUTPUT_NAMES = (
    "output",
    "final_state",
    "gk",
    "aqk",
    "akk",
    "w",
    "u",
    "qg",
    "kg",
    "v_new",
    "h",
    "initial_state",
)


@dataclass
class _ChunkKdaInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    raw_gate: torch.Tensor
    activated_gate: torch.Tensor
    beta: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    initial_state: torch.Tensor
    cu_seqlens: tuple[int, ...]
    chunk_indices: tuple[int, ...]

    def clone(self) -> "_ChunkKdaInputs":
        return _ChunkKdaInputs(
            q=self.q.clone(),
            k=self.k.clone(),
            v=self.v.clone(),
            raw_gate=self.raw_gate.clone(),
            activated_gate=self.activated_gate.clone(),
            beta=self.beta.clone(),
            a_log=self.a_log.clone(),
            dt_bias=self.dt_bias.clone(),
            initial_state=self.initial_state.clone(),
            cu_seqlens=self.cu_seqlens,
            chunk_indices=self.chunk_indices,
        )


def _l2norm(value: torch.Tensor) -> torch.Tensor:
    dtype = value.dtype
    value_fp32 = value.float()
    return (value_fp32 * torch.rsqrt((value_fp32 * value_fp32).sum(dim=-1, keepdim=True) + 1e-6)).to(dtype)


def _build_inputs(tokens: int) -> _ChunkKdaInputs:
    seed = 20260820 + tokens
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)

    shape = (1, tokens, _HEADS, _HEAD_DIM)
    q = _l2norm(torch.randn(shape, device="npu", dtype=torch.bfloat16))
    k = _l2norm(torch.randn(shape, device="npu", dtype=torch.bfloat16))
    v = torch.randn(shape, device="npu", dtype=torch.bfloat16) * 0.2
    raw_gate = torch.randn(shape, device="npu", dtype=torch.bfloat16) * 2.0
    beta = torch.sigmoid(torch.randn((1, tokens, _HEADS), device="npu", dtype=torch.float32))
    a_log = torch.empty((_HEADS,), device="npu", dtype=torch.float32).uniform_(-0.5, 0.8)
    dt_bias = torch.empty((_HEADS * _HEAD_DIM,), device="npu", dtype=torch.float32).uniform_(-7.5, -1.5)
    initial_state = torch.zeros(
        (1, _HEADS, _HEAD_DIM, _HEAD_DIM),
        device="npu",
        dtype=torch.float32,
    )
    activated_gate = _LOWER_BOUND * torch.sigmoid(
        (raw_gate.float() + dt_bias.view(1, 1, _HEADS, _HEAD_DIM)) * a_log.exp().view(1, 1, _HEADS, 1)
    )
    chunk_indices = tuple(value for chunk_index in range(math.ceil(tokens / _CHUNK_SIZE)) for value in (0, chunk_index))
    return _ChunkKdaInputs(
        q=q,
        k=k,
        v=v,
        raw_gate=raw_gate,
        activated_gate=activated_gate,
        beta=beta,
        a_log=a_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        cu_seqlens=(0, tokens),
        chunk_indices=chunk_indices,
    )


def _final_state_reference(inputs: _ChunkKdaInputs) -> torch.Tensor:
    k = inputs.k.detach().cpu()
    v = inputs.v.detach().cpu()
    gate = inputs.activated_gate.detach().cpu()
    beta = inputs.beta.detach().cpu()
    initial_state = inputs.initial_state.detach().cpu()
    _, tokens, heads, head_dim = k.shape
    final_state = torch.empty_like(initial_state, dtype=torch.float32)

    for head_index in range(heads):
        state_kv = initial_state[0, head_index].float().transpose(-1, -2).contiguous()
        for start in range(0, tokens, _CHUNK_SIZE):
            end = min(start + _CHUNK_SIZE, tokens)
            chunk_tokens = end - start
            strict_causal = torch.ones((chunk_tokens, chunk_tokens), dtype=torch.bool).tril(diagonal=-1)
            eye = torch.eye(chunk_tokens, dtype=torch.float32)
            k_block = k[0, start:end, head_index].float()
            v_block = v[0, start:end, head_index].float()
            beta_block = beta[0, start:end, head_index].float()
            gk_block = torch.cumsum(gate[0, start:end, head_index].float(), dim=0) / math.log(2.0)
            relative_gate = gk_block[:, None, :] - gk_block[None, :, :]
            gate_factor = torch.exp2(relative_gate.masked_fill(~strict_causal[:, :, None], 0.0))
            kk = torch.einsum("ik,jk,ijk->ij", k_block, k_block, gate_factor)
            strict_kk = torch.where(strict_causal, kk * beta_block[:, None], 0.0)
            akk_block = torch.linalg.solve_triangular(eye + strict_kk, eye, upper=False)
            w_block = akk_block @ (k_block * beta_block[:, None] * torch.exp2(gk_block))
            u_block = akk_block @ (v_block * beta_block[:, None])
            kg_block = k_block * torch.exp2(gk_block[-1][None, :] - gk_block)
            v_new_block = u_block - w_block @ state_kv
            state_kv = torch.exp2(gk_block[-1])[:, None] * state_kv + kg_block.T @ v_new_block
        final_state[0, head_index] = state_kv.transpose(-1, -2)
    return final_state


def _run_chunk_kda(inputs: _ChunkKdaInputs, gate_mode: str, metadata_mode: str):
    use_gate_in_kernel = gate_mode == "raw_gate"
    gate = inputs.raw_gate if use_gate_in_kernel else inputs.activated_gate
    use_varlen_metadata = metadata_mode == "varlen"
    return torch.ops._C_ascend.chunk_kda_fwd(
        inputs.q,
        inputs.k,
        inputs.v,
        gate,
        inputs.beta,
        _HEAD_DIM**-0.5,
        _CHUNK_SIZE,
        layout="BSND",
        initial_state=inputs.initial_state,
        output_final_state=True,
        cu_seqlens=inputs.cu_seqlens if use_varlen_metadata else None,
        chunk_indices=inputs.chunk_indices if use_varlen_metadata else None,
        safe_gate=True,
        lower_bound=_LOWER_BOUND,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=inputs.a_log if use_gate_in_kernel else None,
        dt_bias=inputs.dt_bias if use_gate_in_kernel else None,
        disable_recompute=True,
        return_intermediate_states=False,
        state_v_first=True,
    )


def _snapshot_outputs(outputs) -> tuple[torch.Tensor | None, ...]:
    torch.npu.synchronize()
    return tuple(output.detach().cpu().contiguous() if isinstance(output, torch.Tensor) else None for output in outputs)


def _describe_difference(
    name: str,
    first: torch.Tensor | None,
    second: torch.Tensor | None,
) -> str | None:
    if first is None or second is None:
        if first is None and second is None:
            return None
        return f"{name}: missing output first={type(first).__name__} second={type(second).__name__}"
    same_metadata = first.shape == second.shape and first.dtype == second.dtype
    first_fp32 = first.float()
    second_fp32 = second.float()
    values_are_finite = torch.isfinite(first_fp32).all().item() and torch.isfinite(second_fp32).all().item()
    values_are_reasonable = (
        first_fp32.abs().max().item() <= _MAX_REASONABLE_ABS and second_fp32.abs().max().item() <= _MAX_REASONABLE_ABS
    )
    same_bits = same_metadata and torch.equal(first.view(torch.uint8), second.view(torch.uint8))
    if same_bits and values_are_finite and values_are_reasonable:
        return None

    if same_metadata:
        changed = first.view(torch.uint8) != second.view(torch.uint8)
        differing_elements = int(changed.reshape(-1, first.element_size()).any(dim=1).sum().item())
        max_abs_diff = (first.double() - second.double()).abs().max().item()
    else:
        differing_elements = -1
        max_abs_diff = float("nan")
    return (
        f"{name}: shape_first={tuple(first.shape)} "
        f"shape_second={tuple(second.shape)} dtype_first={first.dtype} "
        f"dtype_second={second.dtype} max_abs_diff={max_abs_diff:.8e} "
        f"differing_elements={differing_elements} "
        f"values_are_finite={values_are_finite} "
        f"values_are_reasonable={values_are_reasonable}"
    )


@pytest.mark.parametrize(
    "tokens",
    _TAIL_TEST_TOKENS,
    ids=lambda tokens: f"tokens_{tokens}_remainder_{tokens % _CHUNK_SIZE}",
)
@pytest.mark.parametrize("gate_mode", ["external_gate", "raw_gate"])
@pytest.mark.parametrize("metadata_mode", ["dense", "varlen"])
@torch.inference_mode()
def test_kimi_k3_chunk_kda_bf16_tail_is_deterministic(tokens: int, gate_mode: str, metadata_mode: str):
    if not hasattr(torch.ops._C_ascend, "chunk_kda_fwd"):
        pytest.skip("requires the fused chunk KDA AscendC operator")

    inputs = _build_inputs(tokens)
    # Allocate both sets before the first launch so an out-of-bounds write from
    # that launch cannot change tensors allocated for the second invocation.
    first_inputs = inputs.clone()
    second_inputs = inputs.clone()
    first = _snapshot_outputs(_run_chunk_kda(first_inputs, gate_mode, metadata_mode))
    second = _snapshot_outputs(_run_chunk_kda(second_inputs, gate_mode, metadata_mode))
    torch.testing.assert_close(
        first[1],
        _final_state_reference(inputs),
        rtol=3e-2,
        atol=3e-2,
        msg=(f"final_state accuracy failed for tokens={tokens}, gate_mode={gate_mode}, metadata_mode={metadata_mode}"),
    )

    problems = [
        problem
        for name, first_output, second_output in zip(_OUTPUT_NAMES, first, second)
        if (problem := _describe_difference(name, first_output, second_output)) is not None
    ]
    assert not problems, (
        f"Kimi-K3 chunk KDA is not deterministic for tokens={tokens}, "
        f"remainder={tokens % _CHUNK_SIZE}, gate_mode={gate_mode}, "
        f"metadata_mode={metadata_mode}:\n" + "\n".join(problems)
    )
