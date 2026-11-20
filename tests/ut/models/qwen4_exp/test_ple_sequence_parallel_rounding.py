# SPDX-License-Identifier: Apache-2.0
"""CPU regression for PLE residual order and sequence-parallel token layout.

Execute the production forward and sharding definitions without importing the
NPU model. Projection/gating use controlled tensors; the convolution stub
returns nonzero Torch SiLU values. Native convolution rounding is covered by
the NPU operator tests, not this CPU test.
"""

import __future__

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
CHANNELS = 2


def _load_function(path, name, namespace, class_name=None):
    source = ROOT / path
    definitions = ast.parse(source.read_text(encoding="utf-8")).body
    if class_name is not None:
        definitions = next(
            node.body for node in definitions if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    function = next(node for node in definitions if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(source), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


class _Embedding:
    def __init__(self, table):
        self.table = table
        self.ngram_embedding = SimpleNamespace(dequantize=lambda tensor, _dtype: tensor)

    def __call__(self, input_ids, _query_start_loc, _ngram_context):
        # Each vocabulary owner sees the complete token set before SP slicing.
        torch.testing.assert_close(input_ids, torch.arange(self.table.shape[0], device="cpu"))
        return self.table.clone()


class _Projection:
    output_sizes = (CHANNELS, CHANNELS)

    def __call__(self, embeddings):
        return embeddings, None


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("num_tokens", [1, 3, 4, 5])
def test_ple_sp_preserves_staged_residual_rounding_and_token_order(dtype, tp_size, num_tokens):
    # Channel zero is a reachable counterexample: gated=1, hidden=-1 and
    # SiLU(conv_input=1). Channel one varies by token to expose row misalignment.
    rows = torch.arange(num_tokens, dtype=torch.float32, device="cpu")
    gated = torch.stack((torch.ones_like(rows), 0.3125 + rows / 32), dim=-1).to(dtype)
    hidden = torch.stack((-torch.ones_like(rows), 0.7 - rows / 16), dim=-1).to(dtype)
    conv_input = torch.stack((torch.ones_like(rows), (0.1 + rows / 8) * (-1.0) ** rows), dim=-1).to(dtype)
    conv_output = F.silu(conv_input.float()).to(dtype)
    expected = (gated + conv_output) + hidden
    reassociated = (gated + hidden) + conv_output
    assert not torch.equal(expected[:, 0], reassociated[:, 0])

    padded_tokens = ((num_tokens + tp_size - 1) // tp_size) * tp_size

    def padded(tensor):
        return F.pad(tensor, (0, 0, 0, padded_tokens - num_tokens))

    def check_rank(rank):
        shard = _load_function(
            "vllm_ascend/models/common/ops/sequence_parallel.py",
            "_ascend_sp_shard_impl",
            {
                "torch": torch,
                "F": F,
                "get_tensor_model_parallel_world_size": lambda: tp_size,
                "get_tensor_model_parallel_rank": lambda rank=rank: rank,
            },
        )
        gathered_payloads = iter((conv_input, expected))
        gathered_widths = []

        def all_gather(local):
            full = next(gathered_payloads)
            # Check the actual contribution from this rank before supplying
            # precomputed peer rows. The final gather cannot hide wrong adds.
            torch.testing.assert_close(local, shard(full), rtol=0, atol=0)
            gathered_widths.append(local.shape[-1])
            return padded(full).clone()

        def gate(key, value, local_hidden, *_weights):
            assert key.shape == value.shape == local_hidden.shape
            return key.clone(), value.clone()

        def short_conv(inputs, residual, outer_residual, _prefix):
            assert inputs.is_contiguous()
            torch.testing.assert_close(inputs, conv_input, rtol=0, atol=0)
            assert torch.count_nonzero(residual) == 0
            assert torch.count_nonzero(outer_residual) == 0
            residual.copy_(conv_output)

        torch_namespace = SimpleNamespace(
            Tensor=torch.Tensor,
            cat=torch.cat,
            zeros_like=torch.zeros_like,
            ops=SimpleNamespace(vllm=SimpleNamespace(qwen4_exp_ple_short_conv=short_conv)),
        )
        forward = _load_function(
            "vllm_ascend/models/qwen4_exp/model.py",
            "forward",
            {
                "torch": torch_namespace,
                "upstream_ple": SimpleNamespace(ple_gate=gate),
                "sp_shard": shard,
                "sp_all_gather": all_gather,
            },
            class_name="AscendQwen4ExpPLELayer",
        )
        norm = SimpleNamespace(weight=None, eps=1e-6)
        layer = SimpleNamespace(
            use_sequence_parallel=True,
            prefix="layer.ple",
            ple_embedding=_Embedding(torch.cat((gated, conv_input), dim=-1)),
            kv_proj=_Projection(),
            norm_key=norm,
            norm_query=norm,
            norm_conv=norm,
        )
        actual = forward(
            layer,
            shard(hidden),
            torch.arange(num_tokens, device="cpu"),
            torch.tensor([0, num_tokens], dtype=torch.int32, device="cpu"),
            torch.zeros((1, 1), dtype=torch.int32, device="cpu"),
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.shape == (num_tokens, CHANNELS)
        assert gathered_widths == [CHANNELS, CHANNELS]

    for rank in range(tp_size):
        check_rank(rank)
