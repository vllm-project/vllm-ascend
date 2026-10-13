# SPDX-License-Identifier: Apache-2.0
"""Check the Python payload/schema boundary without a model or NPU group."""

import ast
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS

import torch


class Activation(Enum):
    SILU = "silu"
    SWIGLUSTEP = "swiglustep"
    GELU = "gelu"


def test_activation_roundtrip_and_explicit_tensor_inputs():
    source = Path(__file__).resolve().parents[2] / "vllm_ascend/ops/fused_moe/alltoall_region.py"
    tree = ast.parse(source.read_text())
    functions = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("alltoall_routed_experts", "run_alltoall_routed_region"):
            # Replace only runtime imports/registration with test collaborators.
            # The actual flattening, schema arguments and reconstruction remain.
            node.decorator_list = []
            node.body = [part for part in node.body if not isinstance(part, (ast.Import, ast.ImportFrom))]
            functions.append(node)
    method = NS(moe_config=NS(num_local_experts=3))
    seen = []

    def execute(instance, payload):
        assert instance is method
        assert isinstance(payload.activation, Activation)
        seen.append(payload)
        return NS(routed_out=payload.hidden_states + 1, expert_tokens=torch.tensor([2, 1, 0], dtype=torch.int32))

    namespace = dict(
        torch=torch,
        MoEActivation=Activation,
        QuantType=NS(W8A8="w8a8"),
        MoECommType=NS(ALLTOALL="alltoall"),
        MoECommMethod=NS(fused_experts=execute),
        get_moe_comm_method=lambda kind: method,
        build_fused_experts_input=lambda **kwargs: NS(**kwargs),
        FusedExpertsResult=NS,
    )
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    implementation = namespace["alltoall_routed_experts"]

    def schema_boundary(*args):
        assert isinstance(args[15], str), "Torch schema cannot accept MoEActivation Enum"
        return implementation(*args)

    namespace["alltoall_routed_experts"] = schema_boundary
    x = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    weight = torch.ones(3, 8, 8)
    payload = NS(
        hidden_states=x,
        topk_weights=torch.ones(2, 1),
        topk_ids=torch.zeros(2, 1, dtype=torch.int64),
        weights=NS(
            w1=weight,
            w2=[weight],
            w1_scale=weight,
            w2_scale=[weight],
            w1_bias=None,
            w2_bias=None,
            w1_scale_bias=None,
            w2_scale_bias=None,
            w1_offset=None,
            w2_offset=None,
        ),
        quant=NS(quant_type="w8a8", comm_quant_mode=2, is_per_channel_weight=False),
        routing=NS(
            expert_map=None,
            log2phy=None,
            pertoken_scale=None,
            mc2_mask=None,
            global_redundant_expert_num=0,
            apply_router_weight_on_input=False,
        ),
        dynamic_eplb=False,
        lora_context=None,
        activation=Activation.SILU,
        need_trans=False,
        swiglu_limit=10.0,
    )
    for activation in Activation:
        payload.activation = activation
        actual = namespace["run_alltoall_routed_region"](method, payload)
        assert seen[-1].activation is activation
        assert seen[-1].w1[0] is weight and seen[-1].w2[0] is weight
        assert torch.equal(actual.routed_out, x + 1)
        assert actual.expert_tokens.dtype == torch.int64
        assert actual.expert_tokens.tolist() == [2, 1, 0]


if __name__ == "__main__":
    test_activation_roundtrip_and_explicit_tensor_inputs()
    print("PASS: activation enum/schema roundtrip and explicit weight inputs")
