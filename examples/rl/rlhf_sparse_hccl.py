# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
r"""Checkpoint-coordinate sparse HCCL updates, adapted from vLLM's
examples/rl/rlhf_sparse_nccl.py.

Start trainer and server from the SAME unquantized Qwen3 MoE checkpoint.
Example with one trainer NPU and TP2 inference on two separate NPUs:

    ASCEND_RT_VISIBLE_DEVICES=1,2 VLLM_SERVER_DEV_MODE=1 \
      vllm serve /path/to/checkpoint --tensor-parallel-size 2 \
      --enable-expert-parallel --enforce-eager \
      --additional-config '{"weight_nz_mode":0}' \
      --weight-transfer-config '{"backend":"sparse_hccl"}'
    ASCEND_RT_VISIBLE_DEVICES=0 python examples/rl/rlhf_sparse_hccl.py \
      --model /path/to/checkpoint --tp 2

The trainer swaps two gate-projection rows in two global experts in both the
first and last layer. Pipeline parallelism is unsupported. Full checkpoint names and
shapes are sent; TP/EP mapping is delegated to the native loader. The trainer
needs enough memory for the full BF16 model. Updates are nontransactional:
generation remains paused if initialization or transfer fails.
"""

import argparse

import requests
import torch
from vllm.distributed.weight_transfer import HTTPVLLMWeightSyncClient, WeightTransferTrainerFactory
from vllm.utils.network_utils import get_ip, get_open_port

from vllm_ascend.distributed.weight_transfer import register_engine
from vllm_ascend.distributed.weight_transfer.sparse_hccl_engine import SparseHCCLTrainerInitInfo, SparseWeightPatch


@torch.no_grad()
def make_patches(model: torch.nn.Module) -> list[SparseWeightPatch]:
    """Convert HF fused/per-expert storage to full checkpoint gate coordinates."""
    config = model.config
    if config.model_type != "qwen3_moe" or config.num_experts < 2 or config.moe_intermediate_size < 2:
        raise ValueError("Requires Qwen3 MoE with at least two experts and two intermediate rows")
    patches = []
    for layer in sorted({0, config.num_hidden_layers - 1}):
        prefix = f"model.layers.{layer}.mlp.experts"
        for expert in (0, config.num_experts // 2):
            name = f"{prefix}.{expert}.gate_proj.weight"
            try:
                weight = model.get_parameter(name)
            except AttributeError:
                fused = model.get_parameter(f"{prefix}.gate_up_proj")
                expected = (config.num_experts, 2 * config.moe_intermediate_size, config.hidden_size)
                if tuple(fused.shape) != expected:
                    raise ValueError(f"Unexpected fused expert shape: {tuple(fused.shape)} != {expected}") from None
                weight = fused[expert, : config.moe_intermediate_size]
            if tuple(weight.shape) != (config.moe_intermediate_size, config.hidden_size):
                raise ValueError(f"Unexpected checkpoint shape for {name}: {tuple(weight.shape)}")
            replacement = weight[:2].clone().flip(0)
            if torch.equal(weight[:2], replacement):
                raise ValueError(f"Row swap would be a no-op: {name}")
            weight[:2].copy_(replacement)
            patches.append(
                SparseWeightPatch(
                    name=name,
                    full_shape=tuple(weight.shape),
                    indices=torch.arange(replacement.numel(), dtype=torch.int32, device=weight.device),
                    values=replacement.flatten().contiguous(),
                )
            )
    return patches


def generate(url: str, model: str) -> list[int]:
    response = requests.post(
        f"{url}/v1/completions",
        json={"model": model, "prompt": [1, 13, 42, 77], "max_tokens": 4, "temperature": 0, "return_token_ids": True},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["token_ids"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Same checkpoint path/model ID as the server")
    parser.add_argument("--served-model-name", help="Server model alias; defaults to --model")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--dp", type=int, default=1)
    args = parser.parse_args()
    if min(args.tp, args.dp) < 1:
        parser.error("--tp and --dp must be positive and match the server")

    import torch_npu  # noqa: F401
    from transformers import AutoModelForCausalLM

    torch.npu.set_device(0)
    register_engine()
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("npu:0")
    model.eval()
    url = args.url.rstrip("/")
    served_model = args.served_model_name or args.model
    print("before_token_ids=", generate(url, served_model))
    patches = make_patches(model)
    response = requests.post(f"{url}/pause", timeout=60)
    response.raise_for_status()
    engine = None
    try:
        engine = WeightTransferTrainerFactory.trainer_init(
            init_info=SparseHCCLTrainerInitInfo(get_ip(), get_open_port(), 1 + args.tp * args.dp, rank=0),
            client=HTTPVLLMWeightSyncClient(url),
        )
        engine.send_weights(patches)
    finally:
        if engine is not None:
            engine.shutdown()
    # Reached only on success: failed updates must remain paused for recovery.
    response = requests.post(f"{url}/resume", timeout=60)
    response.raise_for_status()
    print("patched_checkpoint_names=", [patch.name for patch in patches])
    print("num_sparse_values=", sum(patch.indices.numel() for patch in patches))
    print("after_token_ids=", generate(url, served_model))


if __name__ == "__main__":
    main()
