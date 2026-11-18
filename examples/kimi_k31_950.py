# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preflight and launch the v30 Kimi K3.1 backport on one 950DT node.

The default is a small eager, text-only acceptance run using the complete
W4A8_MXFP checkpoint. See docs/source/tutorials/models/Kimi-K3.1-v30-950.md.
"""

import argparse
import importlib
import importlib.metadata
import json
import os
import shlex
import shutil
import sys
from pathlib import Path


def validate_checkpoint(model: Path, tp: int) -> dict:
    config = json.loads((model / "config.json").read_text())
    if config.get("model_type") != "kimi_k31" or config.get("architectures") != ["KimiK31ForConditionalGeneration"]:
        raise ValueError("Expected a kimi_k31 / KimiK31ForConditionalGeneration checkpoint")
    text = config["text_config"]
    dsa = text.get("dsa_config")
    if not dsa:
        raise ValueError("K3.1 text_config.dsa_config is required")
    linear = text["linear_attn_config"]
    layers = set(dsa["dsa_layers"])
    if layers != set(linear["full_attn_layers"]) or layers.intersection(linear["kda_layers"]):
        raise ValueError("DSA layers must match full attention layers and be disjoint from KDA")
    if not layers or min(layers) < 1 or max(layers) > text["num_hidden_layers"]:
        raise ValueError("DSA layer numbers must be one-based and inside the model")
    if tp < 1 or text["num_attention_heads"] % tp or linear["num_heads"] % tp:
        raise ValueError("TP must divide both MLA and KDA head counts")
    quant = json.loads((model / "quant_model_description.json").read_text())
    if "MXFP" not in json.dumps(quant).upper():
        raise ValueError("Use the 950DT W4A8_MXFP ModelSlim checkpoint, including its quantization description")
    index_path = model / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shards = set(index["weight_map"].values())
        if not shards or any(not (model / shard).is_file() for shard in shards):
            raise ValueError("Checkpoint index references missing weight shards")
    elif not (model / "model.safetensors").is_file():
        raise ValueError("Expected model.safetensors or a complete model.safetensors.index.json")
    return config


def check_runtime(tp: int, config: dict) -> None:
    # Import device dependencies only for the hardware preflight. Dry-run
    # validates local files and command arguments without requiring an NPU.
    torch = importlib.import_module("torch")
    torch_npu = importlib.import_module("torch_npu")
    version = importlib.metadata.version("vllm")
    if not version.startswith("0.30."):
        raise RuntimeError(f"This launcher requires the pinned vLLM v30 source, found {version}")
    if not torch_npu.npu.is_available() or torch_npu.npu.device_count() < tp:
        raise RuntimeError(f"At least {tp} visible 950DT devices are required")
    names = [torch_npu.npu.get_device_name(i) for i in range(tp)]
    if any("950" not in name or "PR" in name.upper() for name in names):
        raise RuntimeError(f"This acceptance recipe targets 950DT, found {names}")
    utils = importlib.import_module("vllm_ascend.utils")
    if not utils.is_950():
        raise RuntimeError("vLLM Ascend was not built for the 950 device family")
    utils.enable_custom_op()
    required = {
        "attn_res_fwd": "return_materialized",
        "npu_quant_lightning_indexer_v3": "return_value",
        "npu_quant_lightning_indexer_v2_metadata": "causal_seqlens_kv_global",
        "npu_kv_quant_sparse_flash_attention_vllm": "query",
    }
    for name, marker in required.items():
        op = getattr(torch.ops._C_ascend, name, None)
        if op is None or marker not in str(op.default._schema):
            raise RuntimeError(f"Missing or stale _C_ascend.{name}; rebuild custom kernels from this checkout")
    if config["text_config"].get("mla_use_nope", False):
        transformer_ops = importlib.import_module("cann_ops_transformer")
        if not hasattr(transformer_ops, "mla_prolog"):
            raise RuntimeError("NoPE K3.1 requires cann_ops_transformer.mla_prolog")
    patch = importlib.import_module("vllm_ascend.patch.platform.patch_kimi_k31_config")
    hf_config = patch.KimiK31Config(**config)
    layers = hf_config.text_config.dsa_config["dsa_layers"]
    print(f"Preflight passed: vLLM {version}, devices={names}, DSA layers={layers}")
    print(f"Ascend source: {Path(utils.__file__).resolve()}")


def build_command(args: argparse.Namespace) -> list[str]:
    additional = {
        "enable_mlapo": False,
        "enable_flashcomm1": False,
        "enable_shared_expert_dp": False,
        "multistream_overlap_shared_expert": False,
        "enable_fused_mc2": 0,
    }
    command = [
        "vllm",
        "serve",
        str(args.model.resolve()),
        "--served-model-name",
        "kimi-k3.1-pro",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tp),
        "--pipeline-parallel-size",
        "1",
        "--prefill-context-parallel-size",
        "1",
        "--decode-context-parallel-size",
        "1",
        "--distributed-executor-backend",
        "mp",
        "--enable-expert-parallel",
        "--quantization",
        "ascend",
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "fp8",
        "--attention-config.indexer_kv_dtype",
        "fp8",
        "--block-size",
        "1536",
        "--cp-kv-cache-interleave-size",
        "1536",
        "--tokenizer-mode",
        "kimi_k3",
        "--trust-remote-code",
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        "2048",
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        str(args.memory_utilization),
        "--enable-prefix-caching",
        "--no-mm-device-do-normalize",
        "--enforce-eager",
        "--additional-config",
        json.dumps(additional),
        "--seed",
        "1024",
    ]
    if not args.multimodal:
        command.append("--language-model-only")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--memory-utilization", type=float, default=0.90)
    parser.add_argument("--multimodal", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_model_len < 2048 or not 0 < args.memory_utilization < 1 or not 0 < args.port < 65536:
        parser.error("Require max-model-len >= 2048, memory-utilization in (0,1), and port in 1..65535")
    try:
        config = validate_checkpoint(args.model, args.tp)
        command = build_command(args)
        print(shlex.join(command), flush=True)
        if args.dry_run:
            return
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        os.environ["VLLM_VERSION"] = "0.30.0"
        os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "CCU_SCHED")
        os.environ.setdefault("HCCL_BUFFSIZE", "800")
        os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
        check_runtime(args.tp, config)
        if args.check_only:
            return
        if shutil.which(command[0]) is None:
            raise RuntimeError("vllm executable is missing from PATH")
        sys.stdout.flush()
        os.execvp(command[0], command)
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        parser.exit(1, f"K3.1 preflight failed: {exc}\n")


if __name__ == "__main__":
    main()
