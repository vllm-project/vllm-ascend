#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""Compare dense, QuaRot native-KV, and QuaRot KV4 per-layer attention inputs/outputs.

This debug tool runs the same prompt across comparable paths:
  - dense: normal dense model inference with no QuaRot quantization;
  - quarot_native: QuaRot W4A4 model with vLLM native dense KV cache;
  - kv4: QuaRot W4A4 model with QuaRot int4 KV cache.

Each child process patches AscendAttentionBackendImpl.forward_impl before LLM
construction and captures the last-token query, key, and attention backend
output for every attention call. It also patches the Qwen o_proj,
post-attention RMSNorm, and logits processor to capture the downstream points
that can amplify small attention differences. The parent keeps the final
captured occurrence for each layer, so multi-token runs compare the last
generated token layer by layer.

Full layer tensor dumps are temporary by default: exact tensor diffs are saved as
summary.json and layer_*.pt files are removed after comparison unless
--keep-layer-dumps is set.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

DEFAULT_DENSE_MODEL = "/data/weights/qwen3-8B"
DEFAULT_QUAROT_MODEL = "/workspace/Qwen3-8B-QuaRot-W4A4-q_random-debug-perchannel"
DEFAULT_PROMPT = "The capital of France is"
_CAPTURE_MODES = ("dense", "quarot_native", "kv4")
_MODE_ALIASES = {"quarot": "quarot_native"}
_LAYER_TENSOR_KEYS = (
    "query",
    "key",
    "value",
    "backend_output",
    "o_proj_input",
    "o_proj_output",
    "post_attention_residual",
    "mlp_u",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("both", "dense", "quarot", "quarot_native", "kv4", "compare"),
        default="both",
        help="'both' runs dense, QuaRot native-KV, and KV4 captures, then compares attention outputs.",
    )
    parser.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--quarot-model", default=DEFAULT_QUAROT_MODEL)
    parser.add_argument(
        "--model",
        default=None,
        help="Deprecated alias for --quarot-model, kept for older command lines.",
    )
    parser.add_argument("--dense-quantization", default="")
    parser.add_argument("--quarot-quantization", default="ascend")
    parser.add_argument(
        "--quantization",
        default=None,
        help="Deprecated alias for --quarot-quantization, kept for older command lines.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--max-model-len", type=int, default=64)
    parser.add_argument("--out-dir", default="/tmp/quarot_dense_attention_compare")
    parser.add_argument(
        "--npu-device",
        default=os.getenv("ASCEND_RT_VISIBLE_DEVICES", "6"),
        help="Value for ASCEND_RT_VISIBLE_DEVICES in child capture runs.",
    )
    parser.add_argument(
        "--max-capture-layers",
        type=int,
        default=0,
        help="Deprecated: 0 means capture all attention calls.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=36,
        help="Transformer layer count used to map attention call index to layer index.",
    )
    parser.add_argument(
        "--keep-layer-dumps",
        action="store_true",
        help="Keep full layer_*.pt tensor dumps after comparison.",
    )
    return parser


def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.mode = _MODE_ALIASES.get(args.mode, args.mode)
    if args.model is not None:
        args.quarot_model = args.model
    if args.quantization is not None:
        args.quarot_quantization = args.quantization
    return args


def _mode_model_and_quantization(mode: str, args: argparse.Namespace) -> tuple[str, str | None]:
    if mode == "dense":
        return args.dense_model, args.dense_quantization or None
    if mode in {"quarot_native", "kv4"}:
        return args.quarot_model, args.quarot_quantization or None
    raise ValueError(f"capture mode must be dense, quarot_native, or kv4, got {mode!r}")


def _missing_diff(reason: str) -> dict[str, Any]:
    return {
        "same_shape": False,
        "left_shape": None,
        "right_shape": None,
        "max": float("inf"),
        "mean": float("inf"),
        "cosine": None,
        "allclose_1e_2": False,
        "missing": reason,
    }


def _diff_stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if tuple(left.shape) != tuple(right.shape):
        return {
            "same_shape": False,
            "left_shape": tuple(left.shape),
            "right_shape": tuple(right.shape),
            "max": float("inf"),
            "mean": float("inf"),
            "cosine": None,
            "allclose_1e_2": False,
        }
    left_f = left.to(torch.float32)
    right_f = right.to(torch.float32)
    diff = (left_f - right_f).abs()
    left_flat = left_f.flatten()
    right_flat = right_f.flatten()
    denom = left_flat.norm() * right_flat.norm()
    cosine = float((left_flat @ right_flat / denom).item()) if float(denom.item()) != 0.0 else None
    return {
        "same_shape": True,
        "left_shape": tuple(left.shape),
        "right_shape": tuple(right.shape),
        "max": float(diff.max().item()) if diff.numel() else 0.0,
        "mean": float(diff.mean().item()) if diff.numel() else 0.0,
        "cosine": cosine,
        "allclose_1e_2": bool(torch.allclose(left_f, right_f, atol=1e-2, rtol=1e-2)),
    }


def _diff_capture_key(left: dict[str, Any], right: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in left and key not in right:
        return _missing_diff("both")
    if key not in left:
        return _missing_diff("left")
    if key not in right:
        return _missing_diff("right")
    return _diff_stats(left[key], right[key])


def _parse_layer_index(prefix: Any) -> int | None:
    if not isinstance(prefix, str):
        return None
    parts = prefix.split(".")
    for idx, part in enumerate(parts[:-1]):
        if part == "layers":
            try:
                return int(parts[idx + 1])
            except (TypeError, ValueError):
                return None
    return None


def _extract_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    return None


def _run_child_capture(mode: str, args: argparse.Namespace) -> None:
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = args.npu_device
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        mode,
        "--dense-model",
        args.dense_model,
        "--quarot-model",
        args.quarot_model,
        "--dense-quantization",
        args.dense_quantization or "",
        "--quarot-quantization",
        args.quarot_quantization or "",
        "--prompt",
        args.prompt,
        "--max-tokens",
        str(args.max_tokens),
        "--dtype",
        args.dtype,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--out-dir",
        args.out_dir,
        "--npu-device",
        args.npu_device,
        "--max-capture-layers",
        str(args.max_capture_layers),
        "--num-layers",
        str(args.num_layers),
    ]
    if args.keep_layer_dumps:
        cmd.append("--keep-layer-dumps")
    subprocess.run(cmd, check=True, env=env)


def _cleanup_layer_dumps(out_dir: Path) -> int:
    removed = 0
    for mode in _CAPTURE_MODES:
        for pattern in ("layer_*.pt", "call_*.pt", "logits.pt"):
            for path in (out_dir / mode).glob(pattern):
                path.unlink()
                removed += 1
    return removed


def _run_both(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for mode in _CAPTURE_MODES:
        _run_child_capture(mode, args)
    try:
        return _compare(args)
    finally:
        if not args.keep_layer_dumps:
            removed = _cleanup_layer_dumps(out_dir)
            print(f"removed_layer_dumps={removed}")


def _capture(args: argparse.Namespace) -> int:
    mode = args.mode
    model, quantization = _mode_model_and_quantization(mode, args)
    if mode == "quarot_native":
        os.environ["VLLM_ASCEND_QUAROT_USE_NATIVE_KV_CACHE"] = "1"
        os.environ.pop("VLLM_ASCEND_QUAROT_NATIVE_KV4", None)
        os.environ.pop("VLLM_ASCEND_QUAROT_NATIVE_KV4_ASCENDC", None)
    elif mode == "kv4":
        os.environ["VLLM_ASCEND_QUAROT_USE_NATIVE_KV_CACHE"] = "0"
        os.environ.pop("VLLM_ASCEND_QUAROT_NATIVE_KV4", None)
        os.environ.pop("VLLM_ASCEND_QUAROT_NATIVE_KV4_ASCENDC", None)
    else:
        os.environ.pop("VLLM_ASCEND_QUAROT_USE_NATIVE_KV_CACHE", None)
        os.environ.pop("VLLM_ASCEND_QUAROT_NATIVE_KV4", None)
        os.environ.pop("VLLM_ASCEND_QUAROT_NATIVE_KV4_ASCENDC", None)

    mode_dir = Path(args.out_dir) / mode
    if mode_dir.exists():
        shutil.rmtree(mode_dir)
    mode_dir.mkdir(parents=True, exist_ok=True)

    # Import only after setting mode env vars. The backend reads environment
    # switches during runtime, but late import keeps child mode unambiguous.
    from vllm import LLM, SamplingParams
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.linear import RowParallelLinear
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer, Qwen2MLP
    from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer, Qwen3MLP

    import vllm_ascend.attention.attention_v1 as attention_v1

    original_forward_impl = attention_v1.AscendAttentionBackendImpl.forward_impl
    original_row_parallel_forward = RowParallelLinear.forward
    original_rmsnorm_forward = RMSNorm.forward
    original_logits_forward = LogitsProcessor.forward
    original_qwen2_decoder_forward = Qwen2DecoderLayer.forward
    original_qwen3_decoder_forward = Qwen3DecoderLayer.forward
    original_qwen2_mlp_forward = Qwen2MLP.forward
    original_qwen3_mlp_forward = Qwen3MLP.forward
    capture_count = {"value": 0}
    payloads_by_call: dict[int, dict[str, Any]] = {}
    latest_call_by_layer: dict[int, int] = {}

    def _clone_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().to("cpu").clone()

    def _last_token_view(tensor: torch.Tensor, num_tokens: int) -> torch.Tensor:
        if tensor.numel() == 0:
            return tensor.detach()
        return tensor[:num_tokens][-1:].detach()

    def _save_payload(call_idx: int) -> None:
        payload = payloads_by_call.get(call_idx)
        if payload is not None:
            torch.save(payload, mode_dir / f"call_{call_idx:06d}.pt")

    def _latest_call_for_layer(layer_idx: int | None) -> int | None:
        if layer_idx is None:
            return None
        return latest_call_by_layer.get(layer_idx)

    def _patched_forward_impl(self, query, key, value, kv_cache, attn_metadata, output):
        capture_this = (
            attn_metadata is not None
            and args.num_layers > 0
            and (args.max_capture_layers <= 0 or capture_count["value"] < args.max_capture_layers)
        )
        call_idx = capture_count["value"]
        layer_idx = call_idx % args.num_layers
        if capture_this:
            capture_count["value"] += 1
            num_tokens = int(attn_metadata.actual_seq_lengths_q[-1])
            query_in = query[:num_tokens][-1:].detach()
            if key is None:
                key_in = torch.empty(0, dtype=query.dtype, device=query.device)
            else:
                key_in = key[:num_tokens][-1:].detach()
            if value is None:
                value_in = torch.empty(0, dtype=query.dtype, device=query.device)
            else:
                value_in = value[:num_tokens][-1:].detach()
        else:
            num_tokens = 0
            query_in = key_in = value_in = None

        result = original_forward_impl(self, query, key, value, kv_cache, attn_metadata, output)

        if capture_this:
            backend_output = result[:num_tokens][-1:]
            payload = {
                "mode": mode,
                "call_index": call_idx,
                "layer_index": layer_idx,
                "attn_state": attn_metadata.attn_state.name,
                "actual_seq_lengths_q": list(attn_metadata.actual_seq_lengths_q),
                "num_tokens": int(num_tokens),
                "uses_quarot_kv4": bool(getattr(self, "_uses_quarot_kv4_cache", lambda: False)()),
                "query": _clone_to_cpu(query_in),
                "key": _clone_to_cpu(key_in),
                "value": _clone_to_cpu(value_in),
                "backend_output": _clone_to_cpu(backend_output),
            }
            payloads_by_call[call_idx] = payload
            latest_call_by_layer[layer_idx] = call_idx
            _save_payload(call_idx)
            print(
                f"captured mode={mode} call={call_idx} layer={layer_idx} "
                f"state={attn_metadata.attn_state.name} q={tuple(query_in.shape)} "
                f"k={tuple(key_in.shape)} v={tuple(value_in.shape)} output={tuple(backend_output.shape)}",
                flush=True,
            )
        return result

    def _patched_row_parallel_forward(self, input_):
        prefix = getattr(self, "prefix", "")
        capture_o_proj = "self_attn.o_proj" in prefix or "self_attn.out_proj" in prefix
        layer_idx = _parse_layer_index(prefix) if capture_o_proj else None
        call_idx = _latest_call_for_layer(layer_idx)
        payload = payloads_by_call.get(call_idx) if call_idx is not None else None
        if payload is not None:
            num_tokens = int(payload.get("num_tokens", input_.shape[0]))
            payload["o_proj_input"] = _clone_to_cpu(_last_token_view(input_, num_tokens))
        result = original_row_parallel_forward(self, input_)
        if payload is not None:
            output_tensor = _extract_tensor(result)
            if output_tensor is not None:
                num_tokens = int(payload.get("num_tokens", output_tensor.shape[0]))
                payload["o_proj_output"] = _clone_to_cpu(_last_token_view(output_tensor, num_tokens))
                _save_payload(call_idx)
        return result

    def _patched_rmsnorm_forward(self, *args, **kwargs):
        prefix = getattr(self, "prefix", "")
        capture_post_attention = prefix.endswith("post_attention_layernorm")
        layer_idx = _parse_layer_index(prefix) if capture_post_attention else None
        call_idx = _latest_call_for_layer(layer_idx)
        result = original_rmsnorm_forward(self, *args, **kwargs)
        payload = payloads_by_call.get(call_idx) if call_idx is not None else None
        if (
            payload is not None
            and isinstance(result, tuple)
            and len(result) >= 2
            and isinstance(result[1], torch.Tensor)
        ):
            residual = result[1]
            num_tokens = int(payload.get("num_tokens", residual.shape[0]))
            payload["post_attention_residual"] = _clone_to_cpu(_last_token_view(residual, num_tokens))
            _save_payload(call_idx)
        return result

    def _patched_logits_forward(self, lm_head, hidden_states, embedding_bias=None):
        final_hidden = hidden_states[-1:].detach() if isinstance(hidden_states, torch.Tensor) else None
        logits = original_logits_forward(self, lm_head, hidden_states, embedding_bias)
        if isinstance(logits, torch.Tensor) and logits.numel() > 0:
            last_logits = logits[-1:].detach()
            top_values, top_indices = torch.topk(
                last_logits.to(torch.float32), k=min(10, last_logits.shape[-1]), dim=-1
            )
            payload = {
                "mode": mode,
                "logits": _clone_to_cpu(last_logits),
                "topk_ids": top_indices.detach().to("cpu").tolist()[0],
                "topk_values": top_values.detach().to("cpu").tolist()[0],
            }
            if final_hidden is not None:
                payload["final_hidden"] = _clone_to_cpu(final_hidden)
            torch.save(payload, mode_dir / "logits.pt")
            print(
                f"captured logits mode={mode} shape={tuple(last_logits.shape)} "
                f"top1={payload['topk_ids'][0]} value={payload['topk_values'][0]:.6g}",
                flush=True,
            )
        return logits

    def _decoder_layer_index(layer: Any) -> int | None:
        self_attn = getattr(layer, "self_attn", None)
        qkv_proj = getattr(self_attn, "qkv_proj", None)
        return _parse_layer_index(getattr(qkv_proj, "prefix", ""))

    def _mlp_layer_index(mlp: Any) -> int | None:
        gate_up_proj = getattr(mlp, "gate_up_proj", None)
        return _parse_layer_index(getattr(gate_up_proj, "prefix", ""))

    def _patched_qwen_mlp_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        u = self.act_fn(gate_up)

        layer_idx = _mlp_layer_index(self)
        call_idx = _latest_call_for_layer(layer_idx)
        payload = payloads_by_call.get(call_idx) if call_idx is not None else None
        if payload is not None and isinstance(u, torch.Tensor):
            num_tokens = int(payload.get("num_tokens", u.shape[0]))
            payload["mlp_u"] = _clone_to_cpu(_last_token_view(u, num_tokens))
            _save_payload(call_idx)

        x, _ = self.down_proj(u)
        return x

    def _patched_qwen_decoder_forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        layer_idx = _decoder_layer_index(self)
        call_idx = _latest_call_for_layer(layer_idx)
        payload = payloads_by_call.get(call_idx) if call_idx is not None else None
        if payload is not None and isinstance(residual, torch.Tensor):
            num_tokens = int(payload.get("num_tokens", residual.shape[0]))
            payload["post_attention_residual"] = _clone_to_cpu(_last_token_view(residual, num_tokens))
            _save_payload(call_idx)

        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    attention_v1.AscendAttentionBackendImpl.forward_impl = _patched_forward_impl
    RowParallelLinear.forward = _patched_row_parallel_forward
    RMSNorm.forward = _patched_rmsnorm_forward
    LogitsProcessor.forward = _patched_logits_forward
    Qwen2DecoderLayer.forward = _patched_qwen_decoder_forward
    Qwen3DecoderLayer.forward = _patched_qwen_decoder_forward
    Qwen2MLP.forward = _patched_qwen_mlp_forward
    Qwen3MLP.forward = _patched_qwen_mlp_forward

    llm = LLM(
        model=model,
        trust_remote_code=True,
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        disable_log_stats=True,
        quantization=quantization,
    )
    outputs = llm.generate([args.prompt], SamplingParams(max_tokens=args.max_tokens, temperature=0.0))
    text = outputs[0].outputs[0].text
    metadata = {
        "mode": mode,
        "model": model,
        "quantization": quantization,
        "output": repr(text),
        "captured_layers": len(list(mode_dir.glob("call_*.pt"))),
        "captured_logits": (mode_dir / "logits.pt").exists(),
    }
    Qwen2MLP.forward = original_qwen2_mlp_forward
    Qwen3MLP.forward = original_qwen3_mlp_forward
    Qwen2DecoderLayer.forward = original_qwen2_decoder_forward
    Qwen3DecoderLayer.forward = original_qwen3_decoder_forward
    LogitsProcessor.forward = original_logits_forward
    RMSNorm.forward = original_rmsnorm_forward
    RowParallelLinear.forward = original_row_parallel_forward
    attention_v1.AscendAttentionBackendImpl.forward_impl = original_forward_impl
    (mode_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (mode_dir / "output.txt").write_text(repr(text) + "\n", encoding="utf-8")
    print(f"output mode={mode}: {text!r}")
    print(f"captured_layers mode={mode}: {metadata['captured_layers']}")
    return 0


def _load_captures(mode_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(mode_dir.glob("call_*.pt"))
    if not paths:
        paths = sorted(mode_dir.glob("layer_*.pt"))
    return [torch.load(path, map_location="cpu") for path in paths]


def _load_logits(mode_dir: Path) -> dict[str, Any] | None:
    path = mode_dir / "logits.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def _select_last_by_layer(captures: list[dict[str, Any]], num_layers: int) -> list[dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    for capture in captures:
        layer_idx = int(capture.get("layer_index", len(latest) % max(num_layers, 1)))
        latest[layer_idx] = capture
    return [latest[idx] for idx in sorted(latest)]


def _load_metadata(mode_dir: Path) -> dict[str, Any]:
    metadata_path = mode_dir / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    output = (
        (mode_dir / "output.txt").read_text(encoding="utf-8").strip()
        if (mode_dir / "output.txt").exists()
        else "<missing>"
    )
    return {"output": output, "model": "<unknown>", "quantization": None, "captured_layers": 0}


def _format_diff(diff: dict[str, Any]) -> str:
    if diff.get("missing") is not None:
        return f"missing={diff['missing']}"
    if not diff["same_shape"]:
        return f"shape {diff['left_shape']} != {diff['right_shape']}"
    cosine = diff.get("cosine")
    cosine_text = "None" if cosine is None else f"{cosine:.6g}"
    return f"max={diff['max']:.6g} mean={diff['mean']:.6g} cos={cosine_text} close={diff['allclose_1e_2']}"


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"dense_model={summary['dense_model']}")
    print(f"quarot_model={summary['quarot_model']}")
    for mode in _CAPTURE_MODES:
        print(f"{mode}_output={summary[f'{mode}_output']}")
    print("captures " + " ".join(f"{mode}={summary[f'{mode}_captures']}" for mode in _CAPTURE_MODES))
    print("layers " + " ".join(f"{mode}={summary[f'{mode}_layers']}" for mode in _CAPTURE_MODES))
    for pair in summary["pairs"]:
        left = pair["left"]
        right = pair["right"]
        print(f"\n{left}_vs_{right}")
        headers = [
            "layer",
            "query",
            "key",
            "value",
            "attention_output",
            "o_proj_input",
            "o_proj_output",
            "post_attention_residual",
            "mlp_u",
        ]
        print(" | ".join(headers))
        print("-" * 290)
        for row in pair["layers"]:
            print(
                f"{row['layer']:03d} | "
                f"{_format_diff(row['query'])} | "
                f"{_format_diff(row['key'])} | "
                f"{_format_diff(row['value'])} | "
                f"{_format_diff(row['backend_output'])} | "
                f"{_format_diff(row['o_proj_input'])} | "
                f"{_format_diff(row['o_proj_output'])} | "
                f"{_format_diff(row['post_attention_residual'])} | "
                f"{_format_diff(row['mlp_u'])}"
            )
        logits = pair.get("logits")
        if logits is not None:
            final_hidden = logits.get("final_hidden")
            if final_hidden is not None:
                print(f"final_hidden | {_format_diff(final_hidden)}")
            print(f"logits | {_format_diff(logits['diff'])}")
            print(f"{left}_topk={logits['left_topk_ids']}")
            print(f"{right}_topk={logits['right_topk_ids']}")
    if summary.get("layer_count_mismatch", False):
        print("warning: capture layer counts differ")


def _compare(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    summary_path = out_dir / "summary.json"
    captures = {mode: _load_captures(out_dir / mode) for mode in _CAPTURE_MODES}

    if all(not mode_captures for mode_captures in captures.values()) and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        _print_summary(summary)
        return int(summary.get("return_code", 0))

    metadata = {mode: _load_metadata(out_dir / mode) for mode in _CAPTURE_MODES}
    logits = {mode: _load_logits(out_dir / mode) for mode in _CAPTURE_MODES}
    layers = {mode: _select_last_by_layer(mode_captures, args.num_layers) for mode, mode_captures in captures.items()}
    layer_counts = {mode: len(mode_layers) for mode, mode_layers in layers.items()}
    pair_specs = (("dense", "quarot_native"), ("dense", "kv4"), ("quarot_native", "kv4"))
    summary = {
        "dense_model": metadata["dense"].get("model", args.dense_model),
        "quarot_model": metadata["quarot_native"].get("model", args.quarot_model),
        "layer_count_mismatch": len(set(layer_counts.values())) != 1,
        "pairs": [],
    }
    for mode in _CAPTURE_MODES:
        summary[f"{mode}_quantization"] = metadata[mode].get("quantization")
        summary[f"{mode}_output"] = metadata[mode].get("output", "<missing>")
        summary[f"{mode}_captures"] = len(captures[mode])
        summary[f"{mode}_layers"] = len(layers[mode])

    for left_mode, right_mode in pair_specs:
        layer_count = min(len(layers[left_mode]), len(layers[right_mode]))
        pair = {"left": left_mode, "right": right_mode, "layers": []}
        for idx in range(layer_count):
            left = layers[left_mode][idx]
            right = layers[right_mode][idx]
            row = {"layer": idx}
            for key in _LAYER_TENSOR_KEYS:
                row[key] = _diff_capture_key(left, right, key)
            pair["layers"].append(row)
        left_logits = logits.get(left_mode)
        right_logits = logits.get(right_mode)
        if left_logits is not None and right_logits is not None:
            final_hidden_diff = None
            if "final_hidden" in left_logits and "final_hidden" in right_logits:
                final_hidden_diff = _diff_stats(left_logits["final_hidden"], right_logits["final_hidden"])
            pair["logits"] = {
                "diff": _diff_stats(left_logits["logits"], right_logits["logits"]),
                "final_hidden": final_hidden_diff,
                "left_topk_ids": left_logits.get("topk_ids", []),
                "left_topk_values": left_logits.get("topk_values", []),
                "right_topk_ids": right_logits.get("topk_ids", []),
                "right_topk_values": right_logits.get("topk_values", []),
            }
        summary["pairs"].append(pair)

    summary["return_code"] = 2 if summary["layer_count_mismatch"] else 0
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_summary(summary)
    return int(summary["return_code"])


def main() -> int:
    args = _normalize_args(_build_parser().parse_args())
    if args.mode == "both":
        return _run_both(args)
    if args.mode in _CAPTURE_MODES:
        return _capture(args)
    if args.mode == "compare":
        try:
            return _compare(args)
        finally:
            if not args.keep_layer_dumps:
                removed = _cleanup_layer_dumps(Path(args.out_dir))
                print(f"removed_layer_dumps={removed}")
    raise ValueError(f"unsupported mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
