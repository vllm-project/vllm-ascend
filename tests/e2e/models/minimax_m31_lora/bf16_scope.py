# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Real-model inventory and deterministic BF16 adapters for M3 language modules."""

import hashlib
import json
import math
from pathlib import Path

import torch

CORE_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "index_q_proj",
    "index_k_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "experts",
)


def describe_modules(manager):
    """Derive logical targets from the constructed model, not a layer-count guess."""
    from vllm.lora.layers.fused_moe import FusedMoEWithLoRA

    result = []
    for runtime, layer in manager.modules.items():
        if not runtime.startswith(("model.language_model.", "model.layers.")):
            continue
        base = layer.base_layer
        prefix = runtime.rsplit(".", 1)[0]
        if isinstance(layer, FusedMoEWithLoRA):
            intermediate = layer.intermediate_size_per_partition * layer.tp_size
            for expert in range(layer.global_num_experts):
                for projection, inputs, outputs in (
                    ("w1", layer.hidden_size, intermediate),
                    ("w2", intermediate, layer.hidden_size),
                    ("w3", layer.hidden_size, intermediate),
                ):
                    result.append(
                        {
                            "runtime": runtime,
                            "logical": f"{runtime}.{expert}.{projection}",
                            "kind": f"expert_{projection}",
                            "expert": expert,
                            "inputs": inputs,
                            "outputs": outputs,
                        }
                    )
        elif runtime.endswith(".qkv_proj"):
            widths = [
                base.total_num_heads * base.head_size,
                base.total_num_kv_heads * base.head_size,
                base.total_num_kv_heads * base.head_size,
            ]
            names = ["q_proj", "k_proj", "v_proj"]
            if getattr(base, "shard_ids", ()) == ("q", "k", "v", "index_q", "index_k"):
                names += ["index_q_proj", "index_k_proj"]
                widths += [base.total_num_index_heads * base.index_head_size, base.index_head_size]
            for index, (name, width) in enumerate(zip(names, widths, strict=True)):
                result.append(
                    {
                        "runtime": runtime,
                        "logical": f"{prefix}.{name}",
                        "kind": name.removesuffix("_proj"),
                        "slice": index,
                        "inputs": base.input_size,
                        "outputs": width,
                    }
                )
        elif runtime.endswith(".gate_up_proj"):
            group = "shared" if ".shared_experts." in runtime else "mlp"
            for index, name in enumerate(("gate_proj", "up_proj")):
                result.append(
                    {
                        "runtime": runtime,
                        "logical": f"{prefix}.{name}",
                        "kind": f"{group}_{name.removesuffix('_proj')}",
                        "slice": index,
                        "inputs": base.input_size,
                        "outputs": base.output_sizes[index],
                    }
                )
        elif runtime.endswith((".o_proj", ".down_proj")):
            if runtime.endswith(".o_proj"):
                kind = "o"
            else:
                kind = "shared_down" if ".shared_experts." in runtime else "mlp_down"
            result.append(
                {
                    "runtime": runtime,
                    "logical": runtime,
                    "kind": kind,
                    "slice": 0,
                    "inputs": base.input_size,
                    "outputs": base.output_size,
                }
            )
    names = [item["logical"] for item in result]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate logical M3 LoRA targets in runtime inventory")
    return result


def select_targets(inventory, profile):
    groups = {
        "attention": {"q", "k", "v", "o"},
        "indexer": {"index_q", "index_k"},
        "attention-index": {"q", "k", "v", "o", "index_q", "index_k"},
        "mlp": {"mlp_gate", "mlp_up", "mlp_down"},
        "shared": {"shared_gate", "shared_up", "shared_down"},
        "experts": {"expert_w1", "expert_w2", "expert_w3"},
    }
    kinds = groups.get(profile, {profile})
    selected = [row for row in inventory if profile == "all" or row["kind"] in kinds]
    if profile == "expert-subset":
        last = max(row["expert"] for row in inventory if "expert" in row)
        selected = [row for row in inventory if row.get("expert") in {0, 3, last}]
    if not selected:
        raise ValueError(f"The real model has no targets for profile {profile!r}")
    return selected


def disk_module(logical):
    if logical.startswith("model.layers."):
        logical = "model.language_model." + logical
    return "base_model.model." + logical.removeprefix("model.")


def build_adapter(directory, targets, *, variant, rank=8, alpha=16, amplitude=1.5):
    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    tensors = {}
    for row in targets:
        name = disk_module(row["logical"])
        seed = int.from_bytes(hashlib.sha256(f"20261124:{variant}:{name}".encode()).digest()[:8], "little")
        generator = torch.Generator().manual_seed(seed)
        a = torch.randn(rank, row["inputs"], generator=generator)
        b = torch.randn(row["outputs"], rank, generator=generator)
        a *= amplitude / math.sqrt(row["inputs"])
        b *= amplitude / math.sqrt(rank)
        if variant == "zero":
            b.zero_()
        tensors[name + ".lora_A.weight"] = a
        tensors[name + ".lora_B.weight"] = b
    save_file(tensors, str(directory / "adapter_model.safetensors"))
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": sorted({row["logical"].rsplit(".", 1)[-1] for row in targets}),
    }
    (directory / "adapter_config.json").write_text(json.dumps(config, indent=2))
    (directory / "targets.json").write_text(json.dumps(targets, indent=2))
    return directory


def effective_factors(directory):
    from safetensors.torch import load_file

    directory = Path(directory)
    config = json.loads((directory / "adapter_config.json").read_text())
    tensors = load_file(str(directory / "adapter_model.safetensors"))
    result = {}
    for row in json.loads((directory / "targets.json").read_text()):
        name = disk_module(row["logical"])
        a = tensors[name + ".lora_A.weight"].bfloat16()
        b = tensors[name + ".lora_B.weight"].bfloat16() * (config["lora_alpha"] / config["r"])
        result[row["logical"]] = (a, b)
    return result


def local_linear_factors(row, layer, factors):
    if row["logical"] not in factors:
        return None
    a, b = factors[row["logical"]]
    if row["kind"] in {"o", "mlp_down", "shared_down"}:
        start = layer.tp_rank * layer.input_size
        return a[:, start : start + layer.input_size], b
    index = row["slice"]
    width = layer.output_slices[index]
    shard = layer.tp_rank
    if row["kind"] in {"k", "v", "index_q"}:
        shard //= layer.base_layer.num_kv_head_replicas
    elif row["kind"] == "index_k":
        shard = 0
    return a, b[shard * width : (shard + 1) * width]


def comparison(actual, expected):
    actual, expected = actual.detach().cpu().double(), expected.double()
    error = (actual - expected).abs()
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    return {
        "elements": error.numel(),
        "max_abs": float(error.max()) if bool(finite.all()) and error.numel() else None,
        "max_tolerance_ratio": float((error / (1 / 64 + expected.abs() / 64)).max())
        if bool(finite.all()) and error.numel()
        else None,
        "violations": int((~finite | (error > 1 / 64 + expected.abs() / 64)).sum()),
    }


def check_linear(layer, rows, factors, slot):
    generator = torch.Generator().manual_seed(47011)
    x = torch.randn(3, layer.input_size, generator=generator).bfloat16()
    initial = (torch.randn(3, sum(layer.output_slices), generator=generator) * 0.25).bfloat16()
    expected = initial.clone()
    ideal = initial.double()
    inactive = torch.zeros_like(initial, dtype=torch.bool)
    exact, offset = True, 0
    for row in rows:
        index = row["slice"]
        width = layer.output_slices[index]
        pair = local_linear_factors(row, layer, factors)
        a_buffer = layer.lora_a_stacked[index][slot, 0].detach().cpu()
        b_buffer = layer.lora_b_stacked[index][slot, 0].detach().cpu()
        if pair is None:
            exact &= bool(a_buffer.count_nonzero() == 0 and b_buffer.count_nonzero() == 0)
            inactive[:, offset : offset + width] = True
        else:
            a, b = pair
            exact &= torch.equal(a_buffer[: a.shape[0], : a.shape[1]], a)
            exact &= torch.equal(b_buffer[: b.shape[0], : b.shape[1]], b)
            shrink_dtype = torch.bfloat16 if layer.punica_wrapper._single_lora_slot else torch.float32
            shrink = (x.double() @ a.double().T).to(shrink_dtype)
            delta = shrink.double() @ b.double().T
            if shrink_dtype == torch.bfloat16:
                delta = delta.bfloat16()
            expected[:, offset : offset + width] = (
                initial[:, offset : offset + width].double() + delta.double()
            ).bfloat16()
            ideal[:, offset : offset + width] += (x.double() @ a.double().T) @ b.double().T
        offset += width
    output = layer._apply_lora_to_output(x.to(layer.device), initial.to(layer.device))
    metric = comparison(output, expected)
    metric.update(
        loaded_exact=exact,
        inactive_exact=torch.equal(output.detach().cpu()[inactive], initial[inactive]),
        signal_margin=comparison(initial, expected)["max_tolerance_ratio"],
        ideal_fp64=comparison(output, ideal),
    )
    return metric


def check_experts(layer, runtime, factors, slot):
    generator = torch.Generator().manual_seed(47012)
    count = layer.local_num_experts
    first = layer.ep_rank * count if layer.use_ep else 0
    results = []
    for projections, aa, bb in (
        (("w1", "w3"), layer.w13_lora_a_stacked, layer.w13_lora_b_stacked),
        (("w2",), layer.w2_lora_a_stacked, layer.w2_lora_b_stacked),
    ):
        x = torch.randn(count, aa[0].shape[-1], generator=generator).bfloat16()
        initial = (torch.randn(count, sum(w.shape[-2] for w in bb), generator=generator) * 0.25).bfloat16()
        expected = initial.clone()
        ideal = initial.double()
        inactive = torch.zeros_like(initial, dtype=torch.bool)
        exact, offset = True, 0
        for projection, a_stack, b_stack in zip(projections, aa, bb, strict=True):
            width = b_stack.shape[-2]
            for local in range(count):
                pair = factors.get(f"{runtime}.{first + local}.{projection}")
                a_buffer = a_stack[slot, local].detach().cpu()
                b_buffer = b_stack[slot, local].detach().cpu()
                if pair is None:
                    exact &= bool(a_buffer.count_nonzero() == 0 and b_buffer.count_nonzero() == 0)
                    inactive[local, offset : offset + width] = True
                    continue
                a, b = pair
                if projection == "w2":
                    start = layer.tp_rank * a_buffer.shape[-1]
                    a = a[:, start : start + a_buffer.shape[-1]]
                else:
                    start = layer.tp_rank * width
                    b = b[start : start + width]
                exact &= torch.equal(a_buffer[: a.shape[0], : a.shape[1]], a)
                exact &= torch.equal(b_buffer[: b.shape[0], : b.shape[1]], b)
                shrink = (x[local].double() @ a.double().T).float()
                delta = shrink.double() @ b.double().T
                expected[local, offset : offset + width] = (
                    initial[local, offset : offset + width].double() + delta
                ).bfloat16()
                ideal[local, offset : offset + width] += (x[local].double() @ a.double().T) @ b.double().T
            offset += width
        output = initial.to(layer.device)
        layer.punica_wrapper.add_lora_fused_moe(
            y=output,
            x=x.to(layer.device),
            lora_a_stacked=aa,
            lora_b_stacked=bb,
            expert_ids=torch.arange(count, device=layer.device),
            adapter_enabled=layer.adapter_enabled,
            token_lora_mapping=torch.full((count,), slot, dtype=torch.long, device=layer.device),
        )
        metric = comparison(output, expected)
        metric.update(
            loaded_exact=exact,
            local_experts=count,
            first_expert=first,
            projections=projections,
            inactive_exact=torch.equal(output.detach().cpu()[inactive], initial[inactive]),
            signal_margin=comparison(initial, expected)["max_tolerance_ratio"],
            ideal_fp64=comparison(output, ideal),
        )
        results.append(metric)
    return results


class ScopeWorker:
    """Named worker diagnostics; model execution uses the public LoRA APIs."""

    def m31_scope_inventory(self):
        manager = self.model_runner.lora_manager._adapter_manager
        return {
            "targets": describe_modules(manager),
            "mapping": manager.packed_modules_mapping,
            "wrappers": {name: type(layer).__name__ for name, layer in manager.modules.items()},
        }

    def m31_scope_state(self):
        manager = self.model_runner.lora_manager._adapter_manager
        return {
            "manager": type(manager).__name__,
            "slots": manager.lora_index_to_id,
            "registered": sorted(manager.list_adapters()),
            "moe": [
                {
                    "module": name,
                    "dp_size": layer.base_layer.moe_config.dp_size,
                    "ep_rank": layer.ep_rank,
                    "enabled": layer.adapter_enabled.detach().cpu().tolist(),
                }
                for name, layer in manager.modules.items()
                if hasattr(layer, "adapter_enabled")
            ],
        }

    def m31_scope_check_layers(self, directory, adapter_id=1):
        from vllm.lora.layers import LoRAMapping
        from vllm.lora.layers.fused_moe import FusedMoEWithLoRA

        manager = self.model_runner.lora_manager._adapter_manager
        slot = manager.lora_index_to_id.index(adapter_id)
        manager.set_adapter_mapping(LoRAMapping([adapter_id] * 3, [adapter_id], is_prefill=True))
        factors = effective_factors(directory)
        groups = {}
        for row in describe_modules(manager):
            groups.setdefault(row["runtime"], []).append(row)
        result = {}
        with torch.inference_mode():
            for name, rows in groups.items():
                layer = manager.modules[name]
                if isinstance(layer, FusedMoEWithLoRA):
                    result[name] = check_experts(layer, name, factors, slot)
                else:
                    result[name] = [check_linear(layer, rows, factors, slot)]
        return result

    def m31_trace_start(self, kind="batch"):
        if kind not in {"batch", "indexer"}:
            raise ValueError("Unknown M3 diagnostic trace")
        if getattr(self, "_m31_trace_handles", None):
            raise ValueError("An M3 trace is already active")
        self._m31_trace_records = []
        self._m31_trace_handles = []
        manager = self.model_runner.lora_manager._adapter_manager
        if kind == "batch":
            name = next(name for name in manager.modules if name.endswith(".self_attn.qkv_proj"))

            def batch_hook(layer, args):
                rows = args[0].shape[0]
                indices = layer.punica_wrapper.token_lora_indices[:rows].detach().cpu().tolist()
                self._m31_trace_records.append(
                    {
                        "kind": "batch",
                        "rows": rows,
                        "slots": sorted(set(indices)),
                        "slot_ids": list(manager.lora_index_to_id),
                    }
                )

            self._m31_trace_handles.append(manager.modules[name].register_forward_pre_hook(batch_hook))
        else:

            def tensor_identity(value):
                raw = value.detach().cpu().contiguous()
                return {
                    "shape": list(raw.shape),
                    "sha256": hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest(),
                }

            def index_hook(layer, args, output):
                self._m31_trace_records.append(
                    {
                        "kind": "indexer",
                        "query": tensor_identity(args[0]),
                        "selection": [
                            tensor_identity(value) if isinstance(value, torch.Tensor) else None for value in output
                        ],
                    }
                )

            for name, layer in self.model_runner.model.named_modules():
                if name.endswith(".self_attn.indexer"):
                    self._m31_trace_handles.append(layer.register_forward_hook(index_hook))
            if not self._m31_trace_handles:
                raise ValueError("No real M3 indexer module found")
        return {"kind": kind, "hooks": len(self._m31_trace_handles)}

    def m31_trace_stop(self):
        for handle in self._m31_trace_handles:
            handle.remove()
        self._m31_trace_handles = []
        result = self._m31_trace_records
        self._m31_trace_records = []
        return result
