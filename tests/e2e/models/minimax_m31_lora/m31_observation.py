# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Observe the eager BF16 branch before addition, without replacing any result."""

from collections import defaultdict

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def bf16_residual_norm(layer, value):
    # Batch-invariant AddRMSNorm materializes the BF16 residual before RMSNorm;
    # the Gemma offset is also materialized before the FP32 normalization.
    normalized = layer._norm(value.float())
    return (normalized * (1 + layer.weight).float()).to(value.dtype)


def bf16_clipped_swiglu(value, alpha, limit):
    gate, up = value.float().chunk(2, dim=-1)
    gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    return ((gate * torch.sigmoid(gate * alpha)) * (up + 1)).to(value.dtype)


def bf16_rotary(query, key, cos, sin, unsqueeze_dim=1):
    cos, sin = cos.float().unsqueeze(unsqueeze_dim), sin.float().unsqueeze(unsqueeze_dim)
    width = cos.shape[-1]

    def rotate(value):
        part = value[..., :width].float()
        half = torch.cat([-part[..., width // 2 :], part[..., : width // 2]], dim=-1)
        rotated = (part * cos + half * sin).to(value.dtype)
        return torch.cat([rotated, value[..., width:]], dim=-1)

    return rotate(query), rotate(key)


class AddObservation(TorchDispatchMode):
    def __init__(self, target):
        super().__init__()
        self.target = target
        self.base = None
        self.delta = None

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        if function is torch.ops.aten.add_.Tensor and args[0].data_ptr() == self.target.data_ptr():
            if self.delta is not None or (kwargs or {}).get("alpha", 1) != 1:
                raise AssertionError("Expected one unscaled LoRA addition")
            self.base = args[0].detach().cpu().clone()
            self.delta = args[1].detach().cpu().clone()
        return function(*args, **(kwargs or {}))


def install_branch_capture(model, mappings):
    model._m31_branches = defaultdict(list)
    model._m31_active = None
    model._m31_addition = None
    layers = {m["runtime_module"]: model.get_submodule(m["runtime_module"]) for m in mappings}
    wrappers = {id(layer.punica_wrapper): layer.punica_wrapper for layer in layers.values()}
    for wrapper in wrappers.values():
        original = wrapper._lora_linear_matmul

        def matmul(y, *args, original=original, **kwargs):
            if model._m31_active is None:
                return original(y, *args, **kwargs)
            with AddObservation(y) as observation:
                result = original(y, *args, **kwargs)
            if observation.delta is None or model._m31_addition is not None:
                raise AssertionError("Eager LoRA branch was not uniquely observed")
            model._m31_addition = observation
            return result

        wrapper._lora_linear_matmul = matmul
    for name, layer in layers.items():
        original = layer._apply_lora_to_output

        def apply(x, output, original=original, layer=layer, name=name):
            if not model._openbox_context:
                return original(x, output)
            if model._m31_active is not None:
                raise AssertionError("Nested layer collection is unsupported")
            before = output.detach().cpu().clone()
            if before.dtype != torch.bfloat16:
                raise AssertionError("The M3 observation contract requires BF16 output")
            model._m31_active, model._m31_addition = name, None
            try:
                result = original(x, output)
                observation = model._m31_addition
            finally:
                model._m31_active = None
            after = result.detach().cpu()
            delta = torch.zeros_like(before)
            width = sum(layer.output_slices)
            if observation is None:
                if not layer.punica_wrapper.no_lora:
                    raise AssertionError("Unobserved LoRA execution route")
            else:
                if not torch.equal(before[..., :width].reshape_as(observation.base), observation.base):
                    raise AssertionError("Captured addition has a different base")
                delta[..., :width] = observation.delta.reshape_as(delta[..., :width])
            if not torch.equal(after, before + delta):
                raise AssertionError("Final output differs from the captured BF16 addition")
            if not torch.equal(before[..., width:], after[..., width:]):
                raise AssertionError("LoRA changed the unadapted M3 indexer output")
            model._m31_branches[name].append(
                {
                    "base_before_add": before.reshape(-1, before.shape[-1]).double().tolist(),
                    "branch_delta": delta.reshape(-1, delta.shape[-1]).double().tolist(),
                    "addition_exact": True,
                    "indexer_tail_unchanged": True,
                    "route": "no_lora" if observation is None else "eager_matmul_add",
                }
            )
            return result

        layer._apply_lora_to_output = apply


def attach_branch_records(model, records):
    indices = defaultdict(int)
    for record in records:
        index = indices[record["row_id"]]
        record["branch_observation"] = model._m31_branches[record["runtime_module"]][index]
        indices[record["row_id"]] += 1
    return records


def bf16_expected_delta(record, modules, mappings, tensors, scale):
    """Use disk factors and FP64 products with explicit BF16 storage boundaries."""
    if record["profile"]["kind"] not in {"linear", "qkv"}:
        raise ValueError("The M3 oracle supports only linear and split QKV targets")
    x = torch.tensor(record["input"], dtype=torch.float64)
    result = torch.zeros((len(x), len(record["output"][0])), dtype=torch.float64)
    by_id = {module["id"]: module for module in modules}
    for mapping in mappings:
        if mapping["runtime_module"] != record["runtime_module"]:
            continue
        module = by_id[mapping["id"]]
        if module["a_key"] not in tensors and module["b_key"] not in tensors:
            continue
        a = tensors[module["a_key"]].bfloat16().double()
        b = (tensors[module["b_key"]].bfloat16() * scale).double()
        shrunk = (x @ a.T).bfloat16().double()
        delta = (shrunk @ b.T).bfloat16().double()
        observer = mapping.get("observer", {})
        start = sum(observer.get("slice_sizes", [])[: observer.get("slice_index", 0)])
        result[:, start : start + delta.shape[1]] += delta
    return result


def difference_summary(actual, reference, tolerances):
    difference = (actual - reference).abs()
    bound = tolerances["atol"] + tolerances["rtol"] * reference.abs()
    return {
        "elements": actual.numel(),
        "max_abs_error": float(difference.max()),
        "violations": int((difference > bound).sum()),
    }


def layer_measurements(records, modules, mappings, tensors, scale, *, tolerances, diagnostics):
    from engine.lora.observer import base_output, expected_delta
    from engine.lora.observer import layer_measurements as original_measurements

    comparisons = original_measurements(records, modules, mappings, tensors, scale)
    for stage, selected in (("prefill", records[:1]), ("decode", records[1:])):
        measured = {"actual": [], "reference": []}
        output = {"actual": [], "reference": []}
        for record in selected:
            observation = record["branch_observation"]
            if not observation["addition_exact"] or not observation["indexer_tail_unchanged"]:
                raise AssertionError("Invalid branch observation")
            actual_delta = torch.tensor(observation["branch_delta"], dtype=torch.float64)
            actual_output = torch.tensor(record["output"], dtype=torch.float64)
            actual_base = torch.tensor(observation["base_before_add"], dtype=torch.float64)
            base = base_output(record)
            ideal = expected_delta(record, modules, mappings, tensors, scale)
            record["ideal_fp64_difference"] = {
                "delta": difference_summary(actual_delta, ideal, tolerances),
                "output": difference_summary(actual_output, base + ideal, tolerances),
            }
            diagnostics.append(
                {
                    "row_id": record["row_id"],
                    "variant": record["expected_adapter"],
                    "stage": stage,
                    "tokens": len(record["input"]),
                    **record["ideal_fp64_difference"],
                }
            )
            reference_delta = bf16_expected_delta(record, modules, mappings, tensors, scale)
            reference_output = (base.bfloat16() + reference_delta.bfloat16()).double()
            measured["actual"].extend(actual_delta.flatten().tolist())
            measured["reference"].extend(reference_delta.flatten().tolist())
            # The complete output uses an independent rounded XW, never observed Y.
            output["actual"].extend(actual_output.flatten().tolist())
            output["reference"].extend(reference_output.flatten().tolist())
            output["actual"].extend(actual_base.flatten().tolist())
            output["reference"].extend(base.flatten().tolist())
        comparisons[f"{stage}_delta"] = measured
        comparisons[f"{stage}_output"] = output
    p = records[0]["profile"]
    if p.get("slice_index") is not None:
        start, size = sum(p["slice_sizes"][: p["slice_index"]]), p["slice_sizes"][p["slice_index"]]
        inactive = []
        for record in records:
            delta = torch.tensor(record["branch_observation"]["branch_delta"], dtype=torch.float64)
            inactive.extend(torch.cat((delta[:, :start], delta[:, start + size :]), dim=1).flatten().tolist())
        comparisons["inactive_slice"] = {"actual": inactive, "reference": [0.0] * len(inactive)}
    return comparisons
