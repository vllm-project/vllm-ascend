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
"""Per-layer cosine similarity tool for quantization accuracy triage.

Runs the same forward pass through two model instances — typically a BF16
baseline and its quantized sibling — captures the outputs of matching
modules with forward hooks, and reports the cosine similarity of each pair.
A layer whose cosine collapses pinpoints where a quantization recipe loses
accuracy, which end-task metrics cannot.

The tool is device-agnostic: it runs on CPU with small models and on NPU
with real ones, and never imports ``torch_npu`` itself.
"""

from collections.abc import Callable, Mapping
from typing import Any

import torch

ModuleFilter = Callable[[str, torch.nn.Module], bool]


def default_module_filter(name: str, module: torch.nn.Module) -> bool:
    """Capture ``nn.Linear`` modules (the layers weight quantization touches)."""
    return isinstance(module, torch.nn.Linear)


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    """Cosine of two flattened float32 tensors; 1.0 when both are (near) zero."""
    a_flat = a.detach().to(torch.float32).reshape(-1)
    b_flat = b.detach().to(torch.float32).reshape(-1)
    if a_flat.numel() != b_flat.numel():
        raise ValueError(
            f"Captured outputs have different element counts ({a_flat.numel()} vs {b_flat.numel()}); "
            "the two models are not running the same modules on the same inputs."
        )
    dot = torch.dot(a_flat, b_flat)
    norm = a_flat.norm() * b_flat.norm()
    if norm.item() < eps:
        # Both outputs are numerically zero: directionally identical.
        return 1.0
    return (dot / norm).item()


class _OutputCaptor:
    """Forward hook that stores the last output tensor of each module."""

    def __init__(self) -> None:
        self.captured: dict[str, torch.Tensor] = {}

    def __call__(self, module: torch.nn.Module, args: Any, output: torch.Tensor) -> None:
        # The module's fully-qualified name was stashed at registration.
        self.captured[module._cosine_capture_name] = output  # type: ignore[attr-defined]


def _capture_outputs(
    model: torch.nn.Module,
    inputs: Any,
    module_filter: ModuleFilter,
) -> dict[str, torch.Tensor]:
    """Run ``model`` once on ``inputs`` and return the captured outputs.

    Hooks are installed only on the modules passing ``module_filter`` and are
    always removed afterwards, even if the forward pass raises.
    """
    captor = _OutputCaptor()
    handles = []
    try:
        for name, module in model.named_modules():
            if module_filter(name, module):
                # named_modules() guarantees unique dotted names; stash the
                # name on the module so the hook does not need a closure per
                # registration.
                module._cosine_capture_name = name  # type: ignore[attr-defined]
                handles.append(module.register_forward_hook(captor))
        with torch.inference_mode():
            if isinstance(inputs, Mapping):
                model(**inputs)
            elif isinstance(inputs, (tuple, list)):
                model(*inputs)
            else:
                model(inputs)
    finally:
        for handle in handles:
            handle.remove()
        for _, module in model.named_modules():
            if hasattr(module, "_cosine_capture_name"):
                delattr(module, "_cosine_capture_name")
    return captor.captured


def compare_layers_cosine(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    inputs: Any,
    module_filter: ModuleFilter | None = None,
) -> dict[str, float]:
    """Cosine similarity of matching module outputs from two model instances.

    Runs one forward pass per model on the same ``inputs`` with forward hooks
    on every module passing ``module_filter`` (``nn.Linear`` by default), then
    reports, per shared module name, the cosine between the two captured
    outputs. ``model_a`` is conventionally the BF16 baseline and ``model_b``
    the quantized model, so a value of 1.0 means bit-identical outputs and
    lower values localize quantization damage.

    Modules captured in only one model are ignored (the models may share
    inputs but differ in a few layers); this keeps reports usable when one
    instance keeps an embedding wrapper the other drops.

    Args:
        model_a: Baseline model instance.
        model_b: Comparison (e.g. quantized) model instance.
        inputs: Forward-pass arguments. A dict is unpacked as keyword
            arguments, a tuple/list as positional arguments, anything else is
            passed as the single positional argument.
        module_filter: ``(name, module) -> bool`` predicate selecting which
            modules to capture. Defaults to ``nn.Linear`` layers.

    Returns:
        Mapping of module name to cosine similarity in ``[-1, 1]``. Both
        models must be in eval mode and produce deterministic outputs for the
        numbers to be meaningful.
    """
    if module_filter is None:
        module_filter = default_module_filter

    model_a.eval()
    model_b.eval()

    outputs_a = _capture_outputs(model_a, inputs, module_filter)
    outputs_b = _capture_outputs(model_b, inputs, module_filter)

    shared_names = sorted(set(outputs_a) & set(outputs_b))
    return {name: _cosine_similarity(outputs_a[name], outputs_b[name]) for name in shared_names}


def render_cosine_report(
    similarities: dict[str, float],
    worst_first: bool = True,
) -> str:
    """Render a cosine-similarity mapping as a markdown table.

    Layers are sorted worst-first by default so the top of the report is the
    action item: the layer where quantization diverged most from the
    baseline.

    Args:
        similarities: Output of :func:`compare_layers_cosine`.
        worst_first: Sort ascending by similarity (worst layer first) rather
            than by name.

    Returns:
        Markdown table string with ``layer`` and ``cosine`` columns, plus a
        summary row with the minimum and mean similarity.
    """
    if not similarities:
        return "No shared captured layers to compare."

    rows = sorted(similarities.items(), key=lambda item: item[1] if worst_first else item[0])
    lines = ["| layer | cosine |", "|---|---|"]
    for name, value in rows:
        lines.append(f"| {name} | {value:.6f} |")
    values = list(similarities.values())
    summary = f"*min / mean over {len(values)} layers: {min(values):.6f} / {sum(values) / len(values):.6f}*"
    lines.append(f"| {summary} | |")
    return "\n".join(lines)
