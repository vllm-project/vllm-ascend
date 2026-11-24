# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Version-specific M3 collection bridge for the pinned MindSpeed LoRA runner."""

import argparse
import os
import sys
from functools import partial
from pathlib import Path


class ReferenceWorker:
    """Named diagnostic RPCs avoid enabling arbitrary callable serialization."""

    def m31_install(self, mappings, max_elements, directory):
        from engine.lora.observer import install_layout_hooks
        from m31_observation import install_branch_capture

        model = self.model_runner.model
        identity = install_layout_hooks(model, mappings, max_elements, directory)
        install_branch_capture(model, mappings)
        return identity

    def m31_identity(self):
        from engine.lora.driver import worker_identity

        return worker_identity(self)

    def m31_activate(self, request_id, variant):
        from engine.lora.driver import activate_context

        self.model_runner.model._m31_branches.clear()
        activate_context(self, request_id, variant)

    def m31_records(self):
        from engine.lora.driver import take_records
        from m31_observation import attach_branch_records

        model = self.model_runner.model
        return attach_branch_records(model, take_records(model))


class ModelClient:
    """Translate the reference driver's calls into this vLLM version's typed APIs."""

    def __init__(self, llm):
        self.llm = llm

    def __getattr__(self, name):
        return getattr(self.llm, name)

    def apply_model(self, function):
        if isinstance(function, partial) and function.func.__name__ == "install_layout_hooks":
            options = function.keywords
            return self.llm.collective_rpc(
                "m31_install", args=(options["mappings"], options["max_elements"], options["directory"])
            )
        if function.__name__ == "take_records":
            return self.llm.collective_rpc("m31_records")
        raise ValueError("Unsupported model diagnostic callback")

    def collective_rpc(self, method, args=(), **kwargs):
        engine = self.llm.llm_engine
        if method == "list_loras":
            return [engine.list_loras()]
        if method == "add_lora":
            return [engine.add_lora(args[0])]
        if method == "remove_lora":
            return [engine.remove_lora(args[0])]
        if isinstance(method, partial) and method.func.__name__ == "activate_context":
            return self.llm.collective_rpc(
                "m31_activate", args=(method.keywords["request_id"], method.keywords["variant"])
            )
        if callable(method) and method.__name__ == "worker_identity":
            return self.llm.collective_rpc("m31_identity")
        raise ValueError("Unsupported worker diagnostic callback")


class HFReference:
    """Preserve the disk namespace while using the installed HF multimodal model."""

    def __init__(self, model):
        self.model = model

    def get_submodule(self, name):
        prefix = "language_model.model.layers."
        if not name.startswith(prefix):
            raise ValueError("Unexpected M3 adapter target")
        return self.model.get_submodule("model.language_model.layers." + name.removeprefix(prefix))

    def __call__(self, *args, **kwargs):
        from unittest.mock import patch

        from m31_observation import bf16_rotary
        from transformers.models.minimax_m3_vl import modeling_minimax_m3_vl

        # The collector is synchronous and single-request. Restore the native
        # function after each CPU call, including failed reference execution.
        with patch.object(modeling_minimax_m3_vl, "apply_rotary_pos_emb", bf16_rotary):
            return self.model(*args, **kwargs)


def create_model(config, spec, backend):
    import torch_npu  # noqa: F401
    from vllm import LLM

    if backend != "npu":
        raise ValueError("This M3 collector was implemented for NPU")
    return ModelClient(
        LLM(
            model=config["model"],
            enable_lora=True,
            max_loras=1,
            max_cpu_loras=1,
            max_lora_rank=spec["r"],
            lora_dtype="bfloat16",
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            enforce_eager=True,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            language_model_only=True,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            skip_tokenizer_init=True,
            max_num_seqs=1,
            block_size=128,
            generation_config="vllm",
            seed=spec["seed"],
            worker_extension_cls="m31_lora_driver.ReferenceWorker",
            **config.get("vllm_options", {}),
        )
    )


def configure_hf_bf16_reference(model):
    from types import MethodType

    from m31_observation import bf16_clipped_swiglu, bf16_residual_norm
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLDenseMLP, MiniMaxM3VLExperts

    if os.environ.get("VLLM_BATCH_INVARIANT") != "1":
        raise ValueError("This CPU reference requires the frozen batch-invariant execution path")

    def dense(layer, value):
        return layer.down_proj(bf16_clipped_swiglu(layer.gate_up_proj(value), layer.swiglu_alpha, layer.swiglu_limit))

    def expert_gate(layer, value):
        return bf16_clipped_swiglu(value, layer.swiglu_alpha, layer.swiglu_limit)

    text = model.model.language_model
    for index, layer in enumerate(text.layers):
        if index:
            layer.input_layernorm.forward = MethodType(bf16_residual_norm, layer.input_layernorm)
        layer.post_attention_layernorm.forward = MethodType(bf16_residual_norm, layer.post_attention_layernorm)
        if isinstance(layer.mlp, MiniMaxM3VLDenseMLP):
            layer.mlp.forward = MethodType(dense, layer.mlp)
        elif isinstance(layer.mlp.experts, MiniMaxM3VLExperts):
            layer.mlp.experts._apply_gate = MethodType(expert_gate, layer.mlp.experts)
        else:
            raise TypeError("Unexpected M3 MLP reference implementation")
    text.norm.forward = MethodType(bf16_residual_norm, text.norm)


def load_hf_reference(config, prompt, phase):
    if phase not in {"P1", "P6", "P7", "P8"}:
        return None
    import torch
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3SparseForConditionalGeneration

    class BF16M3Reference(MiniMaxM3SparseForConditionalGeneration):
        # M3 stores its MoE router in FP32. Exact tensor names avoid promoting
        # the dense MLP gate_up_proj when loading the rest of the model in BF16.
        _keep_in_fp32_modules_strict = ["mlp.gate.weight", "mlp.gate.e_score_correction_bias"]

    model, issues = BF16M3Reference.from_pretrained(
        config["model"],
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    if any(issues.values()):
        raise ValueError(f"Independent HF checkpoint loading issues: {issues}")
    if max(prompt) >= model.config.text_config.vocab_size:
        raise ValueError("Prompt exceeds the M3 vocabulary")
    configure_hf_bf16_reference(model)
    return HFReference(model.cpu().eval())


def hf_bf16_measurement(model, config, spec, suite, variant, prompt, generated, logprobs):
    """Independent CPU HF BF16 model with disk-derived low-rank forward hooks."""
    import torch
    from engine.lora.observer import load_tensors

    handles = []
    try:
        if variant != "base":
            tensors = load_tensors(suite / f"probe-{variant}")
            for module in spec["modules"]:
                layer = model.get_submodule(module["module"])
                if not isinstance(layer, torch.nn.Linear) or layer.weight.dtype != torch.bfloat16:
                    raise ValueError("BF16 HF reference requires BF16 linear targets")
                a = tensors[module["a_key"]].bfloat16()
                b = tensors[module["b_key"]].bfloat16() * (spec["lora_alpha"] / spec["r"])

                def add_delta(layer, args, result, a=a, b=b):
                    return result + (args[0] @ a.T) @ b.T

                handles.append(layer.register_forward_hook(add_delta))
        with torch.inference_mode():
            steps, cache = [], None
            for step in range(len(generated)):
                tokens = prompt if step == 0 else [generated[step - 1]]
                result = model(input_ids=torch.tensor([tokens]), past_key_values=cache, use_cache=True)
                cache = result.past_key_values
                steps.append(result.logits[0, -1].float())
            logits = torch.stack(steps)
            values = torch.log_softmax(logits, dim=-1)
            actual, expected = [], []
            for step, sampled in enumerate(logprobs):
                for token, value in sorted(sampled.items()):
                    actual.append(value)
                    expected.append(float(values[step, int(token)]))
            return {"actual": actual, "reference": expected}, logits.argmax(-1).tolist() == generated
    finally:
        for handle in handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", required=True, type=Path)
    options, rest = parser.parse_known_args()
    sys.path.insert(0, str(options.engine_root))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        (str(options.engine_root), str(Path(__file__).parent), os.environ.get("PYTHONPATH", ""))
    )
    import torch
    from engine.lora import driver
    from m31_observation import layer_measurements

    torch.set_num_threads(8)
    request_path = Path(rest[rest.index("--request") + 1])
    tolerances = driver.read_json(request_path)["config"]["tolerances"]
    created = []
    diagnostics = []
    original_execute = driver.execute

    def factory(config, spec, backend):
        model = create_model(config, spec, backend)
        created.append(model)
        return model

    def execute(request):
        response = original_execute(request)
        response["measurements"]["ideal_fp64_difference"] = diagnostics
        return response

    def lifecycle(run):
        # Preserve P7's exact load/remove/reload receipt; record B switching
        # separately so the original state-machine contract stays unchanged.
        loaded, removed, reloaded = run("a"), run("base"), run("a")
        canonical = [loaded, removed, reloaded]
        switch_base, switched, after_b = run("base"), run("b"), run("a")
        additional = [switch_base, switched, after_b]
        generations = canonical + additional
        comparisons = {
            "loaded_adapter": loaded.comparison,
            "base_after_remove": removed.comparison,
            "reloaded_adapter": reloaded.comparison,
            "switch_base": switch_base.comparison,
            "switched_b": switched.comparison,
            "after_b_reloaded_a": after_b.comparison,
        }
        return driver.PhaseResult(
            {
                "comparisons": comparisons,
                "operations": [op for value in canonical for op in value.operations],
                "additional_operations": [op for value in additional for op in value.operations],
            },
            {
                "adapter_lifecycle": all(value.greedy_matches for value in generations)
                and all(driver.comparison_passes(value.comparison, tolerances) for value in additional)
            },
            generations,
        )

    driver.create_model = factory
    driver.load_hf_reference = load_hf_reference
    driver.check_adapter_lifecycle = lifecycle
    driver.layer_measurements = partial(layer_measurements, tolerances=tolerances, diagnostics=diagnostics)
    driver.hf_measurement = hf_bf16_measurement
    driver.execute = execute
    sys.argv = [sys.argv[0], *rest]
    try:
        return driver.main()
    finally:
        for model in created:
            model.llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
