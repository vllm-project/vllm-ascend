# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E correctness tests for text-to-speech models on Ascend NPU.

Driven by YAML configs under ``tests/e2e/models/configs/``. The config must set
``model_type: vllm-tts`` (not ``vllm``). Synthesis uses the upstream
``qwen_tts`` Python API (``Qwen3TTSModel.generate_voice_design``), because TTS
checkpoints such as ``qwen3_tts`` are not loaded via vLLM ``lm_eval``.

Example::

    pytest -sv tests/e2e/models/test_tts_eval_correctness.py \\
        --config tests/e2e/models/configs/Qwen3-TTS-12Hz-1.7B-VoiceDesign.yaml
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import yaml
from jinja2 import Environment, FileSystemLoader

# Allow up to 10% relative slack on RTF thresholds (TTS timing varies on NPU).
RTOL = 0.10

TEST_DIR = os.path.dirname(__file__)

DEFAULT_TTS_CASES: list[dict[str, str]] = [
    {
        "name": "chinese_synthesis",
        "text": "人工智能正在深刻改变我们的生活方式。",
        "language": "Chinese",
        "instruct": "温暖友好的女声，正常语速",
    },
    {
        "name": "english_synthesis",
        "text": "Artificial intelligence is transforming our daily lives.",
        "language": "English",
        "instruct": "Professional male voice",
    },
]


@dataclass
class EnvConfig:
    vllm_version: str
    vllm_commit: str
    vllm_ascend_version: str
    vllm_ascend_commit: str
    cann_version: str
    torch_version: str
    torch_npu_version: str


@pytest.fixture
def env_config() -> EnvConfig:
    return EnvConfig(
        vllm_version=os.getenv("VLLM_VERSION", "unknown"),
        vllm_commit=os.getenv("VLLM_COMMIT", "unknown"),
        vllm_ascend_version=os.getenv("VLLM_ASCEND_VERSION", "unknown"),
        vllm_ascend_commit=os.getenv("VLLM_ASCEND_COMMIT", "unknown"),
        cann_version=os.getenv("CANN_VERSION", "unknown"),
        torch_version=os.getenv("TORCH_VERSION", "unknown"),
        torch_npu_version=os.getenv("TORCH_NPU_VERSION", "unknown"),
    )


def _audio_duration_seconds(wav: np.ndarray, sample_rate: int) -> float:
    if wav.ndim == 1:
        num_samples = wav.shape[0]
    else:
        num_samples = wav.shape[0]
    return num_samples / float(sample_rate)


def _resolve_metric_thresholds(task: dict[str, Any]) -> dict[str, float]:
    thresholds: dict[str, float] = {
        "audio_generation_success": 1.0,
        "rtf_average": 2.0,
        "sample_rate": 24000,
    }
    for metric in task.get("metrics", []):
        thresholds[metric["name"]] = float(metric["value"])
    return thresholds


def _resolve_tts_cases(task: dict[str, Any]) -> list[dict[str, str]]:
    cases = task.get("cases")
    if cases:
        return cases
    return DEFAULT_TTS_CASES


def _patch_qwen_tts_transformers_compat() -> None:
    """Make qwen-tts 0.1.1 importable with transformers 5.x.

    Upstream qwen-tts uses ``@check_model_inputs()`` (transformers 4.x style), but
    transformers 5.x defines ``check_model_inputs`` as a plain decorator that
    requires the wrapped function as its first argument.
    """
    import functools

    try:
        from transformers.utils import generic as generic_utils
    except ImportError:
        return

    if getattr(generic_utils, "_qwen_tts_check_model_inputs_patched", False):
        return

    original = generic_utils.check_model_inputs

    @functools.wraps(original)
    def check_model_inputs(func=None, *args, **kwargs):
        if func is not None:
            return original(func, *args, **kwargs)

        def decorator(f):
            return original(f, *args, **kwargs)

        return decorator

    generic_utils.check_model_inputs = check_model_inputs
    generic_utils._qwen_tts_check_model_inputs_patched = True


def _patch_qwen_tts_rope_compat() -> None:
    """Restore ``default`` RoPE init removed from transformers 5.x ROPE_INIT_FUNCTIONS."""
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    except ImportError:
        return

    if "default" in ROPE_INIT_FUNCTIONS:
        return

    import torch

    def _compute_default_rope_parameters(
        config: Any = None,
        device: Any = None,
        seq_len: Any = None,
        layer_type: Any = None,
        **kwargs: Any,
    ) -> tuple[Any, float]:
        base = config.rope_theta
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


def _patch_qwen_tts_config_compat() -> None:
    """Map ``pad_token_id`` for qwen-tts sub-configs under transformers 5.x."""
    try:
        from qwen_tts.core.models.configuration_qwen3_tts import (  # type: ignore[import-not-found]
            Qwen3TTSTalkerCodePredictorConfig,
            Qwen3TTSTalkerConfig,
        )
    except ImportError:
        return

    def _patch_pad_token_alias(config_cls: Any, fallback_attr: str | None) -> None:
        if getattr(config_cls, "_qwen_tts_pad_token_patched", False):
            return

        original_init = config_cls.__init__

        def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            if not hasattr(self, "pad_token_id"):
                pad_token_id = 0
                if fallback_attr is not None and hasattr(self, fallback_attr):
                    pad_token_id = getattr(self, fallback_attr)
                self.pad_token_id = pad_token_id

        config_cls.__init__ = patched_init  # type: ignore[method-assign]
        config_cls._qwen_tts_pad_token_patched = True  # type: ignore[attr-defined]

    _patch_pad_token_alias(Qwen3TTSTalkerConfig, "codec_pad_id")
    _patch_pad_token_alias(Qwen3TTSTalkerCodePredictorConfig, None)


def generate_tts_report(
    eval_config: dict,
    report_data: dict,
    report_dir: str,
    env_config: EnvConfig,
) -> None:
    """Write a Markdown accuracy report using the shared Jinja2 template."""
    env = Environment(loader=FileSystemLoader(TEST_DIR))
    template = env.get_template("report_template.md")

    serve_cfg = eval_config.get("serve", {})
    tp_size = serve_cfg.get("tensor_parallel_size", 1)
    device_map = eval_config.get("device_map", serve_cfg.get("device_map", "npu:0"))
    dtype = eval_config.get("dtype", serve_cfg.get("dtype", "float16"))
    model_args_str = f"device_map={device_map},dtype={dtype},tp={tp_size}"

    report_content = template.render(
        vllm_version=env_config.vllm_version,
        vllm_commit=env_config.vllm_commit,
        vllm_ascend_version=env_config.vllm_ascend_version,
        vllm_ascend_commit=env_config.vllm_ascend_commit,
        cann_version=env_config.cann_version,
        torch_version=env_config.torch_version,
        torch_npu_version=env_config.torch_npu_version,
        hardware=eval_config.get("hardware", "unknown"),
        model_name=eval_config["model_name"],
        model_args=f"'{model_args_str}'",
        model_type=eval_config.get("model_type", "vllm-tts"),
        datasets=",".join(t["name"] for t in eval_config["tasks"]),
        apply_chat_template=False,
        fewshot_as_multiturn=False,
        limit=eval_config.get("limit", "N/A"),
        batch_size=eval_config.get("batch_size", 1),
        num_fewshot="N/A",
        rows=report_data["rows"],
        parallel_mode=f"TP{tp_size}",
        execution_model="qwen_tts API",
        show_command=False,
    )

    report_path = os.path.join(
        report_dir,
        f"{os.path.basename(eval_config['model_name'])}.md",
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)


def test_tts_eval_param(config_filename, tp_size, report_dir, env_config):
    """Parametrised TTS correctness test driven by a YAML config file."""
    eval_config = yaml.safe_load(config_filename.read_text(encoding="utf-8"))

    if eval_config.get("model_type", "vllm") != "vllm-tts":
        pytest.skip(f"Skipping non-TTS config (model_type={eval_config.get('model_type', 'vllm')})")

    success = True
    report_data: dict[str, list[dict]] = {"rows": []}
    try:
        import torch

        _patch_qwen_tts_transformers_compat()
        _patch_qwen_tts_rope_compat()
        from qwen_tts import Qwen3TTSModel  # type: ignore[import-not-found]

        _patch_qwen_tts_config_compat()

        model_name: str = eval_config["model_name"]
        serve_cfg = eval_config.get("serve", {})
        device_map = eval_config.get("device_map", serve_cfg.get("device_map", "npu:0"))
        dtype_name = str(eval_config.get("dtype", serve_cfg.get("dtype", "float16")))
        dtype = getattr(torch, dtype_name, torch.float16)

        if "ASCEND_RT_VISIBLE_DEVICES" not in os.environ and device_map.startswith("npu:"):
            device_id = device_map.split(":", 1)[-1]
            os.environ["ASCEND_RT_VISIBLE_DEVICES"] = device_id

        attn_implementation = serve_cfg.get("attn_implementation", "eager")
        print(f"\nLoading TTS model: {model_name}")
        print(f"  device_map={device_map}, dtype={dtype}, attn_implementation={attn_implementation}")

        model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device_map,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )

        for task in eval_config["tasks"]:
            rtf_values: list[float] = []
            task_name: str = task["name"]
            thresholds = _resolve_metric_thresholds(task)
            cases = _resolve_tts_cases(task)

            print(f"\nTask: {task_name} ({len(cases)} case(s))")

            for case in cases:
                case_name = case.get("name", case["text"][:32])
                text = case["text"]
                language = case["language"]
                instruct = case.get("instruct", "")

                print(f"  Running case: {case_name}")

                start = time.perf_counter()
                wavs, sample_rate = model.generate_voice_design(
                    text=text,
                    language=language,
                    instruct=instruct,
                )
                elapsed = time.perf_counter() - start

                assert len(wavs) > 0, f"No audio returned for case {case_name}"
                wav = np.asarray(wavs[0], dtype=np.float32)
                duration = _audio_duration_seconds(wav, sample_rate)
                rtf = elapsed / duration if duration > 0 else float("inf")
                rtf_values.append(rtf)

                gen_ok = 1.0
                sr_ok = sample_rate == int(thresholds["sample_rate"])
                rtf_ok = rtf <= thresholds["rtf_average"] * (1 + RTOL)

                case_success = gen_ok == thresholds["audio_generation_success"] and sr_ok and rtf_ok
                success = success and case_success

                status = "✅" if case_success else "❌"
                print(
                    f"  {case_name} | sr={sample_rate} (expect {int(thresholds['sample_rate'])}) | "
                    f"rtf={rtf:.3f} (limit {thresholds['rtf_average']}) | {status}"
                )

                report_data["rows"].append(
                    {
                        "task": f"{task_name}/{case_name}",
                        "metric": "sample_rate",
                        "value": f"{'✅' if sr_ok else '❌'}{sample_rate}",
                        "stderr": 0.0,
                    }
                )
                report_data["rows"].append(
                    {
                        "task": f"{task_name}/{case_name}",
                        "metric": "rtf",
                        "value": f"{'✅' if rtf_ok else '❌'}{round(rtf, 4)}",
                        "stderr": 0.0,
                    }
                )

            if rtf_values:
                mean_rtf = round(float(np.mean(rtf_values)), 4)
                mean_rtf_ok = mean_rtf <= thresholds["rtf_average"] * (1 + RTOL)
                success = success and mean_rtf_ok
                status = "✅" if mean_rtf_ok else "❌"
                print(f"{task_name} | rtf_average: limit={thresholds['rtf_average']} | measured={mean_rtf} | {status}")
                report_data["rows"].append(
                    {
                        "task": task_name,
                        "metric": "rtf_average",
                        "value": f"{status}{mean_rtf}",
                        "stderr": 0.0,
                    }
                )
    except Exception as exc:
        success = False
        if not report_data["rows"]:
            report_data["rows"].append(
                {
                    "task": "tts_eval",
                    "metric": "status",
                    "value": f"❌{type(exc).__name__}",
                    "stderr": 0.0,
                }
            )
        raise
    finally:
        generate_tts_report(eval_config, report_data, report_dir, env_config)

    assert success, "One or more TTS tasks failed. See output above."
