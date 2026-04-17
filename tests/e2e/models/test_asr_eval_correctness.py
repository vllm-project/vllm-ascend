import io
import os
from dataclasses import dataclass

import string

import jiwer
import numpy as np
import pytest
import scipy.io.wavfile as wav_io
import soundfile as sf
import yaml
from datasets import Audio
from modelscope.msdatasets import MsDataset
import huggingface_hub.constants as hf_constants
import datasets.config as ds_config
from jinja2 import Environment, FileSystemLoader

from tests.e2e.conftest import RemoteOpenAIServer
from vllm.utils.network_utils import get_open_port

# Allow up to 10% relative deviation from the declared ground-truth WER.
# ASR results have higher variance than classification tasks, so we use a
# more generous tolerance than the 5% used in test_lm_eval_correctness.py.
RTOL = 0.03

TEST_DIR = os.path.dirname(__file__)

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


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


def build_serve_args(eval_config: dict) -> list[str]:
    """Convert the serve: section of the YAML into a vllm serve CLI args list.

    Example — serve: {tensor_parallel_size: 2, dtype: auto} becomes:
        ["--tensor-parallel-size", "2", "--dtype", "auto"]
    """
    serve_cfg = eval_config.get("serve", {})
    flag_map = {
        "tensor_parallel_size": "--tensor-parallel-size",
        "dtype": "--dtype",
        "max_model_len": "--max-model-len",
        "gpu_memory_utilization": "--gpu-memory-utilization",
        "trust_remote_code": "--trust-remote-code",
        "enforce_eager": "--enforce-eager",
        "quantization": "--quantization",
    }
    args: list[str] = []
    for key, flag in flag_map.items():
        value = serve_cfg.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                args.append(flag)
        else:
            args.extend([flag, str(value)])
    return args


def audio_to_wav_bytes(audio_array: np.ndarray, sample_rate: int) -> bytes:
    """Convert a numpy audio array to in-memory WAV bytes at the given sample rate."""
    buf = io.BytesIO()
    # Ensure int16 encoding for maximum API compatibility.
    if audio_array.dtype != np.int16:
        if np.issubdtype(audio_array.dtype, np.floating):
            audio_array = np.clip(audio_array, -1.0, 1.0)
            audio_array = (audio_array * 32767).astype(np.int16)
        else:
            audio_array = audio_array.astype(np.int16)
    wav_io.write(buf, sample_rate, audio_array)
    return buf.getvalue()


def normalize_text(text: str) -> str:
    """Normalize text for WER calculation: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = " ".join(text.split())
    return text


def transcribe_batch(client, model_name: str, audio_items: list[dict], language: str) -> list[str]:
    """Call /v1/audio/transcriptions for a list of audio items.

    Each item in audio_items must have keys: audio_array (np.ndarray), sample_rate (int).
    Returns the raw transcription strings in the same order.
    """
    hypotheses: list[str] = []
    for item in audio_items:
        wav_bytes = audio_to_wav_bytes(item["audio_array"], item["sample_rate"])
        response = client.audio.transcriptions.create(
            model=model_name,
            file=("audio.wav", wav_bytes, "audio/wav"),
            language=language,
        )
        hypotheses.append(response.text)
    return hypotheses


def generate_asr_report(
    eval_config: dict,
    report_data: dict,
    report_dir: str,
    env_config: EnvConfig,
) -> None:
    """Write a Markdown accuracy report using the same Jinja2 template as lm_eval tests."""
    env = Environment(loader=FileSystemLoader(TEST_DIR))
    template = env.get_template("report_template.md")

    serve_cfg = eval_config.get("serve", {})
    tp_size = serve_cfg.get("tensor_parallel_size", 1)
    ep_enabled = serve_cfg.get("enable_expert_parallel", False)
    enforce_eager = serve_cfg.get("enforce_eager", False)

    parallel_mode = f"TP{tp_size}"
    if ep_enabled:
        parallel_mode += " + EP"
    execution_model = "Eager" if enforce_eager else "ACLGraph"

    model_args_str = ",".join(f"{k}={v}" for k, v in serve_cfg.items())

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
        model_type=eval_config.get("model_type", "vllm-asr"),
        datasets=",".join(t["name"] for t in eval_config["tasks"]),
        apply_chat_template=False,
        fewshot_as_multiturn=False,
        limit=eval_config.get("limit", "N/A"),
        batch_size=eval_config.get("batch_size", 8),
        num_fewshot="N/A",
        rows=report_data["rows"],
        parallel_mode=parallel_mode,
        execution_model=execution_model,
    )

    report_path = os.path.join(report_dir, f"{os.path.basename(eval_config['model_name'])}.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)


def test_asr_eval_param(config_filename, tp_size, report_dir, env_config):
    """Parametrised ASR accuracy test driven by a YAML config file.

    Skips automatically when the config's model_type is not "vllm-asr".
    """
    eval_config = yaml.safe_load(config_filename.read_text(encoding="utf-8"))

    if eval_config.get("model_type", "vllm") != "vllm-asr":
        pytest.skip(f"Skipping non-ASR config (model_type={eval_config.get('model_type', 'vllm')})")

    model_name: str = eval_config["model_name"]
    language: str = eval_config.get("language", "en")
    limit: int | None = eval_config.get("limit", None)
    batch_size: int = eval_config.get("batch_size", 8)

    # Build serve args, letting --tp-size CLI flag override the YAML value.
    serve_args = build_serve_args(eval_config)
    if tp_size and tp_size != "1":
        # Drop any --tensor-parallel-size already in serve_args, then append
        # the CLI-supplied value so it takes precedence over the YAML setting.
        it = iter(serve_args)
        serve_args = [a for a in it if a != "--tensor-parallel-size" or not next(it, None)]
        serve_args += ["--tensor-parallel-size", str(tp_size)]

    print(f"\nStarting vllm serve for {model_name}")
    print(f"  serve args: {serve_args}")

    success = True
    report_data: dict[str, list[dict]] = {"rows": []}

    server_port = get_open_port()
    serve_args = serve_args + ["--port", str(server_port)]
    with RemoteOpenAIServer(model_name, serve_args, server_port=server_port, auto_port=False) as server:
        client = server.get_client()

        for task in eval_config["tasks"]:
            task_name: str = task["name"]
            dataset_name: str = task["dataset"]
            split: str = task["split"]
            dataset_config_name: str | None = task.get("dataset_config")
            audio_col: str = task.get("audio_column", "audio")
            text_col: str = task.get("text_column", "text")

            split_expr = f"{split}[:{limit}]" if limit is not None else split
            print(f"\nLoading dataset via modelscope: {dataset_name} / {dataset_config_name} ({split_expr})")
            # MsDataset accesses ModelScope (not HuggingFace), but datasets/huggingface_hub
            # cache HF_HUB_OFFLINE at import time and block all HTTP. Patch temporarily.
            _orig_hf_offline = hf_constants.HF_HUB_OFFLINE
            _orig_ds_offline = ds_config.HF_DATASETS_OFFLINE
            hf_constants.HF_HUB_OFFLINE = False
            ds_config.HF_DATASETS_OFFLINE = False
            try:
                ds = MsDataset.load(
                    dataset_name,
                    subset_name=dataset_config_name,
                    split=split_expr,
                )
            finally:
                hf_constants.HF_HUB_OFFLINE = _orig_hf_offline
                ds_config.HF_DATASETS_OFFLINE = _orig_ds_offline
            if limit is not None:
                ds = ds.select(range(min(limit, len(ds))))

            # Disable automatic audio decoding so we can use soundfile instead
            # of torchcodec (which requires CUDA libs unavailable on Ascend NPU).
            if hasattr(ds, "cast_column"):
                ds = ds.cast_column(audio_col, Audio(decode=False))

            print(f"  {len(ds)} samples to evaluate")

            # Collect audio items and references in batches.
            all_hypotheses: list[str] = []
            all_references: list[str] = []

            for batch_start in range(0, len(ds), batch_size):
                batch = ds.select(range(batch_start, min(batch_start + batch_size, len(ds))))
                audio_items = []
                for sample in batch:
                    raw = sample[audio_col]
                    if isinstance(raw, dict) and "bytes" in raw and raw["bytes"] is not None:
                        audio_array, sample_rate = sf.read(io.BytesIO(raw["bytes"]))
                    elif isinstance(raw, dict) and "path" in raw and raw["path"] is not None:
                        audio_array, sample_rate = sf.read(raw["path"])
                    else:
                        # Already decoded (e.g. MsDataset with native decoding)
                        audio_array = raw["array"]
                        sample_rate = raw["sampling_rate"]
                    audio_items.append({"audio_array": audio_array, "sample_rate": sample_rate})
                references = [sample[text_col] for sample in batch]

                hypotheses = transcribe_batch(client, model_name, audio_items, language)
                all_hypotheses.extend(hypotheses)
                all_references.extend(references)

                if (batch_start // batch_size + 1) % 5 == 0:
                    print(f"  processed {batch_start + len(batch)}/{len(ds)} samples …")

            # Normalise both sides before WER calculation.
            norm_hypotheses = [normalize_text(h) for h in all_hypotheses]
            norm_references = [normalize_text(r) for r in all_references]

            measured_wer = round(jiwer.wer(norm_references, norm_hypotheses), 4)
            print(f"\n{task_name} WER = {measured_wer:.4f}")

            for metric in task["metrics"]:
                if metric["name"] != "wer":
                    continue
                ground_truth = metric["value"]
                # Pass if measured WER is at or below the threshold (better is OK);
                # allow up to RTOL relative degradation above the threshold.
                task_success = measured_wer <= ground_truth * (1 + RTOL)
                success = success and task_success

                status = "✅" if task_success else "❌"
                print(
                    f"{task_name} | wer: ground_truth={ground_truth} | "
                    f"measured={measured_wer} | {status}"
                )

                report_data["rows"].append(
                    {
                        "task": task_name,
                        "metric": "wer",
                        "value": f"{status}{measured_wer}",
                        "stderr": "N/A",
                    }
                )

    generate_asr_report(eval_config, report_data, report_dir, env_config)
    assert success, "One or more ASR tasks exceeded the WER tolerance. See output above."
