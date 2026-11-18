from pathlib import Path
from unittest.mock import patch

import pytest

from tools import aisbench


@pytest.mark.parametrize(
    ("gates", "score", "error"),
    [
        ({"baseline": None}, 89.5, None),
        ({"baseline": None}, float("nan"), ValueError),
        ({}, 89.5, AssertionError),
        ({"baseline": 90, "threshold": 1}, 89.5, None),
        ({"baseline": 92, "threshold": 1}, 89.5, AssertionError),
    ],
)
def test_accuracy_calibration(tmp_path, gates, score, error):
    csv = tmp_path / "accuracy.csv"
    csv.write_text(f"dataset,accuracy\ngpqa,{score}\n", encoding="utf-8")
    config = {
        "case_type": "accuracy",
        "dataset_path_local": str(tmp_path),
        "model_path": str(tmp_path),
        "dataset_conf": "gpqa/gpqa_gen_0_shot_cot_chat_prompt",
        "request_conf": "vllm_api_general_chat",
        "batch_size": 32,
        "max_out_len": 32768,
        **gates,
    }

    def result_ready(runner):
        runner.result_line = f"write csv to {csv}"

    with (
        patch.object(aisbench.AisbenchRunner, "_init_dataset_conf"),
        patch.object(aisbench.AisbenchRunner, "_init_request_conf"),
        patch.object(aisbench.AisbenchRunner, "_run_aisbench_task"),
        patch.object(aisbench.AisbenchRunner, "_wait_for_task", result_ready),
    ):
        if error:
            with pytest.raises(error):
                aisbench.AisbenchRunner("model", 8080, config)
        else:
            runner = aisbench.AisbenchRunner("model", 8080, config)
            assert runner.result == score
            assert runner.num_prompts is None


@pytest.mark.parametrize("reasoning_effort", [None, "low"])
def test_request_config_reasoning_effort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reasoning_effort: str | None):
    request_conf = tmp_path / "vllm_api_general_chat.py"
    request_conf.write_text(
        "model='test',\n"
        "host_port=8000,\n"
        "host_ip='localhost',\n"
        "max_out_len=1024,\n"
        "batch_size=1,\n"
        "trust_remote_code=True,\n"
        "generation_kwargs=dict(\n"
        "    temperature=0,\n"
        "    ignore_eos=False,\n"
        "),\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(aisbench, "REQUEST_CONF_DIR", str(tmp_path))
    runner = aisbench.AisbenchRunner.__new__(aisbench.AisbenchRunner)
    runner.__dict__.update(
        model="test-model",
        port=8001,
        host_ip="localhost",
        max_out_len=65536,
        batch_size=32,
        trust_remote_code=True,
        request_conf="vllm_api_general_chat",
        top_p=None,
        top_k=None,
        seed=None,
        min_p=None,
        presence_penalty=None,
        repetition_penalty=None,
        thinking=True,
        reasoning_effort=reasoning_effort,
        task_type="accuracy",
        temperature=None,
        no_pred=False,
        pd_cache_client=None,
    )

    runner._init_request_conf()

    content = (tmp_path / "vllm_api_general_chat_custom.py").read_text(encoding="utf-8")
    assert 'chat_template_kwargs={"thinking": True}' in content
    if reasoning_effort is None:
        assert "reasoning_effort=" not in content
    else:
        assert 'reasoning_effort="low"' in content
