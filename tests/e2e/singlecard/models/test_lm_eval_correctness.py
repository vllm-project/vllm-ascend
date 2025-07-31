import lm_eval
import numpy as np
import yaml

# TODO(yikun): fix this
RTOL = 0.02


def launch_lm_eval(eval_config, tp_size):
    trust_remote_code = eval_config.get("trust_remote_code", False)
    max_model_len = eval_config.get("max_model_len", 4096)
    model = eval_config.get("model", "vllm")
    limit = eval_config.get("limit", None)
    model_args = (f"pretrained={eval_config['model_name']},"
                  f"tensor_parallel_size={tp_size},"
                  f"enforce_eager=true,"
                  f"add_bos_token=true,"
                  f"trust_remote_code={trust_remote_code},"
                  f"max_model_len={max_model_len},")

    for s in ["max_images"]:
        val = eval_config.get(s, None)
        if val:
            model_args += f"{s}={val},"

    eval_params = {
        "model": model,
        "model_args": model_args,
        "tasks": [task["name"] for task in eval_config["tasks"]],
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "limit": limit,
        "batch_size": "auto",
    }

    for s in ["num_fewshot"]:
        val = eval_config.get(s, None)
        if val:
            eval_params[s] = val

    print(eval_params)

    results = lm_eval.simple_evaluate(**eval_params)
    return results


def test_lm_eval_correctness_param(config_filename, tp_size):
    eval_config = yaml.safe_load(config_filename.read_text(encoding="utf-8"))

    results = launch_lm_eval(eval_config, tp_size)

    success = True
    for task in eval_config["tasks"]:
        for metric in task["metrics"]:
            ground_truth = metric["value"]
            measured_value = results["results"][task["name"]][metric["name"]]
            print(f"{task['name']} | {metric['name']}: "
                  f"ground_truth={ground_truth} | measured={measured_value}")
            success = success and np.isclose(
                ground_truth, measured_value, rtol=RTOL)

    assert success
