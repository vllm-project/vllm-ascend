#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
#

import argparse
import gc
import json
import multiprocessing
import sys
import time
from multiprocessing import Queue

import lm_eval
import torch

# URLs for version information in Markdown report
VLLM_URL = "https://github.com/vllm-project/vllm/commit/"
VLLM_ASCEND_URL = "https://github.com/vllm-project/vllm-ascend/commit/"

# Model and task configurations
UNIMODAL_MODEL_NAME = ["Qwen/Qwen3-8B-Base", "Qwen/Qwen3-30B-A3B"]
UNIMODAL_TASK = ["ceval-valid", "gsm8k"]
MULTIMODAL_NAME = ["Qwen/Qwen2.5-VL-7B-Instruct"]
MULTIMODAL_TASK = ["mmmu_val"]

# Batch size configurations per task
BATCH_SIZE = {"ceval-valid": 1, "mmlu": 1, "gsm8k": "auto", "mmmu_val": 1}

# Model type mapping (vllm for text, vllm-vlm for vision-language)
MODEL_TYPE = {
    "Qwen/Qwen3-8B-Base": "vllm",
    "Qwen/Qwen3-30B-A3B": "vllm",
    "Qwen/Qwen2.5-VL-7B-Instruct": "vllm-vlm"
}

# Command templates for running evaluations
MODEL_RUN_INFO = {
    "Qwen/Qwen3-30B-A3B":
    ("export MODEL_ARGS='pretrained={model},max_model_len=4096,dtype=auto,tensor_parallel_size=4,gpu_memory_utilization=0.6,enable_expert_parallel=True'\n"
     "lm_eval --model vllm --model_args $MODEL_ARGS --tasks {datasets} \ \n"
     "--apply_chat_template --fewshot_as_multiturn --num_fewshot 5 --batch_size 1"
     ),
    "Qwen/Qwen3-8B-Base":
    ("export MODEL_ARGS='pretrained={model},max_model_len=4096,dtype=auto,tensor_parallel_size=2,gpu_memory_utilization=0.6'\n"
     "lm_eval --model vllm --model_args $MODEL_ARGS --tasks {datasets} \ \n"
     "--apply_chat_template --fewshot_as_multiturn --num_fewshot 5 --batch_size 1"
     ),
    "Qwen/Qwen2.5-VL-7B-Instruct":
    ("export MODEL_ARGS='pretrained={model},max_model_len=8192,dtype=auto,tensor_parallel_size=2,max_images=2'\n"
     "lm_eval --model vllm-vlm --model_args $MODEL_ARGS --tasks {datasets} \ \n"
     "--apply_chat_template --fewshot_as_multiturn  --batch_size 1"),
}

# Evaluation metric filters per task
FILTER = {
    "gsm8k": "exact_match,flexible-extract",
    "ceval-valid": "acc,none",
    "mmmu_val": "acc,none"
}

# Expected accuracy values for models
EXPECTED_VALUE = {
    "Qwen/Qwen3-30B-A3B": {
        "ceval-valid": 0.83,
        "gsm8k": 0.85
    },
    "Qwen/Qwen3-8B-Base": {
        "ceval-valid": 0.82,
        "gsm8k": 0.83
    },
    "Qwen/Qwen2.5-VL-7B-Instruct": {
        "mmmu_val": 0.51
    }
}
PARALLEL_MODE = {
    "Qwen/Qwen3-8B-Base": "TP",
    "Qwen/Qwen2.5-VL-7B-Instruct": "TP",
    "Qwen/Qwen3-30B-A3B": "EP"
}

# Execution backend configuration
EXECUTION_MODE = {
    "Qwen/Qwen3-8B-Base": "ACLGraph",
    "Qwen/Qwen2.5-VL-7B-Instruct": "ACLGraph",
    "Qwen/Qwen3-30B-A3B": "ACLGraph"
}

# Model arguments for evaluation
MODEL_ARGS = {
    "Qwen/Qwen3-8B-Base":
    "pretrained=Qwen/Qwen3-8B-Base,max_model_len=4096,dtype=auto,tensor_parallel_size=2,gpu_memory_utilization=0.6",
    "Qwen/Qwen2.5-VL-7B-Instruct":
    "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_model_len=8192,dtype=auto,tensor_parallel_size=2,max_images=2",
    "Qwen/Qwen3-30B-A3B":
    "pretrained=Qwen/Qwen3-30B-A3B,max_model_len=4096,dtype=auto,tensor_parallel_size=4,gpu_memory_utilization=0.6,enable_expert_parallel=True"
}

# Whether to apply chat template formatting
APPLY_CHAT_TEMPLATE = {
    "Qwen/Qwen3-8B-Base": True,
    "Qwen/Qwen2.5-VL-7B-Instruct": True,
    "Qwen/Qwen3-30B-A3B": False
}
# Few-shot examples handling as multi-turn dialogues.
FEWSHOT_AS_MULTITURN = {
    "Qwen/Qwen3-8B-Base": True,
    "Qwen/Qwen2.5-VL-7B-Instruct": True,
    "Qwen/Qwen3-30B-A3B": False
}

# Relative tolerance for accuracy checks
RTOL = 0.03
ACCURACY_FLAG = {}


def run_accuracy_test(queue, model, dataset):
    """Run accuracy evaluation for a model on a dataset in separate process"""
    try:
        eval_params = {
            "model": MODEL_TYPE[model],
            "model_args": MODEL_ARGS[model],
            "tasks": dataset,
            "apply_chat_template": APPLY_CHAT_TEMPLATE[model],
            "fewshot_as_multiturn": FEWSHOT_AS_MULTITURN[model],
            "batch_size": BATCH_SIZE[dataset]
        }

        if MODEL_TYPE[model] == "vllm":
            eval_params["num_fewshot"] = 5

        results = lm_eval.simple_evaluate(**eval_params)
        print(f"Success: {model} on {dataset} ")
        measured_value = results["results"]
        queue.put(measured_value)
    except Exception as e:
        print(f"Error in run_accuracy_test: {e}")
        queue.put(e)
        sys.exit(1)
    finally:
        if 'results' in locals():
            del results
        gc.collect()
        torch.npu.empty_cache()
        time.sleep(5)


def generate_md(model_name, tasks_list, args, datasets):
    """Generate Markdown report with evaluation results"""
    # Format the run command
    run_cmd = MODEL_RUN_INFO[model_name].format(model=model_name,
                                                datasets=datasets)
    model = model_name.split("/")[1]

    # Version information section
    version_info = (
        f"**vLLM Version**: vLLM: {args.vllm_version} "
        f"([{args.vllm_commit}]({VLLM_URL+args.vllm_commit})), "
        f"vLLM Ascend: {args.vllm_ascend_version} "
        f"([{args.vllm_ascend_commit}]({VLLM_ASCEND_URL+args.vllm_ascend_commit}))  "
    )

    # Report header with system info
    preamble = f"""# {model}
{version_info}
**Software Environment**: CANN: {args.cann_version}, PyTorch: {args.torch_version}, torch-npu: {args.torch_npu_version}  
**Hardware Environment**: Atlas A2 Series  
**Datasets**: {datasets}  
**vLLM Engine**: V{args.vllm_use_v1}  
**Parallel Mode**: {PARALLEL_MODE[model_name]}  
**Execution Mode**: {EXECUTION_MODE[model_name]}  
**Command**:  
```bash
{run_cmd}
```
  """

    header = (
        "| Task                  | Filter | n-shot | Metric   | Value   | Stderr |\n"
        "|-----------------------|-------:|-------:|----------|--------:|-------:|"
    )
    rows = []
    rows_sub = []
    # Process results for each task
    for task_dict in tasks_list:
        for key, stats in task_dict.items():
            alias = stats.get("alias", key)
            task_name = alias.strip()
            if "exact_match,flexible-extract" in stats:
                metric_key = "exact_match,flexible-extract"
            else:
                metric_key = None
                for k in stats:
                    if "," in k and not k.startswith("acc_stderr"):
                        metric_key = k
                        break
            if metric_key is None:
                continue
            metric, flt = metric_key.split(",", 1)

            value = stats[metric_key]
            stderr = stats.get(f"{metric}_stderr,{flt}", 0)
            if model_name in UNIMODAL_MODEL_NAME:
                n_shot = "5"
            else:
                n_shot = "0"
            flag = ACCURACY_FLAG.get(task_name, "")
            row = (f"| {task_name:<37} "
                   f"| {flt:<6} "
                   f"| {n_shot:6} "
                   f"| {metric:<6} "
                   f"| {flag}{value:>5.4f} "
                   f"| ± {stderr:>5.4f} |")
            if not task_name.startswith("-"):
                rows.append(row)
                rows_sub.append("<details>" + "\n" + "<summary>" + task_name +
                                " details" + "</summary>" + "\n" * 2 + header)
            rows_sub.append(row)
        rows_sub.append("</details>")
    # Combine all Markdown sections
    md = preamble + "\n" + header + "\n" + "\n".join(rows) + "\n" + "\n".join(
        rows_sub) + "\n"
    print(md)
    return md


def safe_md(args, accuracy, datasets):
    """
    Safely generate and save Markdown report from accuracy results.
    """
    data = json.loads(json.dumps(accuracy))
    for model_key, tasks_list in data.items():
        md_content = generate_md(model_key, tasks_list, args, datasets)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"create Markdown file:{args.output}")


def main(args):
    """Main evaluation workflow"""
    accuracy = {}
    accuracy[args.model] = []
    result_queue: Queue[float] = multiprocessing.Queue()
    if args.model in UNIMODAL_MODEL_NAME:
        datasets = UNIMODAL_TASK
    else:
        datasets = MULTIMODAL_TASK
    datasets_str = ",".join(datasets)
    # Evaluate model on each dataset
    for dataset in datasets:
        accuracy_expected = EXPECTED_VALUE[args.model][dataset]
        p = multiprocessing.Process(target=run_accuracy_test,
                                    args=(result_queue, args.model, dataset))
        p.start()
        p.join()
        if p.is_alive():
            p.terminate()
            p.join()
        gc.collect()
        torch.npu.empty_cache()
        time.sleep(10)
        result = result_queue.get()
        print(result)
        if accuracy_expected - RTOL < result[dataset][
                FILTER[dataset]] < accuracy_expected + RTOL:
            ACCURACY_FLAG[dataset] = "✅"
        else:
            ACCURACY_FLAG[dataset] = "❌"
        accuracy[args.model].append(result)
    print(accuracy)
    safe_md(args, accuracy, datasets_str)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description="Run model accuracy evaluation and generate report")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--vllm_ascend_version", type=str, required=False)
    parser.add_argument("--torch_version", type=str, required=False)
    parser.add_argument("--torch_npu_version", type=str, required=False)
    parser.add_argument("--vllm_version", type=str, required=False)
    parser.add_argument("--cann_version", type=str, required=False)
    parser.add_argument("--vllm_commit", type=str, required=False)
    parser.add_argument("--vllm_ascend_commit", type=str, required=False)
    parser.add_argument("--vllm_use_v1", type=str, required=False)
    args = parser.parse_args()
    main(args)
