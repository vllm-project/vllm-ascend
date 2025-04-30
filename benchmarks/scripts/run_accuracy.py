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
from multiprocessing import Queue

import lm_eval
import torch

UNIMODAL_MODEL_NAME = [
    "Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B"
]
UNIMODAL_TASK = ["ceval-valid", "mmlu", "gsm8k"]
MULTIMODAL_NAME = ["Qwen/Qwen2.5-VL-7B-Instruct"]
MULTIMODAL_TASK = ["mmmu_val"]

batch_size_dict = {"ceval-valid": 1, "mmlu": 1, "gsm8k": "auto", "mmmu_val": 1}

MODEL_RUN_INFO = {
    "Qwen/Qwen2.5-7B-Instruct":
    ("export MODEL_AEGS='{model}, max_model_len=4096,dtype=auto,tensor_parallel_size=2,gpu_memory_utilization=0.6'\n"
     "lm_eval --model vllm --modlel_args $MODEL_ARGS --tasks {datasets} \ \n"
     "--apply_chat_template --fewshot_as_multiturn --num_fewshot 5 --batch_size 1"
     ),
    "LLM-Research/Meta-Llama-3.1-8B-Instruct":
    ("export MODEL_AEGS='{model}, max_model_len=4096,dtype=auto,tensor_parallel_size=2,gpu_memory_utilization=0.6'\n"
     "lm_eval --model vllm --modlel_args $MODEL_ARGS --tasks {datasets} \ \n"
     "--apply_chat_template --fewshot_as_multiturn --num_fewshot 5 --batch_size 1"
     ),
    "Qwen/Qwen3-8B":
    ("export MODEL_AEGS='{model}, max_model_len=4096,dtype=auto,tensor_parallel_size=2,gpu_memory_utilization=0.6'\n"
     "lm_eval --model vllm --modlel_args $MODEL_ARGS --tasks {datasets} \ \n"
     "--apply_chat_template --fewshot_as_multiturn --num_fewshot 5 --batch_size 1"
     ),
    "Qwen/Qwen2.5-VL-7B-Instruct":
    ("export MODEL_AEGS='{model}, max_model_len=8192,dtype=auto,tensor_parallel_size=2,max_images=2'\n"
     "lm_eval --model vllm-vlm --modlel_args $MODEL_ARGS --tasks {datasets} \ \n"
     "--apply_chat_template --fewshot_as_multiturn  --batch_size 1"),
}


def run_accuracy_unimodal(queue, model, dataset):
    try:
        model_args = f"pretrained={model},max_model_len=4096,dtype=auto,tensor_parallel_size=2,gpu_memory_utilization=0.6"
        results = lm_eval.simple_evaluate(
            model="vllm",
            model_args=model_args,
            tasks=dataset,
            apply_chat_template=True,
            fewshot_as_multiturn=True,
            batch_size=batch_size_dict[dataset],
            num_fewshot=5,
        )
        print(f"Success: {model} on {dataset}")
        measured_value = results["results"]
        queue.put(measured_value)
    except Exception as e:
        print(f"Error in run_accuracy_unimodal: {e}")
        queue.put(e)
        sys.exit(1)
    finally:
        torch.npu.empty_cache()
        gc.collect()


def run_accuracy_multimodal(queue, model, dataset):
    try:
        model_args = f"pretrained={model},max_model_len=8192,dtype=auto,tensor_parallel_size=2,max_images=2"
        results = lm_eval.simple_evaluate(
            model="vllm-vlm",
            model_args=model_args,
            tasks=dataset,
            apply_chat_template=True,
            fewshot_as_multiturn=True,
            batch_size=batch_size_dict[dataset],
        )
        print(f"Success: {model} on {dataset}")
        measured_value = results["results"]
        queue.put(measured_value)
    except Exception as e:
        print(f"Error in run_accuracy_multimodal: {e}")
        queue.put(e)
        sys.exit(1)
    finally:
        torch.npu.empty_cache()
        gc.collect()


def generate_md(model_name, tasks_list, args, datasets):
    run_cmd = MODEL_RUN_INFO[model_name].format(model=model_name,
                                                datasets=datasets)
    model = model_name.split("/")[1]
    preamble = f"""# {model} Accuracy Test
  <div>
    <strong>vLLM version:</strong> vLLM: {args.vllm_version}, vLLM Ascend: {args.vllm_ascend_version} <br>
  </div>
  <div>
      <strong>Software Environment:</strong> CANN: {args.cann_version}, PyTorch: {args.torch_version}, torch-npu: {args.torch_npu_version} <br>
  </div>
  <div>
      <strong>Hardware Environment</strong>: Atlas A2 Series <br>
  </div>
  <div>
      <strong>Datasets</strong>: {datasets} <br>
  </div>
  <div>
      <strong>Command</strong>: 

  ```bash
  {run_cmd}
  ```
  </div>
  <div>&nbsp;</div>
  """

    header = (
        "| Task                  | Filter | n-shot | Metric   | Value   | Stderr |\n"
        "|-----------------------|-------:|-------:|----------|--------:|-------:|"
    )
    rows = []
    rows_sub = []
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
            row = (f"| {task_name:<37} "
                   f"| {flt:<6} "
                   f"| {n_shot:6} "
                   f"| {metric:<6} "
                   f"| ↑ {value:>5.4f} "
                   f"| ± {stderr:>5.4f} |")
            if not task_name.startswith("-"):
                rows.append(row)
                rows_sub.append("<details>" + "\n" + "<summary>" + task_name +
                                " details" + "</summary>" + "\n" * 2 + header)
            rows_sub.append(row)
        rows_sub.append("</details>")
    md = preamble + "\n" + header + "\n" + "\n".join(rows) + "\n" + "\n".join(
        rows_sub) + "\n"
    print(md)
    return md


def safe_md(args, accuracy, datasets):
    data = json.loads(json.dumps(accuracy))
    for model_key, tasks_list in data.items():
        md_content = generate_md(model_key, tasks_list, args, datasets)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"create Markdown file:{args.output}")


def main(args):
    accuracy = {}
    accuracy[args.model] = []
    result_queue: Queue[float] = multiprocessing.Queue()
    if args.model in UNIMODAL_MODEL_NAME:
        datasets = ",".join(UNIMODAL_TASK)
        for dataset in UNIMODAL_TASK:
            p = multiprocessing.Process(target=run_accuracy_unimodal,
                                        args=(result_queue, args.model,
                                              dataset))
            p.start()
            p.join()
            result = result_queue.get()
            print(result)
            accuracy[args.model].append(result)
    if args.model in MULTIMODAL_NAME:
        datasets = ",".join(MULTIMODAL_TASK)
        for dataset in MULTIMODAL_TASK:
            p = multiprocessing.Process(target=run_accuracy_multimodal,
                                        args=(result_queue, args.model,
                                              dataset))
            p.start()
            p.join()
            result = result_queue.get()
            print(result)
            accuracy[args.model].append(result)
    print(accuracy)
    safe_md(args, accuracy, datasets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--vllm_ascend_version", type=str, required=False)
    parser.add_argument("--torch_version", type=str, required=False)
    parser.add_argument("--torch_npu_version", type=str, required=False)
    parser.add_argument("--vllm_version", type=str, required=False)
    parser.add_argument("--cann_version", type=str, required=False)
    args = parser.parse_args()
    main(args)
