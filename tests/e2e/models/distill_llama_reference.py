# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in real-weight reference validation for DeepSeek-R1-Distill-Llama-70B.

Run hf, graph, teacher, and check sequentially with the same model/output directory.
HF distributes 80 layers across four NPUs; vLLM uses TP=4, DP=1, PP=1.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401 - register NPU device support for the HF reference
from transformers import AutoModelForCausalLM, TokenizersBackend


def check_results(root):
    """Use the existing prefill/decode logprob tolerances on shared prefixes."""

    def load(name):
        return {row["name"]: row for row in json.loads((root / name).read_text())}

    hf = load("hf.json")
    graph = load("graph.json")
    teacher = load("teacher.json")
    prefill_atol = 0.0689  # tests/e2e/pull_request/utils.py::compare_logprobs
    decode_atol = 2 * prefill_atol
    results = []
    for name in ["short_en", "short_zh", "reasoning", "long"]:
        base, candidate, aligned = hf[name], graph[name], teacher[name]
        assert len(base["token_ids"]) == len(candidate["token_ids"]) == 32
        assert base["token_ids"][0] == candidate["token_ids"][0]
        first = str(base["token_ids"][0])
        assert abs(base["logprobs"][0][first] - candidate["logprobs"][0][first]) <= prefill_atol
        differences = []
        for i, token in enumerate(base["token_ids"]):
            difference = abs(base["logprobs"][i][str(token)] - aligned["logprobs"][i][str(token)])
            assert difference <= (prefill_atol if i == 0 else decode_atol), (name, i, difference)
            differences.append(difference)
        mismatch = next((i for i, (a, b) in enumerate(zip(base["token_ids"], candidate["token_ids"])) if a != b), None)
        if mismatch is not None:
            token = str(base["token_ids"][mismatch])
            assert token in candidate["logprobs"][mismatch]
            assert abs(base["logprobs"][mismatch][token] - candidate["logprobs"][mismatch][token]) <= decode_atol
        results.append(
            dict(
                name=name,
                exact_greedy=base["token_ids"] == candidate["token_ids"],
                first_divergence=mismatch,
                prefill_logprob_error=differences[0],
                max_decode_logprob_error=max(differences[1:]),
            )
        )
    functional = load("functional.json")
    for name, answer in [("reasoning", r"\boxed{7}"), ("long", "MAPLE-742")]:
        assert functional[name]["finish_reason"] == "stop"
        assert answer in functional[name]["text"].split("</think>")[-1]
    report = dict(
        status="passed",
        prefill_atol=prefill_atol,
        decode_atol=decode_atol,
        cases=results,
        functional_answers={"reasoning": "7", "long": "MAPLE-742"},
    )
    (root / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


def run_hf(model_path, root, tok, cases):
    start = time.time()
    results = []
    device_map = {"model.embed_tokens": "npu:0", "model.norm": "npu:3", "lm_head": "npu:3"}
    device_map.update({f"model.layers.{i}": f"npu:{i // 20}" for i in range(80)})
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=device_map, attn_implementation="eager"
    ).eval()
    print("HF_LOADED", time.time() - start, flush=True)
    for case in cases:
        ids = torch.tensor([case["prompt_token_ids"]], device="npu:0")
        mask = torch.ones_like(ids)
        cache = None
        generated = []
        scores = []
        t = time.time()
        with torch.inference_mode():
            for step in range(32):
                output = model(
                    input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True, logits_to_keep=1
                )
                cache = output.past_key_values
                lp = output.logits[0, -1].float().log_softmax(-1)
                values, indices = lp.topk(20)
                token = int(lp.argmax().item())
                generated.append(token)
                scores.append({str(int(i)): float(v) for i, v in zip(indices.cpu().tolist(), values.cpu().tolist())})
                ids = torch.tensor([[token]], device="npu:0")
                mask = torch.cat([mask, torch.ones((1, 1), device="npu:0", dtype=mask.dtype)], dim=1)
        row = dict(
            name=case["name"],
            token_ids=generated,
            logprobs=scores,
            text=tok.decode(generated),
            elapsed_s=time.time() - t,
        )
        results.append(row)
        (root / "hf.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print("HF_CASE", row["name"], row["elapsed_s"], row["text"], flush=True)


def run_vllm(backend, model_path, root, cases):
    # Keep vLLM imports out of the independent Transformers reference path.
    results = []
    from vllm import LLM, SamplingParams

    model = LLM(
        model=model_path,
        tokenizer_mode="deepseek_r1_distill_llama",
        dtype="bfloat16",
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        max_model_len=8192,
        max_num_seqs=2,
        gpu_memory_utilization=0.9,
        enforce_eager=False,
    )
    if backend == "teacher":
        hf = json.loads((root / "hf.json").read_text())
        combined = [{"prompt_token_ids": c["prompt_token_ids"] + h["token_ids"]} for c, h in zip(cases, hf)]
        teacher = model.generate(combined, SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=20))
        aligned = []
        for c, h, out in zip(cases, hf, teacher):
            offset = len(c["prompt_token_ids"])
            aligned.append(
                dict(
                    name=c["name"],
                    logprobs=[
                        {str(k): v.logprob for k, v in x.items()}
                        for x in out.prompt_logprobs[offset : offset + len(h["token_ids"])]
                    ],
                )
            )
        (root / "teacher.json").write_text(json.dumps(aligned, ensure_ascii=False, indent=2))
        print("TEACHER_DONE", flush=True)
        return
    outputs = model.generate(
        [{"prompt_token_ids": x["prompt_token_ids"]} for x in cases],
        SamplingParams(temperature=0, max_tokens=32, ignore_eos=True, logprobs=20),
    )
    for case, out in zip(cases, outputs):
        seq = out.outputs[0]
        results.append(
            dict(
                name=case["name"],
                token_ids=list(seq.token_ids),
                logprobs=[{str(k): v.logprob for k, v in x.items()} for x in seq.logprobs],
                text=seq.text,
            )
        )
    (root / (backend + ".json")).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    functional = model.generate(
        [{"prompt_token_ids": x["prompt_token_ids"]} for x in cases[2:]],
        SamplingParams(temperature=0, max_tokens=512),
    )
    answers = []
    for case, out in zip(cases[2:], functional):
        seq = out.outputs[0]
        answers.append(
            dict(name=case["name"], text=seq.text, token_ids=list(seq.token_ids), finish_reason=seq.finish_reason)
        )
    (root / "functional.json").write_text(json.dumps(answers, ensure_ascii=False, indent=2))
    print("VLLM_COMPLETE", backend, flush=True)


def main():
    p = argparse.ArgumentParser(description="Validate this checkpoint against an independent Transformers reference.")
    p.add_argument("backend", choices=["hf", "graph", "teacher", "check"])
    p.add_argument("--model", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    if args.backend == "check":
        check_results(root)
        return
    model_path = args.model
    tok = TokenizersBackend.from_pretrained(model_path)
    prompts = [
        ("short_en", "Complete the sentence with the city name: The capital of France is"),
        ("short_zh", "请用中文简短回答：一周有几天？"),
        (
            "reasoning",
            "Think step by step: A box has 3 red balls and 2 blue balls. Two more red balls are added. "
            "How many balls are in the box now? Give the final number.",
        ),
        (
            "long",
            "Read the records and answer the question at the end.\n"
            + "".join(f"Record {i}: the warehouse stores notebooks and pencils.\n" for i in range(180))
            + "\nThe secret access code is MAPLE-742.\n"
            + "".join(f"Record {i}: routine stock verification completed.\n" for i in range(180, 240))
            + "\nWhat is the secret access code? State the code exactly.",
        ),
    ]
    cases = []
    for name, text in prompts:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True, return_dict=False
        )
        cases.append(dict(name=name, prompt=text, prompt_token_ids=ids))
    (root / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    print("CASES", [(x["name"], len(x["prompt_token_ids"])) for x in cases], flush=True)
    start = time.time()
    if args.backend == "hf":
        run_hf(model_path, root, tok, cases)
    else:
        run_vllm(args.backend, model_path, root, cases)
    print("DONE", args.backend, time.time() - start, flush=True)


if __name__ == "__main__":
    main()
