# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Record greedy completions and compare pool PD with a standalone baseline."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx


async def validate(args):
    async with httpx.AsyncClient(timeout=300) as client:
        if args.reference:
            reference = json.loads(Path(args.reference).read_text())
            cases = [entry["request"] for entry in reference]
        else:
            reference = None
            tokenized = await client.post(
                args.tokenizer_url.rstrip("/") + "/tokenize",
                json={"model": "qwen3-30b", "prompt": "Explain how a key value cache helps language model inference. "},
            )
            tokenized.raise_for_status()
            seed_tokens = tokenized.json()["tokens"]
            cases = [
                {
                    "model": "qwen3-30b",
                    "prompt": (seed_tokens * (length // len(seed_tokens) + 1))[:length],
                    "max_tokens": 16,
                    "temperature": 0,
                    "seed": 42,
                    "ignore_eos": True,
                    "return_token_ids": True,
                }
                for length in (1, 127, 128, 129, 255, 256, 257, 513, 129)
            ]
        semaphore = asyncio.Semaphore(args.concurrency)

        async def run(index, body):
            async with semaphore:
                start = time.monotonic()
                response = await client.post(args.url.rstrip("/") + "/v1/completions", json=body)
                if response.is_error:
                    raise RuntimeError(
                        f"Case {index}, {len(body['prompt'])} tokens: HTTP {response.status_code}: {response.text}"
                    )
                result = response.json()
                choice = result["choices"][0]
                if not isinstance(choice.get("token_ids"), list):
                    raise RuntimeError("Backend did not return generated token IDs")
                entry = {"request": body, "response": result, "elapsed_seconds": time.monotonic() - start}
                if reference is not None:
                    expected = reference[index]["response"]["choices"][0]
                    entry["matches_baseline"] = choice["token_ids"] == expected["token_ids"]
                print(
                    json.dumps(
                        {
                            "case": index,
                            "prompt_tokens": len(body["prompt"]),
                            "generated_tokens": choice["token_ids"],
                            "matches_baseline": entry.get("matches_baseline"),
                        }
                    ),
                    flush=True,
                )
                return entry

        results = await asyncio.gather(*(run(index, body) for index, body in enumerate(cases)))
        Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2))
        if reference is not None and not all(entry["matches_baseline"] for entry in results):
            raise SystemExit("Pool PD output differs from the standalone baseline; inspect the output file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--tokenizer-url", default="http://127.0.0.1:18079")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--concurrency", type=int, default=1)
    asyncio.run(validate(parser.parse_args()))
