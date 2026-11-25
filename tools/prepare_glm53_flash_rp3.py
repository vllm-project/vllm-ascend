"""Prepare reproducible RP3-shaped Flash workloads without private dataset paths.

The seeds and workload sizes follow the RP3 A3 release table. These generated
GSM8K-derived prompts are not byte-identical copies of the private RP3 datasets.
Only the tokenizer is loaded; no model weights or NPU execution is needed.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import regex as re

INPUT_TOKENS = 16384
SOURCE_DATASET = "vllm-ascend/GSM8K_prefix100_in16384_bs500_GLM5_3_flash_w8a8"
MODEL = "Eco-Tech/GLM-5.3-Flash-w8a8"
OUTPUT_ROOT = Path("/tmp/glm53_flash_rp3")


def make_prompt(tokenizer, words: list[str], rng: random.Random, input_tokens: int) -> str:
    """Shuffle source words and truncate at a round-trip-safe token boundary."""
    text = " ".join(rng.choices(words, k=input_tokens * 2))
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < input_tokens:
        raise ValueError("Generated source is shorter than the requested input")
    prompt = tokenizer.decode(tokens[:input_tokens], skip_special_tokens=False)
    if len(tokenizer.encode(prompt, add_special_tokens=False)) != input_tokens:
        raise ValueError("Tokenizer does not preserve the selected token boundary")
    return prompt


def write_dataset(path: Path, prompts: list[str]) -> dict:
    path.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps({"question": prompt, "answer": "0"}) + "\n" for prompt in prompts)
    (path / "test.jsonl").write_text(payload, encoding="utf-8")
    (path / "train.jsonl").write_text("", encoding="utf-8")
    return {
        "path": str(path.resolve()),
        "requests": len(prompts),
        "unique_prompts": len(set(prompts)),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }


def prepare_datasets(tokenizer, source: str, output_root: Path = OUTPUT_ROOT) -> dict:
    words = source.split()
    if not words:
        raise ValueError("Empty source prompt")
    manifest = {
        "model": MODEL,
        "source_dataset": SOURCE_DATASET,
        "source_prompt_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "input_content_tokens": INPUT_TOKENS,
    }
    for name, count, seed in (("prefill", 280, 2026101902), ("pd", 560, 2026101906)):
        rng = random.Random(seed)
        prompts = [make_prompt(tokenizer, words, rng, INPUT_TOKENS) for _ in range(count + 8)]
        if len(set(prompts)) != len(prompts):
            raise ValueError("Cold workload contains duplicate prompts")
        manifest[name] = write_dataset(output_root / name, prompts[:count]) | {"seed": seed}
        manifest[name + "_warmup"] = write_dataset(output_root / (name + "_warmup"), prompts[count:])
    identical = make_prompt(tokenizer, words, random.Random(2026101906), INPUT_TOKENS)
    # Enough entries for the largest formal/warmup run; never rely on AISBench
    # silently cycling a 500-row dataset for a 1168-request workload.
    manifest["decode"] = write_dataset(output_root / "decode", [identical] * 1168)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def prepare_request_configs(request_dir: Path) -> None:
    """Keep native AISBench scheduling/metrics, with explicit no-retry clients.

    Unique cache salts implement RP3's cold-cache P/PD semantics while keeping
    the documented service-side prefix caching enabled for the decode replay.
    The PD proxy forwards the same salt to both P and D.
    """
    source = (request_dir / "vllm_api_stream_chat.py").read_text(encoding="utf-8")
    source, changed = re.subn(r"\bretry\s*=\s*\d+", "retry=0", source)
    if changed != 1:
        raise ValueError("Unsupported AISBench request template: expected one retry setting")
    for name in ("low", "high"):
        warm_source, changed = re.subn(
            r"(generation_kwargs\s*=\s*dict\()",
            r'\1\n            cache_salt="glm53-flash-rp3-' + name + '",',
            source,
        )
        if changed != 1:
            raise ValueError("Unsupported AISBench request template: expected generation_kwargs")
        (request_dir / f"vllm_api_stream_chat_rp3_{name}.py").write_text(warm_source, encoding="utf-8")
    subclass = """
from uuid import uuid4


class ColdPrefixAPI(VLLMCustomAPIChat):
    async def get_request_body(self, *args, **kwargs):
        body = await super().get_request_body(*args, **kwargs)
        body["cache_salt"] = uuid4().hex
        return body


"""
    source, changed = re.subn(r"\btype\s*=\s*VLLMCustomAPIChat\b", "type=ColdPrefixAPI", source)
    if changed != 1 or "models =" not in source:
        raise ValueError("Unsupported AISBench request template: expected VLLMCustomAPIChat")
    source = source.replace("models =", subclass + "models =", 1)
    (request_dir / "vllm_api_stream_chat_rp3_cold.py").write_text(source, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path")
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    # CLI-only dependencies: unit tests exercise data/config preparation without
    # downloading a checkpoint or importing the inference runtime.
    from transformers import AutoTokenizer

    from tools.aisbench import REQUEST_CONF_DIR, maybe_download_from_modelscope

    model_path = args.model_path or maybe_download_from_modelscope(MODEL)
    source_path = args.source_path or Path(maybe_download_from_modelscope(SOURCE_DATASET, repo_type="dataset"))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    with (source_path / "test.jsonl").open(encoding="utf-8") as stream:
        source = json.loads(next(stream))["question"]
    manifest = prepare_datasets(tokenizer, source, args.output_root)
    prepare_request_configs(Path(REQUEST_CONF_DIR))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
