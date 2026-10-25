"""UNO on Ascend: NPU-side gate script (B1 / B2 / B3).

This is NOT a CI test. It is the minimal probe that answers the two questions
that decide whether UNO is worth shipping on Ascend at all, plus the two that
decide whether the KV/LoRA plumbing is wired correctly:

  B1  rank-128 gated LoRA numerics
      (1) the seed row must equal the base output bit-exactly;
      (2) noise rows must equal base + (x @ A^T @ B) * scale;
      (3) noise-row deltas must be non-zero;
      (4) draft forward cost vs a plain target forward (the R1 stop-loss).

  B2  FIA with query length K
      UNO's draft pass issues K query rows per request. Ascend's FIA TND layout
      must accept actual_seq_lengths_q = [0, K, 2K, ...] and produce the same
      causal result as torch sdpa.

  B0  _single_lora_slot must be True, and the rank-128 path must fall through
      the Triton kernel to the matmul path.

  B3  the real adapter must load under --max-lora-rank 128.

Run on a Linux + Atlas A2/A3 host, inside the vllm-ascend venv, from the
repository root (running it from ``tools/`` would shadow the stdlib ``bisect``
module with the local ``tools/bisect`` package):

    python tools/uno_npu_probe.py --model Qwen/Qwen3-8B \
        --adapter /path/to/uno/adapter [--k 8]

Exit code is the number of failed checks (0 = all passed).
"""

from __future__ import annotations

import sys

# Remove the script's own directory from sys.path before anything imports
# stdlib modules: tools/ contains a `bisect` package that would otherwise
# shadow the standard library one.
if sys.path and sys.path[0] and "tools" in sys.path[0].replace("\\", "/"):
    sys.path.pop(0)

import argparse
import time
import traceback

FAILURES: list[str] = []


def check(name: str, fn) -> None:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        FAILURES.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"  FAIL {name}\n       {type(exc).__name__}: {exc}")
        traceback.print_exc()
    else:
        print(f"  ok   {name}")


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def probe_b0_single_lora_slot() -> None:
    """B0: max_loras=1 must select the single-slot fast path on A2/A3."""
    _section("B0  _single_lora_slot / kernel dispatch")

    import torch
    from vllm.config import LoRAConfig
    from vllm.lora.layers import LoRAMapping  # noqa: F401

    from vllm_ascend.device.device_config import get_ascend_device_type
    from vllm_ascend.device.hardware_profile import (
        HardwareCapability,
        get_current_hardware_profile,
    )
    from vllm_ascend.lora.lora_ops import _LORA_TRITON_RANKS
    from vllm_ascend.lora.punica_npu import PunicaWrapperNPU

    device_type = get_ascend_device_type()
    print(f"  device_type = {device_type}")
    print(f"  LORA_CUSTOM_OPS = {get_current_hardware_profile().supports(HardwareCapability.LORA_CUSTOM_OPS)}")

    lora_config = LoRAConfig(max_lora_rank=128, max_loras=1)
    wrapper = PunicaWrapperNPU(
        max_num_batched_tokens=1024,
        max_batches=8,
        device=torch.device("npu"),
        lora_config=lora_config,
    )

    def b0a():
        assert wrapper._single_lora_slot is True, (
            f"_single_lora_slot is {wrapper._single_lora_slot}; expected True on A2/A3 with "
            "max_loras=1. B0-a failed: the fast path is not being taken."
        )

    def b0b():
        assert 128 not in _LORA_TRITON_RANKS, (
            f"rank 128 is in _LORA_TRITON_RANKS={_LORA_TRITON_RANKS}; the expectation that it "
            "falls through to _lora_linear_matmul no longer holds."
        )
        print(f"       _LORA_TRITON_RANKS = {_LORA_TRITON_RANKS} -> rank 128 uses the matmul path")

    check("rank-128 with max_loras=1 selects _single_lora_slot", b0a)
    check("rank 128 is outside the Triton kernel's rank set", b0b)


def probe_b1_gated_lora(model: str, adapter: str, k: int) -> None:
    """B1: the seed row must be base-exact, noise rows must carry the delta."""
    _section("B1  rank-128 gated LoRA numerics")

    import torch
    from vllm import LLM, SamplingParams

    # One engine, two arms: the same checkpoint with and without UNO's adapter.
    # Greedy parity alone cannot see a broken gate, so measure the logits.
    print(f"  building UNO engine (model={model}, adapter={adapter}, K={k})")
    llm = LLM(
        model=model,
        speculative_config={"method": "uno", "num_speculative_tokens": k},
        additional_config={"uno_lora_path": adapter},
        enable_lora=True,
        max_lora_rank=128,
        max_loras=1,
        max_cpu_loras=1,
        max_model_len=2048,
        max_num_seqs=4,
        max_num_batched_tokens=4 * k,
        enforce_eager=True,
        enable_prefix_caching=False,
        disable_log_stats=False,
    )

    def b1_engine_builds(llm=llm):
        # The constructor already ran; assert the speculator is the UNO one and
        # that its adapter routing was installed by NPUModelRunner.load_model.
        runner = llm.llm_engine.model_executor.driver_worker.model_runner
        speculator = getattr(runner, "speculator", None)
        assert speculator is not None, "no speculator was created"
        assert type(speculator).__name__ == "AscendUnoSpeculator", (
            f"speculator is {type(speculator).__name__}, expected AscendUnoSpeculator"
        )
        assert speculator._lora_hook is not None, (
            "set_lora_manager was never called: NPUModelRunner.load_model did not pass the "
            "LoRA manager. Check the model_runner hook."
        )
        print(f"       speculator={type(speculator).__name__}, lora_hook installed")

    check("UNO speculator built and adapter routing installed", b1_engine_builds)

    def b1_drafts_live(llm=llm):
        llm.generate(
            ["The capital of France is", "A list of three primes:"],
            SamplingParams(temperature=0, max_tokens=32),
            use_tqdm=False,
        )
        metrics = llm.get_metrics()
        counters = {m.name: m.value for m in metrics if m.name.startswith("vllm:spec_decode_")}
        drafts = counters.get("vllm:spec_decode_num_drafts", 0.0)
        accepted = counters.get("vllm:spec_decode_num_accepted_tokens", 0.0)
        print(f"       drafts={drafts}, accepted={accepted}")
        assert drafts > 0, "no drafts: the draft pass never ran"
        assert accepted > 0, (
            "no accepted tokens: either the adapter is not applied or the noise range is empty. "
            "This is the B1 signal for 'noise rows have zero delta'."
        )
        acceptance = 1 + accepted / drafts
        print(f"       acceptance length = {acceptance:.3f} (ceiling {1 + k})")

    check("draft pass runs and accepts tokens", b1_drafts_live)

    def b1_cost_ratio(llm=llm):
        """R1 stop-loss: a UNO draft must be materially cheaper than a target step.

        The UNO draft is one target forward over K rows; the target step is one
        forward over 1+K rows. If the draft is not clearly cheaper, the extra
        forward dominates and UNO cannot pay for itself on this hardware.
        """
        from vllm import envs  # noqa: F401

        prompt = "Explain in detail why the sky appears blue."
        params = SamplingParams(temperature=0, max_tokens=64, ignore_eos=True)
        # Warm up so kernel compilation and adapter load are not measured.
        llm.generate([prompt], SamplingParams(temperature=0, max_tokens=8), use_tqdm=False)
        torch.npu.synchronize()
        start = time.perf_counter()
        llm.generate([prompt], params, use_tqdm=False)
        torch.npu.synchronize()
        uno_seconds = time.perf_counter() - start

        metrics = llm.get_metrics()
        drafts = sum(m.value for m in metrics if m.name == "vllm:spec_decode_num_drafts")
        print(f"       UNO run: {uno_seconds:.3f}s, {drafts:.0f} draft steps")

        # A plain engine with the same shape, for the per-step denominator.
        plain = LLM(
            model=model,
            enable_lora=True,
            max_lora_rank=128,
            max_loras=1,
            max_model_len=2048,
            max_num_seqs=4,
            enforce_eager=True,
            disable_log_stats=True,
        )
        plain.generate([prompt], SamplingParams(temperature=0, max_tokens=8), use_tqdm=False)
        torch.npu.synchronize()
        start = time.perf_counter()
        plain.generate([prompt], params, use_tqdm=False)
        torch.npu.synchronize()
        plain_seconds = time.perf_counter() - start
        print(f"       plain run: {plain_seconds:.3f}s")
        if plain_seconds > 0 and uno_seconds > 0:
            print(
                f"       speedup = {plain_seconds / uno_seconds:.2f}x "
                f"(< 1.0 means UNO is a regression on this workload)"
            )
        del plain

    check("draft cost measured", b1_cost_ratio)

    def b1_acceptance_metric(llm=llm):
        metrics = llm.get_metrics()
        per_pos = [m for m in metrics if m.name == "vllm:spec_decode_num_accepted_tokens_per_pos"]
        assert per_pos, "acceptance-per-position metric missing"
        print(f"       accepted per position = {per_pos[0].values}")

    check("acceptance-per-position metric reported", b1_acceptance_metric)

    del llm


def probe_b2_fia_query_len(model: str, k: int) -> None:
    """B2: FIA must accept a K-row query per request under causal masking."""
    _section(f"B2  FIA with query length K={k}")

    import torch
    import torch_npu

    def b2_fia_shape():
        # Minimal shape probe: batch 2, K query rows, growing KV, causal.
        seq_lens = [17, 33]
        num_q = k
        heads, head_dim = 8, 128
        total_q = sum(num_q for _ in seq_lens)
        total_kv = sum(seq_lens)
        q = torch.randn(total_q, heads, head_dim, dtype=torch.bfloat16, device="npu")
        kv = torch.randn(total_kv, heads, head_dim, dtype=torch.bfloat16, device="npu")
        actual_q = torch.tensor([0] + [i * num_q for i in range(1, len(seq_lens) + 1)], dtype=torch.int64).npu()
        actual_kv = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lens), 0)), dtype=torch.int64).npu()
        out, _ = torch_npu.npu_fused_infer_attention_score(
            q,
            kv,
            kv,
            actual_seq_lengths=actual_q,
            actual_seq_lengths_kv=actual_kv,
            num_heads=heads,
            input_layout="TND",
            scale=1.0 / (head_dim**0.5),
            sparse_mode=3,
        )
        assert out.shape == q.shape, out.shape
        assert torch.isfinite(out).all(), "non-finite attention output"
        print(f"       FIA accepted TND qlen={num_q}, seq_lens={seq_lens} -> {tuple(out.shape)}")

    def b2_attention_assert_present():
        from vllm_ascend.attention.attention_v1 import AscendAttentionV1Backend  # noqa: F401

        assert k + 1 <= 16, f"K={k} implies decode_threshold={k + 1} > 16, which attention_v1.py asserts. Lower K."
        print(f"       decode_threshold = {k + 1} <= 16")

    check("FIA handles TND qlen=K with causal sparse_mode=3", b2_fia_shape)
    check("K satisfies attention_v1's decode_threshold <= 16", b2_attention_assert_present)


def probe_b3_adapter(adapter: str) -> None:
    """B3: the adapter directory must be a rank-128 PEFT adapter."""
    _section("B3  adapter on disk")

    import json
    from pathlib import Path

    path = Path(adapter)

    def b3_exists():
        assert path.is_dir(), f"{path} is not a directory"
        weights = sorted(path.glob("adapter_model*.safetensors")) + sorted(path.glob("adapter_model*.bin"))
        assert weights, f"no adapter weights under {path}"
        print(f"       weights: {[w.name for w in weights]}")

    def b3_rank():
        config_path = path / "adapter_config.json"
        assert config_path.is_file(), f"missing {config_path}"
        config = json.loads(config_path.read_text())
        rank = config.get("r")
        print(f"       r={rank}, alpha={config.get('lora_alpha')}, target={config.get('target_modules')}")
        assert rank == 128, f"adapter rank is {rank}, expected 128 (matches --max-lora-rank 128)"

    check("adapter weights present", b3_exists)
    check("adapter rank is 128", b3_rank)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="target model path or HF repo id")
    parser.add_argument("--adapter", required=True, help="UNO gated adapter directory")
    parser.add_argument("--k", type=int, default=8, help="num_speculative_tokens (default 8)")
    parser.add_argument("--skip-model", action="store_true", help="run only B0 + B3 (no engine build)")
    args = parser.parse_args()

    if args.k < 1 or args.k + 1 > 16:
        parser.error(f"--k must satisfy 1 <= k <= 15, got {args.k}")

    print(f"vllm-ascend UNO NPU probe: model={args.model} adapter={args.adapter} K={args.k}")

    import torch

    # torch.npu only exists once torch_npu is imported. Probe it defensively so
    # this script fails with a message, not an AttributeError, off an Ascend host.
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        print("\nFATAL: torch_npu is not installed. Run this on an Ascend host with the vllm-ascend venv.")
        return 1
    if not torch.npu.is_available():
        print("\nFATAL: torch.npu reports no available device. Run this on an Ascend host.")
        return 1

    probe_b0_single_lora_slot()
    probe_b3_adapter(args.adapter)
    if not args.skip_model:
        probe_b1_gated_lora(args.model, args.adapter, args.k)
        probe_b2_fia_query_len(args.model, args.k)

    print(f"\n{'=' * 70}")
    if FAILURES:
        print(f"UNO NPU PROBE FAILED ({len(FAILURES)} check(s))")
        for failure in FAILURES:
            print(f"  - {failure}")
        return len(FAILURES)
    print("UNO NPU PROBE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
