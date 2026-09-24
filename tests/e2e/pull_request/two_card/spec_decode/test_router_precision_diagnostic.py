"""Temporary sequential router A/B experiment; not a production regression gate."""

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


def _variant(source: str, arm: str) -> str:
    tree = ast.parse(source)
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendMoERunner")
    helper = next(
        node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "_compute_router_logits"
    )
    lines = source.splitlines(keepends=True)
    fp32 = arm.startswith("fp32")
    cast = (
        "        hidden_states_fp32 = "
        "router_logits if router_logits.dtype == torch.float32 else hidden_states.float()\n"
    )
    body = (
        "    def _compute_router_logits(self, hidden_states, router_logits):\n"
        "        gate = self.gate\n"
        "        assert gate is not None\n"
    )
    if arm == "fp32_original":
        body += cast
    body += '        if hasattr(gate, "weight_fp32"):\n'
    if arm != "fp32_original":
        body += "    " + cast
    body += "            return F.linear(hidden_states_fp32, gate.weight_fp32)\n"
    if arm == "bf16_direct":
        body += "        return F.linear(hidden_states, gate.weight, gate.bias)\n"
    else:
        body += "        gate_out = gate(hidden_states)\n"
        body += "        return gate_out[0] if isinstance(gate_out, tuple) else gate_out\n"
    lines[helper.lineno - 1 : helper.end_lineno] = [body]
    result = "".join(lines)
    if fp32:
        anchor = "        self.ascend_shared_experts = None\n"
        assert result.count(anchor) == 1
        result = result.replace(
            anchor,
            '        if gate is not None and not hasattr(gate, "weight_fp32"):\n'
            "            gate.precast_fp32_weight = True\n\n" + anchor,
        )
    ast.parse(result)
    return result


def test_router_precision_same_environment(tmp_path):
    root = Path(__file__).resolve().parents[5]
    module = root / "vllm_ascend/ops/fused_moe/fused_moe.py"
    original = module.read_bytes()
    source = original.decode()
    assert "gate.precast_fp32_weight = True" not in source
    node = "tests/e2e/pull_request/two_card/spec_decode/test_spec_decode.py::test_qwen36_35b_dspark_spec_decoding"
    arms = ["fp32_original", "fp32_clean", "bf16_direct", "bf16_gate"]
    records = []
    try:
        # Reverse the second pass to expose warm-cache or execution-order effects.
        for repeat in range(2):
            for arm in arms if repeat == 0 else reversed(arms):
                variant = _variant(source, arm)
                module.write_text(variant)
                # Do not allow equal-sized source replacements to reuse stale bytecode.
                for cache in (module.parent / "__pycache__").glob("fused_moe.*.pyc"):
                    cache.unlink()
                log = tmp_path / f"{repeat}_{arm}.log"
                with log.open("w") as stream:
                    result = subprocess.run(
                        [sys.executable, "-m", "pytest", "-sv", "--color=no", node],
                        cwd=root,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        timeout=1000,
                    )
                text = log.read_text()
                matches = re.findall(r"ROUTER_DIAGNOSTIC_RESULT=(\{[^\n]+\})", text)
                assert matches, f"{arm} did not produce metrics; tail:\n{text[-12000:]}"
                metrics = json.loads(matches[0])
                metrics.update(
                    arm=arm,
                    repeat=repeat,
                    returncode=result.returncode,
                    source_sha256=hashlib.sha256(variant.encode()).hexdigest(),
                )
                metrics["token_sha256"] = hashlib.sha256(json.dumps(metrics["token_ids"]).encode()).hexdigest()
                records.append(metrics)
                # Include tokens in the uploaded pytest log for first-divergence analysis.
                print("ROUTER_AB_RECORD=" + json.dumps(metrics), flush=True)
    finally:
        module.write_bytes(original)
        for cache in (module.parent / "__pycache__").glob("fused_moe.*.pyc"):
            cache.unlink()
        (tmp_path / "router_ab_results.json").write_text(json.dumps(records, indent=2))
    assert len(records) == 8
    assert len({json.dumps(record["prompt_token_ids"]) for record in records}) == 1
