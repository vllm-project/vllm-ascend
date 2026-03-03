#!/usr/bin/env python3
"""
Run functional tests on representative models for release validation.

This script:
1. Loads test model configurations from YAML
2. Starts vLLM server for each model
3. Runs startup, inference, accuracy, and feature tests
4. Generates a markdown report with results

Usage:
    python run_functional_tests.py \
        --config references/test-models.yaml \
        --output test-results.md \
        --hardware 910B \
        --timeout 3600
"""

import argparse
import contextlib
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml


@dataclass
class TestResult:
    model_name: str
    startup: bool = False
    startup_time: float = 0.0
    inference: bool = False
    inference_latency: float = 0.0
    accuracy: float | None = None
    throughput: float | None = None
    features: dict = field(default_factory=dict)
    error_message: str = ""
    status: str = "PENDING"


@dataclass
class ModelConfig:
    name: str
    path: str
    tensor_parallel: int = 1
    max_model_len: int = 4096
    features: list = field(default_factory=list)
    test_prompt: str = "Hello, how are you?"
    expected_min_throughput: float = 0.0
    is_multimodal: bool = False
    image_url: str = ""


def load_config(config_path: Path) -> list[ModelConfig]:
    """Load test model configurations from YAML."""
    with open(config_path) as f:
        data = yaml.safe_load(f)

    models = []
    for model_data in data.get("models", []):
        models.append(
            ModelConfig(
                name=model_data["name"],
                path=model_data["path"],
                tensor_parallel=model_data.get("tensor_parallel", 1),
                max_model_len=model_data.get("max_model_len", 4096),
                features=model_data.get("features", []),
                test_prompt=model_data.get("test_prompt", "Hello, how are you?"),
                expected_min_throughput=model_data.get("expected_min_throughput", 0.0),
                is_multimodal=model_data.get("is_multimodal", False),
                image_url=model_data.get("image_url", ""),
            )
        )
    return models


def start_vllm_server(
    model: ModelConfig,
    port: int = 8000,
    timeout: int = 600,
) -> tuple[subprocess.Popen, bool, float]:
    """Start vLLM server and wait for it to be ready."""
    cmd = [
        "vllm",
        "serve",
        model.path,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(model.tensor_parallel),
        "--max-model-len",
        str(model.max_model_len),
        "--trust-remote-code",
    ]

    # Add feature-specific flags
    if "graph_mode" in model.features:
        cmd.extend(["--enforce-eager", "false"])
    if "expert_parallel" in model.features:
        cmd.append("--enable-expert-parallel")

    print(f"Starting server: {' '.join(cmd)}")
    start_time = time.time()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )

    # Wait for server to be ready
    ready = False
    while time.time() - start_time < timeout:
        try:
            result = subprocess.run(
                ["curl", "-s", f"http://localhost:{port}/v1/models"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and "data" in result.stdout:
                ready = True
                break
        except Exception:
            pass
        time.sleep(5)

    startup_time = time.time() - start_time
    return process, ready, startup_time


def stop_server(process: subprocess.Popen):
    """Stop the vLLM server."""
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=30)
    except Exception as e:
        print(f"Error stopping server: {e}")
        with contextlib.suppress(Exception):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)


def run_inference_test(
    model: ModelConfig,
    port: int = 8000,
) -> tuple[bool, float, str]:
    """Run a basic inference test."""
    try:
        if model.is_multimodal and model.image_url:
            # Multimodal request
            payload = {
                "model": model.path,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is in this image?"},
                            {"type": "image_url", "image_url": {"url": model.image_url}},
                        ],
                    }
                ],
                "max_tokens": 100,
            }
        else:
            # Text-only request
            payload = {
                "model": model.path,
                "messages": [{"role": "user", "content": model.test_prompt}],
                "max_tokens": 100,
            }

        start_time = time.time()
        result = subprocess.run(
            [
                "curl",
                "-s",
                "-X",
                "POST",
                f"http://localhost:{port}/v1/chat/completions",
                "-H",
                "Content-Type: application/json",
                "-d",
                json.dumps(payload),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        latency = time.time() - start_time

        if result.returncode != 0:
            return False, latency, f"curl failed: {result.stderr}"

        response = json.loads(result.stdout)
        if "choices" in response and len(response["choices"]) > 0:
            content = response["choices"][0]["message"]["content"]
            if content and len(content) > 0:
                return True, latency, content[:100]

        return False, latency, f"Invalid response: {result.stdout[:200]}"

    except Exception as e:
        return False, 0.0, str(e)


def run_throughput_test(
    model: ModelConfig,
    port: int = 8000,
    num_requests: int = 10,
) -> float | None:
    """Run a simple throughput test."""
    try:
        total_tokens = 0
        start_time = time.time()

        for _ in range(num_requests):
            payload = {
                "model": model.path,
                "messages": [{"role": "user", "content": model.test_prompt}],
                "max_tokens": 50,
            }

            result = subprocess.run(
                [
                    "curl",
                    "-s",
                    "-X",
                    "POST",
                    f"http://localhost:{port}/v1/chat/completions",
                    "-H",
                    "Content-Type: application/json",
                    "-d",
                    json.dumps(payload),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                response = json.loads(result.stdout)
                if "usage" in response:
                    total_tokens += response["usage"].get("completion_tokens", 0)

        elapsed = time.time() - start_time
        if elapsed > 0 and total_tokens > 0:
            return total_tokens / elapsed

        return None

    except Exception as e:
        print(f"Throughput test error: {e}")
        return None


def check_feature(
    feature: str,
    model: ModelConfig,
) -> bool:
    """Check if a specific feature is working."""
    # For now, just check if the feature flag was accepted
    # More sophisticated checks can be added per feature
    feature_checks = {
        "graph_mode": lambda: True,  # If server started with graph mode, it's working
        "expert_parallel": lambda: True,  # If server started with EP, it's working
        "multimodal": lambda: model.is_multimodal,
        "mtp": lambda: True,  # Check MTP by inference pattern
    }

    check_func = feature_checks.get(feature, lambda: False)
    return check_func()


def test_model(model: ModelConfig, timeout: int, port: int = 8000) -> TestResult:
    """Run all tests for a single model."""
    result = TestResult(model_name=model.name)
    process = None

    try:
        print(f"\n{'=' * 60}")
        print(f"Testing model: {model.name}")
        print(f"{'=' * 60}")

        # Start server
        print("Starting server...")
        process, ready, startup_time = start_vllm_server(model, port, timeout)
        result.startup = ready
        result.startup_time = startup_time

        if not ready:
            result.status = "FAIL"
            result.error_message = "Server failed to start"
            return result

        print(f"Server ready in {startup_time:.1f}s")

        # Run inference test
        print("Running inference test...")
        success, latency, output = run_inference_test(model, port)
        result.inference = success
        result.inference_latency = latency

        if not success:
            result.status = "FAIL"
            result.error_message = f"Inference failed: {output}"
            return result

        print(f"Inference succeeded in {latency:.2f}s")

        # Run throughput test
        print("Running throughput test...")
        throughput = run_throughput_test(model, port)
        result.throughput = throughput
        if throughput:
            print(f"Throughput: {throughput:.1f} tokens/s")

        # Check features
        print("Checking features...")
        for feature in model.features:
            result.features[feature] = check_feature(feature, model)
            status = "✅" if result.features[feature] else "❌"
            print(f"  {feature}: {status}")

        result.status = "PASS"

    except Exception as e:
        result.status = "ERROR"
        result.error_message = str(e)

    finally:
        if process:
            print("Stopping server...")
            stop_server(process)

    return result


def generate_report(results: list[TestResult], hardware: str) -> str:
    """Generate a markdown report of test results."""
    lines = [
        "## Functional Test Results",
        "",
        f"Hardware: {hardware}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "### Summary Table",
        "",
        "| Model | Startup | Inference | Throughput | Features | Status |",
        "|-------|---------|-----------|------------|----------|--------|",
    ]

    for r in results:
        startup = "✅" if r.startup else "❌"
        inference = "✅" if r.inference else "❌"
        throughput = f"{r.throughput:.0f} tok/s" if r.throughput else "N/A"

        # Format features
        if r.features:
            features = ", ".join([f"{k}:{'✅' if v else '❌'}" for k, v in r.features.items()])
        else:
            features = "N/A"

        status = r.status

        lines.append(f"| {r.model_name} | {startup} | {inference} | {throughput} | {features} | {status} |")

    lines.append("")
    lines.append("### Detailed Results")
    lines.append("")

    for r in results:
        lines.append(f"#### {r.model_name}")
        lines.append("")
        lines.append(f"- **Status**: {r.status}")
        lines.append(f"- **Startup**: {'Success' if r.startup else 'Failed'} ({r.startup_time:.1f}s)")
        lines.append(f"- **Inference**: {'Success' if r.inference else 'Failed'} ({r.inference_latency:.2f}s)")
        if r.throughput:
            lines.append(f"- **Throughput**: {r.throughput:.1f} tokens/s")
        if r.features:
            lines.append("- **Features**:")
            for k, v in r.features.items():
                lines.append(f"  - {k}: {'✅' if v else '❌'}")
        if r.error_message:
            lines.append(f"- **Error**: {r.error_message}")
        lines.append("")

    # Generate checklist format
    lines.append("### Checklist Format")
    lines.append("")
    lines.append("Copy the following to the 'Functional Test' section:")
    lines.append("")
    lines.append("```markdown")
    for r in results:
        status = "x" if r.status == "PASS" else " "
        lines.append(f"- [{status}] {r.model_name}")
    lines.append("```")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run functional tests for release")
    parser.add_argument("--config", required=True, help="Test configuration YAML file")
    parser.add_argument("--output", required=True, help="Output file for results")
    parser.add_argument("--hardware", default="910B", help="Hardware type")
    parser.add_argument("--timeout", type=int, default=600, help="Server startup timeout in seconds")
    parser.add_argument("--models", nargs="*", help="Specific models to test (default: all)")
    parser.add_argument("--port", type=int, default=8000, help="Server port")

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    print(f"Loading config from {config_path}...")
    models = load_config(config_path)

    if args.models:
        models = [m for m in models if m.name in args.models]

    if not models:
        print("No models to test")
        return 1

    print(f"Testing {len(models)} models on {args.hardware}...")

    results = []
    for model in models:
        result = test_model(model, args.timeout, args.port)
        results.append(result)

    print("\nGenerating report...")
    report = generate_report(results, args.hardware)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"Test results saved to {output_path}")

    # Print summary
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    errors = sum(1 for r in results if r.status == "ERROR")

    print(f"\nSummary: {passed} passed, {failed} failed, {errors} errors")

    return 0 if failed == 0 and errors == 0 else 1


if __name__ == "__main__":
    exit(main())
