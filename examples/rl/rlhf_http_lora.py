# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstrates reinforcement learning from human feedback (RLHF) with LoRA
weight updates using vLLM via HTTP API.

Unlike the full-model weight transfer examples (``rlhf_http_hccl.py``),
this script demonstrates the LoRA-based RL training workflow where:

* The inference server runs the **base model** with LoRA support enabled.
* The trainer fine-tunes **LoRA adapters** (lightweight, low-rank matrices)
  on a separate device.
* Updated LoRA weights are pushed to the inference server via the HTTP API
  (``/v1/load_lora_adapter`` and ``/v1/unload_lora_adapter``).
* The server immediately serves requests with or without the active LoRA,
  without needing to pause/resume or rebuild the KV cache.

This is a common pattern in online RLHF (e.g., PPO, GRPO) where the policy
is fine-tuned with LoRA and the updated adapter is synced to the inference
engine at each iteration. LoRA updates are lighter-weight than full-model
transfers — no HCCL/NCCL process group is needed, and the server never
pauses inference.

Prerequisites:
    * The base model and the LoRA adapters must be reachable from this
      machine (local directories or HuggingFace model IDs).
    * Port 8000 must be free, or point ``BASE_URL`` at an already running
      server that has the runtime LoRA endpoints enabled.

The script starts its own vLLM server (setting
``VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`` and ``--enable-lora``) and stops it
on exit, so a single command runs the whole demo::

    $ python rlhf_http_lora.py

The example performs the following steps:

1. Generate text using the vLLM server via OpenAI-compatible API
   **without** any LoRA adapter (baseline).
2. Deploy the initial policy: load the **Alice** adapter under the fixed
   name ``policy`` via ``POST /v1/load_lora_adapter`` and generate — the
   model identifies as Alice.
3. **Push updated weights in place**: load the **Bob** adapter onto the
   *same* ``policy`` name via ``POST /v1/load_lora_adapter`` with
   ``"load_inplace": true``, and generate — the model now identifies as
   Bob.  The adapter name never changes: it is just a slot, and the
   trainer pushes new checkpoints onto it each RL iteration — no unload,
   no pause/resume, and in-flight requests keep running.
4. **Roll back to a previous checkpoint**: push Alice's weights onto
   ``policy`` in place again (Bob → Alice) and generate — the model
   identifies as Alice again, showing that any older checkpoint can be
   restored the same way.
5. **Unload** ``policy`` and generate again — the server falls back to
   the base model, confirming the rollback path.

A single identity prompt is used so the whole demo completes in about a
minute.  The script unloads ``policy`` at startup (tolerating 404), so it
can be re-run against a warm server.

This demonstrates the complete RL training LoRA update cycle:
**serve → push Alice → serve → push Bob (updated weights) → serve →
push Alice (rollback) → serve → unload → serve**.

Usage in a real RL training loop
--------------------------------

In a real RLHF setup (e.g., with TRL or OpenRLHF), the trainer would:

.. code-block:: python

    # Each RL iteration:
    for iteration in range(num_iterations):
        # 1. Generate rollouts using the current LoRA adapter
        rollouts = generate_with_lora(client, model_name, prompts, lora_name)

        # 2. Compute rewards and update the LoRA weights on the trainer
        lora_state_dict = train_lora_step(rollouts, reward_model)

        # 3. Save the updated LoRA adapter to disk
        save_lora_adapter(lora_state_dict, "/tmp/lora_checkpoint")

        # 4. Push the updated LoRA onto the same adapter name in place
        load_lora_adapter(base_url, lora_name, "/tmp/lora_checkpoint",
                          load_inplace=True)

        # The server now serves requests with the updated LoRA
        # No unload, no pause/resume, no HCCL broadcast needed
"""

import os
import subprocess
import time
from urllib.parse import urlsplit

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Use 127.0.0.1 instead of localhost: on some hosts "localhost" resolves to
# IPv6 first and the HTTP clients cannot fall back to the IPv4-bound server.
BASE_URL = "http://127.0.0.1:8000"
# Local paths on this machine; swap for HuggingFace model IDs if desired.
MODEL_NAME = "Qwen/Qwen3-0.6B"

# How long to wait for the server started by this script to become ready.
SERVER_START_TIMEOUT = 600

# Two self-cognition LoRA adapters to demonstrate the update workflow.
# Local paths on this machine; in real RL training, replace these with your
# own trained adapter paths.
LORA_ALICE = (
    "Alice",
    "charent/self_cognition_Alice",
)
LORA_BOB = (
    "Bob",
    "charent/self_cognition_Bob",
)

# The fixed adapter name the trainer pushes weights onto each iteration.
# The name is just a slot; only the weights behind it change.
POLICY_NAME = "policy"

# One identity prompt is enough to observe the adapter switch.
PROMPTS = ["Hi, tell me about you"]


# ---------------------------------------------------------------------------
# OpenAI-compatible generation
# ---------------------------------------------------------------------------
def generate_completions(
    client: OpenAI,
    model: str,
    prompts: list[str],
    lora_name: str | None = None,
) -> list[str]:
    """Generate completions using the OpenAI-compatible API.

    Args:
        client: OpenAI client pointing at the vLLM server.
        model: Base model name.
        prompts: List of prompt strings.
        lora_name: If set, target the LoRA adapter registered under this
            name via /v1/load_lora_adapter.  The ``model`` field carries
            the adapter name directly — vLLM Ascend resolves it against
            the loaded adapters.

    Returns:
        List of generated text strings (one per prompt).
    """
    if lora_name is not None:
        # The adapter name is used as the model id: the server matches it
        # against the adapters registered via /v1/load_lora_adapter.
        model = lora_name

    results = []
    for prompt in prompts:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=64,
            temperature=0,
        )
        results.append(response.choices[0].text)
    return results


# ---------------------------------------------------------------------------
# LoRA adapter management via HTTP
# ---------------------------------------------------------------------------
def load_lora_adapter(
    base_url: str,
    lora_name: str,
    lora_path: str,
    load_inplace: bool = False,
) -> None:
    """Load (or reload) a LoRA adapter into the running vLLM server.

    In an RL training loop, call this after each training step to push
    updated LoRA weights to the inference engine.

    Args:
        base_url: vLLM server base URL (e.g. ``"http://localhost:8000"``).
        lora_name: A unique name for this adapter (used in generation
            requests to activate it).
        lora_path: Path to the LoRA adapter — can be a local directory
            or a HuggingFace model ID.
        load_inplace: If True, replace the adapter registered under
            ``lora_name`` without unloading it first.  This is the
            production RL pattern: the trainer keeps a stable adapter
            name and pushes new checkpoints onto it each iteration.
            Loading a name that already exists without ``load_inplace``
            is rejected by the server.
    """
    url = f"{base_url}/v1/load_lora_adapter"
    payload = {
        "lora_name": lora_name,
        "lora_path": lora_path,
    }
    if load_inplace:
        payload["load_inplace"] = True
    print(f"[trainer] Loading LoRA '{lora_name}' from {lora_path} "
          f"(load_inplace={load_inplace}) ...")
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    print(f"[trainer] LoRA '{lora_name}' loaded successfully")


def unload_lora_adapter(
    base_url: str,
    lora_name: str,
    ignore_missing: bool = False,
) -> None:
    """Unload a LoRA adapter from the running vLLM server.

    Use this to free GPU memory or to switch to a different adapter
    between RL iterations.

    Args:
        base_url: vLLM server base URL.
        lora_name: Name of the adapter to unload.
        ignore_missing: If True, treat a 404 (adapter not loaded) as
            success — useful to reset adapter state at startup.
    """
    url = f"{base_url}/v1/unload_lora_adapter"
    payload = {"lora_name": lora_name}
    print(f"[trainer] Unloading LoRA '{lora_name}' ...")
    response = requests.post(url, json=payload, timeout=60)
    if ignore_missing and response.status_code == 404:
        print(f"[trainer] LoRA '{lora_name}' was not loaded; nothing to unload")
        return
    response.raise_for_status()
    print(f"[trainer] LoRA '{lora_name}' unloaded successfully")


# ---------------------------------------------------------------------------
# vLLM server management
# ---------------------------------------------------------------------------
class LoRAServer:
    """Start a local vLLM server for the demo and stop it on exit.

    If a server is already serving ``base_url`` (e.g. started manually),
    it is detected and left running.
    """

    def __init__(self, model: str, base_url: str) -> None:
        self.external = False
        # If a server is already serving this port, reuse it.
        try:
            if requests.get(f"{base_url}/health", timeout=5).status_code == 200:
                print(f"[server] using an existing server at {base_url}",
                      flush=True)
                self.external = True
                self.proc = None
                return
        except requests.exceptions.RequestException:
            pass

        port = urlsplit(base_url).port or 8000
        env = os.environ.copy()
        # Enable the runtime /v1/load_lora_adapter and /v1/unload_lora_adapter
        # endpoints.
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
        command = [
            "vllm", "serve", model,
            "--enforce-eager",
            "--enable-lora",
            "--max-lora-rank", "8",
            "--max-cpu-loras", "4",
            "--port", str(port),
        ]
        print(f"[server] starting: {' '.join(command)}", flush=True)
        self.proc = subprocess.Popen(command, env=env)
        self._wait_until_ready(base_url)

    def _wait_until_ready(self, base_url: str) -> None:
        health_url = f"{base_url}/health"
        deadline = time.monotonic() + SERVER_START_TIMEOUT
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                # The spawned server exited.  If something else is already
                # serving this port, use it instead of failing.
                try:
                    if requests.get(health_url, timeout=5).status_code == 200:
                        print(f"[server] using an existing server at {base_url}",
                              flush=True)
                        self.external = True
                        return
                except requests.exceptions.RequestException:
                    pass
                raise RuntimeError(
                    f"vLLM server exited with status {self.proc.returncode}"
                )
            try:
                if requests.get(health_url, timeout=5).status_code == 200:
                    print(f"[server] ready: {health_url}", flush=True)
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)
        self.stop()
        raise TimeoutError(
            f"vLLM server did not become ready within {SERVER_START_TIMEOUT}s"
        )

    def stop(self) -> None:
        if self.external or self.proc.poll() is not None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
def generate_and_print(
    client: OpenAI,
    title: str,
    model: str,
    lora_name: str | None = None,
) -> list[str]:
    """Run one generation round and print the outputs.

    Args:
        client: OpenAI client pointing at the vLLM server.
        title: Section header for the round.
        model: Base model name.
        lora_name: If set, target the adapter registered under this name.

    Returns:
        List of generated text strings (one per prompt).
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    outputs = generate_completions(client, model, PROMPTS, lora_name=lora_name)
    for prompt, output in zip(PROMPTS, outputs):
        print(f"Prompt: {prompt!r}")
        print(f"Output: {output!r}")
        print("-" * 40)
    return outputs


def main():
    # Start a local vLLM server (unless one is already serving BASE_URL)
    # and stop it again on exit.
    server = LoRAServer(MODEL_NAME, BASE_URL)
    try:
        # Create an OpenAI client pointing to the vLLM server.
        client = OpenAI(
            base_url=f"{BASE_URL}/v1",
            api_key="EMPTY",  # vLLM doesn't require an API key by default
        )

        _, alice_path = LORA_ALICE
        _, bob_path = LORA_BOB

        # Reset adapter state so the demo is re-runnable against a warm server.
        try:
            unload_lora_adapter(BASE_URL, POLICY_NAME, ignore_missing=True)
        except requests.exceptions.ConnectionError as exc:
            raise SystemExit(
                f"Could not reach the vLLM server at {BASE_URL}."
            ) from exc

        # ── Step 1: Generate WITHOUT LoRA (baseline) ───────────────────
        baseline_outputs = generate_and_print(
            client, "Step 1: Generating WITHOUT LoRA (baseline)", MODEL_NAME
        )

        # ── Step 2: Deploy the initial policy (Alice under a fixed name) ──
        load_lora_adapter(BASE_URL, POLICY_NAME, alice_path)
        alice_outputs = generate_and_print(
            client,
            "Step 2: Generating with the initial policy (Alice)",
            MODEL_NAME,
            lora_name=POLICY_NAME,
        )

        # ── Step 3: Push updated weights in place (Alice → Bob) ───────
        # The trainer pushes new checkpoint weights onto the existing adapter
        # name with load_inplace=True.  The name stays "policy": requests
        # keep using it, but they now hit the new weights — no unload, no
        # pause/resume, in-flight requests are not interrupted.
        load_lora_adapter(BASE_URL, POLICY_NAME, bob_path, load_inplace=True)
        bob_outputs = generate_and_print(
            client,
            "Step 3: Generating after inplace push Alice → Bob",
            MODEL_NAME,
            lora_name=POLICY_NAME,
        )

        # ── Step 4: Roll back to a previous checkpoint (Bob → Alice) ──
        load_lora_adapter(BASE_URL, POLICY_NAME, alice_path, load_inplace=True)
        rollback_outputs = generate_and_print(
            client,
            "Step 4: Generating after inplace push Bob → Alice (rollback)",
            MODEL_NAME,
            lora_name=POLICY_NAME,
        )

        # ── Step 5: Unload the policy and verify rollback to base ─────
        unload_lora_adapter(BASE_URL, POLICY_NAME)
        after_unload_outputs = generate_and_print(
            client,
            "Step 5: Generating after unload (back to the base model)",
            MODEL_NAME,
        )

        # ── Summary ────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        identity_idx = 0  # "Hi, tell me about you"
        print(f"Baseline                 : {baseline_outputs[identity_idx]!r}")
        print(f"policy <- Alice (deploy) : {alice_outputs[identity_idx]!r}")
        print(f"policy <- Bob (inplace)  : {bob_outputs[identity_idx]!r}")
        print(f"policy <- Alice (rollback): {rollback_outputs[identity_idx]!r}")
        print(f"After unload             : {after_unload_outputs[identity_idx]!r}")

        # Verify each stage of the policy lifecycle.  An assertion failure
        # means the corresponding update did not take effect.
        assert alice_outputs[identity_idx] != baseline_outputs[identity_idx], (
            "Deploying Alice under 'policy' did not change the output vs baseline"
        )
        assert bob_outputs[identity_idx] != alice_outputs[identity_idx], (
            "Inplace push Alice → Bob did not take effect"
        )
        assert rollback_outputs[identity_idx] != bob_outputs[identity_idx], (
            "Inplace push Bob → Alice (rollback) did not take effect"
        )
        assert after_unload_outputs[identity_idx] != rollback_outputs[identity_idx], (
            "Unload did not roll inference back to the base model"
        )
        print("\nAll checks passed.")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
