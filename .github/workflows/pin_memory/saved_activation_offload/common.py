# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Shared helpers for pin_memory experiment drivers (subprocess arms, JSON lines)."""

from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import sys


def free_port():
    # A stale listener from an interrupted rank makes the next init hang in a
    # collective, so always rendezvous on a fresh ephemeral port.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def dist_env():
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(free_port()),
        RANK="0",
        WORLD_SIZE="1",
        LOCAL_RANK="0",
    )


def patch_cpp_extension_drop_cxx17():
    # torch-nightly requires C++20; SYCL toolchain flags may carry -std=c++17,
    # which (appearing last) downgrades the dialect and breaks torch headers.
    import torch.utils.cpp_extension as cpp_ext

    orig_load = cpp_ext.load

    def load_without_cxx17(*args, **kwargs):
        for key in ("extra_cflags", "extra_cxxflags"):
            if kwargs.get(key):
                kwargs[key] = [flag for flag in kwargs[key] if flag != "-std=c++17"]
        return orig_load(*args, **kwargs)

    cpp_ext.load = load_without_cxx17


def print_arm_result(result):
    print("ARMRESULT=" + json.dumps(result), flush=True)


def print_driver_result(summary):
    print("DRIVERRESULT=" + json.dumps(summary), flush=True)


def parse_arm_result(stdout):
    for line in stdout.splitlines():
        if line.startswith("ARMRESULT="):
            return json.loads(line[len("ARMRESULT=") :])
    return None


def run_arm_subprocess(script_path, extra_args, env=None):
    command = [sys.executable, os.path.abspath(script_path), *extra_args]
    print(f"[driver] {' '.join(command)}", flush=True)
    proc = subprocess.run(command, env=env or os.environ.copy(), capture_output=True, text=True)
    arm = parse_arm_result(proc.stdout)
    if arm is None:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"arm produced no ARMRESULT (rc={proc.returncode})")
    return arm


def param_numel(param):
    return int(getattr(param, "ds_numel", param.numel()))


def _call_with_known_kwargs(fn, kwargs):
    import inspect

    params = inspect.signature(fn).parameters
    accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    used = kwargs if accepts_var_kw else {k: v for k, v in kwargs.items() if k in params}
    fn(**used)
    return used


def apply_liger(model_id):
    """Patch modeling classes before from_config. Returns enabled feature names."""
    enabled = []
    lower = (model_id or "").lower()
    if "qwen3.5" in lower or "qwen3_5" in lower:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5 as apply_fn

        family = "qwen3_5"
    else:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3 as apply_fn

        family = "qwen3"
    # FLCE-only: rope/rms_norm/swiglu fusion left disabled. Their custom
    # autograd.Functions stash backward tensors outside torch.autograd.graph's
    # saved_tensors_hooks (likely via direct ctx attribute assignment rather
    # than ctx.save_for_backward), so DeepSpeed's non_reentrant_checkpoint
    # placeholder-swap can never discard them -- confirmed empirically to
    # blow up per-layer forward memory (retained until backward touches it).
    kwargs = {
        "rope": False,
        "rms_norm": False,
        "swiglu": False,
        "cross_entropy": False,
        "fused_linear_cross_entropy": True,
    }
    used = _call_with_known_kwargs(apply_fn, kwargs)
    enabled.append(f"liger_{family}")
    if used.get("fused_linear_cross_entropy"):
        enabled.append("liger_fused_linear_ce")
    print(f"liger applied via {apply_fn.__name__} kwargs={sorted(used)}", flush=True)
    return enabled


def _enable_non_reentrant_checkpointing(model):
    # Pack hooks only see checkpoint inputs when use_reentrant is False.
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()


def _install_tiled_mlp(model):
    from deepspeed.runtime.sequence_parallel.ulysses_sp import TiledMLP

    patched = []
    seen = set()
    for module in model.modules():
        if not (hasattr(module, "gate_proj") and hasattr(module, "up_proj") and hasattr(module, "down_proj")):
            continue
        cls = type(module)
        if cls in seen:
            continue
        seen.add(cls)
        inner = cls.forward

        def tiled_forward(self, hidden_states, *args, _inner=inner, **kwargs):
            x = hidden_states
            if not hasattr(x, "dim") or x.dim() != 3:
                return _inner(self, x, *args, **kwargs)
            seqlen = int(x.shape[-2])
            hidden = int(x.shape[-1])
            num_shards = max(1, math.ceil(seqlen / max(hidden, 1)))
            compute_params = [self.down_proj.weight, self.gate_proj.weight, self.up_proj.weight]

            def mlp_forward(mod, shard):
                return _inner(mod, shard, *args, **kwargs)

            return TiledMLP.apply(mlp_forward, self, x, num_shards, compute_params)

        cls.forward = tiled_forward
        patched.append(cls.__name__)
        print(f"tiled MLP patched {cls.__name__}", flush=True)
    if not patched:
        raise RuntimeError("TiledMLP: no gate/up/down MLP modules found")
    return ["tiled_mlp:" + ",".join(patched)]


def _tiled_fused_ce_loss(causal_lm, hidden_states, labels):
    import torch.nn.functional as F
    from deepspeed.runtime.sequence_parallel.ulysses_sp import TiledFusedLogitsLoss

    shift_h = hidden_states[:, :-1, :].contiguous()
    shift_y = labels[:, 1:].contiguous()
    seqlen = int(shift_h.shape[1])
    hidden = int(shift_h.shape[-1])
    shards = max(1, math.ceil(seqlen / max(hidden, 1)))
    compute_params = [causal_lm.lm_head.weight]

    def loss_fn(mod, x, y):
        logits = mod.lm_head(x)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

    return TiledFusedLogitsLoss.apply(loss_fn, causal_lm, shift_h, shift_y, None, shards, compute_params, "mean")


def _backbone_hidden(module, ids):
    core = getattr(module, "model", None)
    if core is None:
        raise RuntimeError("causal LM has no .model backbone for tiled CE")
    return core(input_ids=ids).last_hidden_state


def build_hf_causal_lm(model_id, torch_dtype):
    """Config + random weights. Prefer text-only if the hub card is multimodal.

    Skip Hugging Face ``post_init`` weight fill: under ``deepspeed.zero.Init``
    parameters are already partitioned (local ``.shape`` can be 0), so Qwen3.5
    ``copy_`` init and ``Embedding.weight.shape[0]`` as vocab both break.
    """
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    try:
        # transformers 5.16 (Qwen3.5 pin): moved out of modeling_utils.
        from transformers.initialization import no_init_weights
    except ImportError:
        from transformers.modeling_utils import no_init_weights

    print(f"transformers={transformers.__version__} model_id={model_id}", flush=True)
    config = AutoConfig.from_pretrained(model_id)
    text_config = config
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        text_config = get_text_config() or config
    elif getattr(config, "text_config", None) is not None:
        text_config = config.text_config

    model_type = getattr(text_config, "model_type", None) or getattr(config, "model_type", "")
    if "moe" in str(model_type):
        raise ValueError(f"MoE model_type {model_type!r} is out of scope for this bench")

    def from_config(**extra):
        try:
            return AutoModelForCausalLM.from_config(text_config, dtype=torch_dtype, **extra)
        except TypeError:
            return AutoModelForCausalLM.from_config(text_config, torch_dtype=torch_dtype, **extra)

    # Pin/offload wall time does not need pretrained-style init values.
    with no_init_weights():
        try:
            model = from_config(attn_implementation="flash_attention_2")
        except Exception as exc:
            print(f"flash_attention_2 unavailable ({type(exc).__name__}: {exc}); default attn", flush=True)
            model = from_config()

    vocab = int(getattr(text_config, "vocab_size", 0) or getattr(config, "vocab_size", 0) or 0)
    if vocab <= 0:
        weight = model.get_input_embeddings().weight
        vocab = int(getattr(weight, "ds_shape", weight.shape)[0])
    if vocab <= 0:
        raise ValueError(f"could not determine vocab_size for {model_id}")
    return model, vocab, transformers.__version__, model_type
