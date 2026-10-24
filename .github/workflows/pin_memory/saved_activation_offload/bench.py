# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Benchmark credit-bounded saved-intermediate activation CPU offload."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import statistics
import sys
import tempfile
import threading
import time

import regex as re
import torch

INTER_MLP_ENTRY_DEFINITION = (
    "mean consecutive target-MLP forward-entry interval; under offload this is the observed "
    "layer period T and includes credit stalls, so the reference forward period must come "
    "from a --mode none run"
)
MLP_SPAN_DEFINITION = (
    "mean compute-stream interval from post-acquire enqueue to module exit; excludes the host "
    "credit wait but may include compute already queued ahead of the module"
)
T_D_DEFINITION = "mean per-group sum of individually bracketed D2H copies"

_PIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PIN_ROOT not in sys.path:
    sys.path.insert(0, _PIN_ROOT)

from common import (  # noqa: E402
    apply_liger,
    build_hf_causal_lm,
    dist_env,
    param_numel,
    patch_cpp_extension_drop_cxx17,
    print_arm_result,
    print_driver_result,
    run_arm_subprocess,
)
from credit_offload import (  # noqa: E402
    Credit,
    CreditDeadlock,
    CreditPool,
    CreditTimeout,
    HostBufferPool,
    SavedIntermediateOffload,
    has_internal_overlap,
    noop_saved_tensors,
    register_forward_hook_always,
    storage_identity,
    storages_of,
    tensor_signature,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="HF model id; omit for the synthetic MLP stack")
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--credits", type=int, default=2)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument(
        "--offload-tensors",
        type=int,
        default=3,
        help="top-K eligible tensors per layer to offload, by size. -1 means "
        "no cap: offload every eligible tensor in scope (apples-to-apples "
        "with activation_offload's full-block discard+recompute).",
    )
    parser.add_argument(
        "--offload-modules",
        default=r"(?:^|\.)layers\.\d+$",
        help="regex matched against model.named_modules() names; must match "
        "exactly one module per decoder layer (_resolve_scopes enforces "
        "this). Default brackets the whole decoder layer (self_attn+mlp+"
        "norms), matching the granularity activation_offload's "
        "gradient_checkpointing_enable() wraps.",
    )
    parser.add_argument("--min-offload-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--skip-last-layers", type=int, default=1)
    parser.add_argument("--credit-timeout-s", type=float, default=60.0)
    parser.add_argument(
        "--act-mem-opt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Liger FLCE-only kernels when --model is set, matching "
        "activation_offload/bench.py's --act-mem-opt env minus TiledMLP: TiledMLP's "
        "autograd.Function only saves the MLP's input tensor, which this benchmark's "
        "Discovery._pack always excludes as aliases_input, so every scope would have "
        "nothing eligible to offload. Not portable here; left out.",
    )
    parser.add_argument(
        "--offload-optimizer",
        action="store_true",
        help="ZeRO-2 CPU AdamW offload (params stay on GPU). HF --model only. Matches "
        "activation_offload/bench.py's optimizer-offload axis; does not change this "
        "benchmark's own credit-bounded MLP-tensor offload.",
    )
    parser.add_argument("--mode", choices=("offload", "none", "remat"), default="offload")
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument(
        "--discover-calib-seqs",
        type=int,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="run Discovery at these two (smaller) seqlens instead of --seq, then "
        "linearly extrapolate each record's shape/stride/nbytes to --seq before "
        "freezing the manifest. Lets Discovery's own full-residency forward (which "
        "must materialize every candidate tensor at once) stay cheap even when "
        "--seq itself would OOM a real discovery pass.",
    )
    parser.add_argument("--n-sweep", action="store_true")
    parser.add_argument("--pin", type=int, choices=(0, 1), default=None, help="internal: single subprocess arm")
    parser.add_argument("--discover-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--smoke", action="store_true", help="run CPU-safe scheduler tests and exit")
    return parser.parse_args()


def _build_synthetic(hidden, layers, device):
    import torch

    class MLP(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.gate_proj = torch.nn.Linear(width, 4 * width, bias=False)
            self.up_proj = torch.nn.Linear(width, 4 * width, bias=False)
            self.down_proj = torch.nn.Linear(4 * width, width, bias=False)

        def forward(self, hidden_states):
            gate = torch.nn.functional.silu(self.gate_proj(hidden_states))
            return self.down_proj(gate * self.up_proj(hidden_states))

    class Block(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.mlp = MLP(width)

        def forward(self, hidden_states):
            return hidden_states + self.mlp(hidden_states)

    class Net(torch.nn.Module):
        def __init__(self, width, depth):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block(width) for _ in range(depth)])

        def forward(self, hidden_states):
            for layer in self.layers:
                hidden_states = layer(hidden_states)
            return hidden_states.float().square().mean()

    model = Net(hidden, layers).to(device=device, dtype=torch.bfloat16)
    return model


def _build_recipe(args, device):
    import torch

    torch.manual_seed(1234)
    if not args.model:
        model = _build_synthetic(args.hidden, args.layers, device)
        inputs = torch.randn(args.batch, args.seq, args.hidden, device=device, dtype=torch.bfloat16)

        def forward_loss():
            return model(inputs)

        return model, forward_loss, None, None, None, None

    if args.offload_optimizer:
        dist_env()
    if args.act_mem_opt:
        try:
            apply_liger(args.model)
        except Exception as exc:
            print(f"liger unavailable ({type(exc).__name__}: {exc})", flush=True)
    model, vocab, transformers_version, model_type = build_hf_causal_lm(args.model, torch.bfloat16)
    ds_engine = None
    if args.offload_optimizer:
        import deepspeed

        ds_config = {
            "train_micro_batch_size_per_gpu": args.batch,
            "gradient_accumulation_steps": 1,
            "bf16": {"enabled": True},
            "optimizer": {"type": "AdamW", "params": {"lr": 1e-4}},
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {"device": "cpu", "pin_memory": bool(args.pin)},
            },
        }
        ds_engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)
        model.train()
        ids = torch.randint(0, vocab, (args.batch, args.seq), dtype=torch.long, device=ds_engine.device)

        def forward_loss():
            return ds_engine(input_ids=ids, labels=ids).loss
    else:
        model = model.to(device=device, dtype=torch.bfloat16)
        model.train()
        ids = torch.randint(0, vocab, (args.batch, args.seq), dtype=torch.long, device=device)

        def forward_loss():
            return model(input_ids=ids, labels=ids).loss

    return model, forward_loss, transformers_version, model_type, vocab, ds_engine


def _resolve_scopes(model, pattern, expected_layers):
    regex = re.compile(pattern)
    matches = [(name, module) for name, module in model.named_modules() if regex.search(name)]
    if len(matches) != expected_layers:
        names = [name for name, _ in matches[:8]]
        raise RuntimeError(f"{pattern!r} matched {len(matches)} modules, expected {expected_layers}: {names}")
    indexed = {}
    seen_decoder_layers = set()
    for fallback_idx, (name, module) in enumerate(matches):
        layer_match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
        layer_idx = int(layer_match.group(1)) if layer_match else fallback_idx
        if layer_idx in seen_decoder_layers:
            raise RuntimeError(f"more than one offload scope matched decoder layer {layer_idx}")
        seen_decoder_layers.add(layer_idx)
        indexed[layer_idx] = module
    return indexed


class Discovery:
    """Record all saves first, then freeze a per-layer selector."""

    def __init__(self, model, scopes, accelerator, min_bytes):
        self.model = model
        self.scopes = scopes
        self.accelerator = accelerator
        self.min_bytes = min_bytes
        self.parameter_storages = {
            storage_identity(param) for param in model.parameters() if param.numel() and param.device.type != "meta"
        }
        self.records = []
        self._tls = threading.local()
        self._hooks = []

    def _pre(self, module, args):
        if getattr(self._tls, "active", False):
            raise RuntimeError("nested target-module execution is not supported")
        self._tls.active = True
        self._tls.layer_idx = module._ds_layer_idx
        self._tls.pack_index = 0
        self._tls.input_storages = storages_of(args)

    def _post(self, module, args, output):
        self._tls.active = False
        self._tls.pack_index = 0
        self._tls.input_storages = set()

    def _pack(self, tensor):
        if not getattr(self._tls, "active", False):
            return tensor
        pack_index = self._tls.pack_index
        self._tls.pack_index += 1
        storage_id = storage_identity(tensor)
        nbytes = tensor.numel() * tensor.element_size()
        overlap = has_internal_overlap(tensor)
        aliases_param = storage_id in self.parameter_storages
        aliases_input = storage_id in self._tls.input_storages
        eligible = (
            self.accelerator.on_accelerator(tensor)
            and nbytes >= self.min_bytes
            and not aliases_param
            and not aliases_input
            and not isinstance(tensor, torch.nn.Parameter)
            and not (hasattr(torch.nn, "Buffer") and isinstance(tensor, torch.nn.Buffer))
            and tensor.storage_offset() == 0
            and not overlap
        )
        self.records.append(
            {
                "layer_idx": self._tls.layer_idx,
                "pack_index": pack_index,
                "signature": signature_to_json(tensor_signature(tensor)),
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype),
                "nbytes": nbytes,
                "storage_id": storage_id,
                "aliases_param": aliases_param,
                "aliases_input": aliases_input,
                "eligible": eligible,
                "selected": False,
                "slot": None,
                "slot_owner": False,
            }
        )
        return tensor

    def run(self, forward_loss):
        from torch.autograd.graph import saved_tensors_hooks

        for layer_idx, module in self.scopes.items():
            module._ds_layer_idx = layer_idx
            self._hooks.append(module.register_forward_pre_hook(self._pre))
            self._hooks.append(register_forward_hook_always(module, self._post))
        try:
            with saved_tensors_hooks(self._pack, lambda tensor: tensor):
                loss = forward_loss()
            del loss
        finally:
            for hook in self._hooks:
                hook.remove()
            self._hooks.clear()
        return self.records


def signature_to_json(signature):
    shape, stride, dtype, device, offset = signature
    return [list(shape), list(stride), dtype, device, offset]


def signature_from_json(signature):
    shape, stride, dtype, device, offset = signature
    return (tuple(shape), tuple(stride), dtype, device, offset)


def freeze_manifest(records, offload_tensors, skip_last_layers, layers):
    selected = set()
    last_offloaded = layers - skip_last_layers
    for layer_idx in range(last_offloaded):
        layer_records = [record for record in records if record["layer_idx"] == layer_idx and record["eligible"]]
        signatures_by_storage = {}
        for record in layer_records:
            signature = json.dumps(record["signature"], separators=(",", ":"))
            signatures_by_storage.setdefault(record["storage_id"], set()).add(signature)
        safe_storages = {storage_id for storage_id, signatures in signatures_by_storage.items() if len(signatures) == 1}
        candidates = [record for record in layer_records if record["storage_id"] in safe_storages]
        candidates.sort(key=lambda record: (-record["nbytes"], record["pack_index"]))
        chosen_storages = []
        for record in candidates:
            if record["storage_id"] in chosen_storages:
                continue
            chosen_storages.append(record["storage_id"])
            if offload_tensors >= 0 and len(chosen_storages) == offload_tensors:
                break
        slot_by_storage = {storage_id: slot for slot, storage_id in enumerate(chosen_storages)}
        for record in records:
            if record["layer_idx"] != layer_idx or record["storage_id"] not in slot_by_storage:
                continue
            record["selected"] = True
            record["slot"] = slot_by_storage[record["storage_id"]]
            record["slot_owner"] = record["pack_index"] == min(
                candidate["pack_index"] for candidate in candidates if candidate["storage_id"] == record["storage_id"]
            )
            selected.add((record["layer_idx"], record["pack_index"], signature_from_json(record["signature"])))
    return {
        "version": 1,
        "layers": layers,
        "skip_last_layers": skip_last_layers,
        "offload_tensors": offload_tensors,
        "records": records,
    }, selected


def manifest_bytes(manifest):
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()


def load_selector(manifest):
    return {
        (record["layer_idx"], record["pack_index"], signature_from_json(record["signature"]))
        for record in manifest["records"]
        if record["selected"]
    }


def _linear_fit(x0, y0, x1, y1):
    if x1 == x0:
        raise ValueError("calibration seqlens must differ")
    slope = (y1 - y0) / (x1 - x0)
    intercept = y0 - slope * x0
    return slope, intercept


def _extrapolate_value(x0, y0, x1, y1, x_target):
    slope, intercept = _linear_fit(x0, y0, x1, y1)
    return slope * x_target + intercept


def extrapolate_manifest(
    manifest_lo, manifest_hi, seq_lo, seq_hi, seq_target, offload_tensors, skip_last_layers, layers, min_offload_bytes
):
    """Rescale a pair of small-seq Discovery manifests to describe seq_target.

    Validated empirically (direct probes on Qwen3-8B): every seq-dependent
    record has exactly one shape dimension equal to the discovery seqlen,
    and shape/stride/nbytes are exact linear functions of seq for every
    record -- including non-contiguous (transposed-view) ones, where naively
    assuming a contiguous layout would be wrong. credit_offload.py's
    tensor_signature() match stays untouched: a mistake here only causes a
    miss (nothing selected), never a wrong match.
    """
    records_lo = {(record["layer_idx"], record["pack_index"]): record for record in manifest_lo["records"]}
    records_hi = {(record["layer_idx"], record["pack_index"]): record for record in manifest_hi["records"]}
    if set(records_lo) != set(records_hi):
        only_lo = sorted(set(records_lo) - set(records_hi))[:8]
        only_hi = sorted(set(records_hi) - set(records_lo))[:8]
        raise RuntimeError(
            f"calibration manifests disagree on recorded tensors: only in lo-seq pass "
            f"{only_lo}, only in hi-seq pass {only_hi}"
        )

    extrapolated = []
    for key in sorted(records_lo):
        lo, hi = records_lo[key], records_hi[key]
        if lo["dtype"] != hi["dtype"]:
            raise RuntimeError(f"dtype changed between calibration seqlens for record {key}")
        shape_lo, shape_hi = lo["shape"], hi["shape"]
        if len(shape_lo) != len(shape_hi):
            raise RuntimeError(f"rank changed between calibration seqlens for record {key}")
        if lo["aliases_param"] != hi["aliases_param"] or lo["aliases_input"] != hi["aliases_input"]:
            raise RuntimeError(f"parameter/input aliasing changed between calibration seqlens for record {key}")
        if shape_lo == shape_hi:
            new_shape = list(shape_lo)
            new_stride = list(lo["stride"])
            new_nbytes = lo["nbytes"]
        else:
            stride_lo, stride_hi = lo["stride"], hi["stride"]
            new_shape = [
                round(_extrapolate_value(seq_lo, a, seq_hi, b, seq_target)) for a, b in zip(shape_lo, shape_hi)
            ]
            new_stride = [
                round(_extrapolate_value(seq_lo, a, seq_hi, b, seq_target)) for a, b in zip(stride_lo, stride_hi)
            ]
            new_nbytes = round(_extrapolate_value(seq_lo, lo["nbytes"], seq_hi, hi["nbytes"], seq_target))
        device, offset = lo["signature"][3], lo["signature"][4]
        signature = signature_to_json((tuple(new_shape), tuple(new_stride), lo["dtype"], device, offset))
        eligible = not lo["aliases_param"] and not lo["aliases_input"] and new_nbytes >= min_offload_bytes
        extrapolated.append(
            {
                "layer_idx": lo["layer_idx"],
                "pack_index": lo["pack_index"],
                "signature": signature,
                "shape": new_shape,
                "stride": new_stride,
                "dtype": lo["dtype"],
                "nbytes": new_nbytes,
                "storage_id": lo["storage_id"],
                "aliases_param": lo["aliases_param"],
                "aliases_input": lo["aliases_input"],
                "eligible": eligible,
                "selected": False,
                "slot": None,
                "slot_owner": False,
            }
        )
    return freeze_manifest(extrapolated, offload_tensors, skip_last_layers, layers)


def _write_manifest(path, manifest):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _read_manifest(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _max_memory(device_api, name):
    try:
        return getattr(device_api, name)() / 1e9
    except Exception:
        return None


def _grad_norm(model):
    import torch

    squares = []
    for parameter in model.parameters():
        if parameter.grad is not None:
            squares.append(parameter.grad.detach().float().square().sum())
    if not squares:
        return 0.0
    return torch.stack(squares).sum().sqrt().item()


class ScopeEntryTimer:
    """Time target-module forward entries with no offload work attached.

    The offload run's own entry intervals absorb credit stalls, so the
    uncontaminated per-layer forward period has to come from a reference arm
    carrying the same hook-installation overhead.
    """

    def __init__(self, accelerator, scopes):
        self.accelerator = accelerator
        self.events = []
        self._hooks = []
        for module in scopes.values():
            self._hooks.append(module.register_forward_pre_hook(self._pre))
            self._hooks.append(module.register_full_backward_pre_hook(lambda module, grad_output: None))
            self._hooks.append(module.register_full_backward_hook(lambda module, grad_input, grad_output: None))

    def _pre(self, module, args):
        event = self.accelerator.Event(enable_timing=True)
        event.record()
        self.events.append(event)

    def intervals_ms(self):
        intervals = []
        for start, end in zip(self.events, self.events[1:]):
            end.synchronize()
            intervals.append(start.elapsed_time(end))
        return intervals

    def reset(self):
        self.events = []

    def remove(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def _discover(args, model, forward_loss, scopes, accelerator):
    discovery = Discovery(model, scopes, accelerator, args.min_offload_bytes)
    records = discovery.run(forward_loss)
    return freeze_manifest(records, args.offload_tensors, args.skip_last_layers, args.layers)


def _discover_via_subprocess(args, manifest_path):
    """Ensure a frozen manifest exists at manifest_path, computed off-process.

    Discovery intentionally keeps every saved-for-backward tensor resident
    for a full, non-offloaded forward pass -- and that residency is never
    released for the rest of a process's life (confirmed empirically: a
    gc.collect()+empty_cache() and a ds_engine.zero_grad() right after
    discovery both left the memory unchanged). A measured arm must never run
    Discovery in its own process, or that residue would sit permanently
    under its real training loop.
    """
    if args.discover_calib_seqs:
        _calibrated_discover_via_subprocess(args, manifest_path)
        return
    script = os.path.abspath(__file__)
    discovery_args = _passthrough(args, manifest_path) + ["--pin", "0", "--discover-only"]
    run_arm_subprocess(script, discovery_args)


def _calibrated_discover_via_subprocess(args, manifest_path):
    """Discover at two cheap seqlens and linearly extrapolate to args.seq.

    Lets Discovery's own full-residency forward -- which must materialize
    every candidate tensor at once, and so scales with seq just like the
    real training loop -- stay cheap even when args.seq itself would OOM a
    direct discovery pass (see extrapolate_manifest).
    """
    seq_lo, seq_hi = args.discover_calib_seqs
    if seq_lo == seq_hi:
        raise ValueError("--discover-calib-seqs requires two distinct seqlens")
    script = os.path.abspath(__file__)
    with tempfile.TemporaryDirectory() as tmpdir:
        path_lo = os.path.join(tmpdir, "calib-lo.json")
        path_hi = os.path.join(tmpdir, "calib-hi.json")
        run_arm_subprocess(script, _passthrough(args, path_lo, seq_override=seq_lo) + ["--pin", "0", "--discover-only"])
        run_arm_subprocess(script, _passthrough(args, path_hi, seq_override=seq_hi) + ["--pin", "0", "--discover-only"])
        manifest_lo = _read_manifest(path_lo)
        manifest_hi = _read_manifest(path_hi)
    if manifest_lo["layers"] != manifest_hi["layers"]:
        raise RuntimeError(
            f"calibration manifests disagree on resolved layer count: "
            f"{manifest_lo['layers']} vs {manifest_hi['layers']}"
        )
    manifest, _ = extrapolate_manifest(
        manifest_lo,
        manifest_hi,
        seq_lo,
        seq_hi,
        args.seq,
        args.offload_tensors,
        args.skip_last_layers,
        manifest_lo["layers"],
        args.min_offload_bytes,
    )
    _write_manifest(manifest_path, manifest)


def run_arm(args):
    if args.mode == "none" and args.pin:
        raise ValueError("--pin has no effect without offload")
    if args.mode == "remat" and not args.model:
        raise ValueError("--mode remat requires --model; the synthetic stack has no gradient checkpointing")
    if args.offload_optimizer and not args.model:
        raise ValueError("--offload-optimizer requires --model")
    if args.steps < 1 or args.warmup < 0:
        raise ValueError("--steps must be positive and --warmup nonnegative")
    if args.skip_last_layers < 1 or args.skip_last_layers >= args.layers:
        raise ValueError("--skip-last-layers must be in [1, layers)")

    # Both arms use the same runtime registration policy. The only experiment
    # variable is whether HostBufferPool invokes pin_memory.
    os.environ["DS_PIN_MEMORY_BACKEND"] = "native"
    os.environ["DS_PIN_MEMORY_REGISTER_DEVICE"] = "1"
    patch_cpp_extension_drop_cxx17()

    import torch
    from deepspeed.accelerator import get_accelerator

    accelerator = get_accelerator()
    if not accelerator.is_available():
        raise RuntimeError("saved activation offload benchmark requires an accelerator device")
    accelerator.set_device(0)
    device = accelerator.current_device_name()
    device_api = accelerator

    # A measured arm never runs Discovery's forward pass itself (see
    # _discover_via_subprocess), and this process must not build its own
    # model/ZeRO engine on the GPU until any such child subprocess has fully
    # exited -- otherwise this process's own resident model and the child's
    # transient full-residency forward would compete for the same GPU memory
    # at the same time, which is strictly worse than running discovery
    # in-process ever was.
    if not args.discover_only:
        if not args.manifest_out:
            raise ValueError("a measured arm (no --discover-only) requires --manifest-out")
        if not os.path.exists(args.manifest_out):
            _discover_via_subprocess(args, args.manifest_out)

    model, forward_loss, transformers_version, model_type, _, ds_engine = _build_recipe(args, device)
    expected_layers = int(getattr(getattr(model, "config", None), "num_hidden_layers", args.layers))
    if args.model and args.layers != expected_layers:
        args.layers = expected_layers
    scopes = _resolve_scopes(model, args.offload_modules, args.layers)

    discovery_peak_gb = None
    if args.discover_only:
        discovered, _ = _discover(args, model, forward_loss, scopes, accelerator)
        # Discovery is a full forward with every activation resident, which is
        # exactly what offload avoids. Its peak must not be reported as the
        # credit-bounded loop's peak.
        discovery_peak_gb = _max_memory(device_api, "max_memory_allocated")
        device_api.reset_peak_memory_stats()
        if not args.manifest_out:
            raise ValueError("--discover-only requires --manifest-out")
        _write_manifest(args.manifest_out, discovered)
        print_arm_result(
            {
                "experiment": "saved_activation_offload_discovery",
                "manifest_out": os.path.abspath(args.manifest_out),
                "manifest_sha256": hashlib.sha256(manifest_bytes(discovered)).hexdigest(),
                "selected_tensors": sum(record["slot_owner"] for record in discovered["records"]),
            }
        )
        return

    manifest = _read_manifest(args.manifest_out)
    selector = load_selector(manifest)
    manifest_hash = hashlib.sha256(manifest_bytes(manifest)).hexdigest()
    selected_records = [record for record in manifest["records"] if record["slot_owner"]]
    selected_bytes = sum(record["nbytes"] for record in selected_records)
    selected_groups = len({record["layer_idx"] for record in selected_records})
    k_effective_min = None
    k_effective_max = None
    if args.mode == "offload":
        expected_groups = args.layers - args.skip_last_layers
        owners_by_layer = {
            layer_idx: sum(record["layer_idx"] == layer_idx for record in selected_records)
            for layer_idx in range(expected_groups)
        }
        # K is a request, not a contract: a fused SwiGLU or a different
        # transformers graph legitimately exposes fewer offloadable storages.
        empty = sorted(layer_idx for layer_idx, count in owners_by_layer.items() if count == 0)
        if empty:
            raise RuntimeError(
                f"frozen manifest selects no offloadable storage in layers {empty}; "
                "check --offload-modules and --min-offload-bytes"
            )
        k_effective_min = min(owners_by_layer.values())
        k_effective_max = max(owners_by_layer.values())
        if args.pin:
            memlock_bytes = resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]
            if memlock_bytes != resource.RLIM_INFINITY and memlock_bytes < selected_bytes:
                raise RuntimeError(
                    f"RLIMIT_MEMLOCK is {memlock_bytes} bytes but this arm pins about "
                    f"{selected_bytes} bytes; raise it with `ulimit -l unlimited`"
                )

    if args.mode == "remat":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        from deepspeed.runtime.activation_checkpointing.offload_activations import CheckpointHiddenStatesOffload

        # The reference panel runs pinned by default, since that is what a
        # practitioner enabling checkpoint offload today would get.
        execution_context = CheckpointHiddenStatesOffload(use_pin_memory=bool(args.pin), use_streams=True)
        engine = None
        entry_timer = None
    elif args.mode == "offload":
        engine = SavedIntermediateOffload(
            model=model,
            scope_modules=scopes,
            selector=selector,
            accelerator=accelerator,
            use_pin_memory=bool(args.pin),
            forward_credits=args.credits,
            backward_credits=args.prefetch,
            min_offload_bytes=args.min_offload_bytes,
            credit_timeout_s=args.credit_timeout_s,
        )
        execution_context = engine
        entry_timer = None
    else:
        engine = None
        execution_context = None
        entry_timer = ScopeEntryTimer(accelerator, scopes)

    # Momentum-free SGD has no persistent optimizer state, keeping this
    # activation-transfer experiment focused on forward/backward memory.
    # Skipped when --offload-optimizer hands optimizer state to ds_engine's
    # own ZeRO-2 CPU AdamW instead.
    optimizer = None if ds_engine is not None else torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.0)
    step_times = []
    metrics = []
    step0_loss = None
    step0_grad_norm = None
    first_loss = None
    first_grad_norm = None
    try:
        for step in range(args.warmup + args.steps):
            if ds_engine is not None:
                ds_engine.zero_grad()
            else:
                optimizer.zero_grad(set_to_none=True)
            if engine is not None:
                engine.reset_step()
                context = engine
            elif execution_context is not None:
                context = execution_context
            else:
                context = noop_saved_tensors()
            if entry_timer is not None:
                entry_timer.reset()
            if step == args.warmup:
                device_api.reset_peak_memory_stats()
            accelerator.synchronize()
            started = time.perf_counter()
            with context:
                loss = forward_loss()
                if ds_engine is not None:
                    ds_engine.backward(loss)
                else:
                    loss.backward()
                if engine is not None:
                    engine.assert_backward_consumed()
            if ds_engine is not None:
                ds_engine.step()
            else:
                optimizer.step()
            accelerator.synchronize()
            elapsed = time.perf_counter() - started
            if step == 0:
                step0_loss = loss.detach().float().item()
                step0_grad_norm = _grad_norm(model)
            if step >= args.warmup:
                step_times.append(elapsed)
                if first_loss is None:
                    first_loss = loss.detach().float().item()
                    first_grad_norm = _grad_norm(model)
                row = {
                    "entry_intervals_ms": entry_timer.intervals_ms() if entry_timer is not None else [],
                    "mlp_spans_ms": [],
                }
                if engine is not None:
                    timings = engine.timing_metrics()
                    row.update(
                        {
                            "fwd_wait": engine.fwd_pool.wait_s,
                            "bwd_wait": engine.bwd_pool.wait_s,
                            "fwd_blocking": engine.fwd_pool.blocking_acquisitions,
                            "bwd_blocking": engine.bwd_pool.blocking_acquisitions,
                            "fwd_max_held": engine.fwd_pool.max_held,
                            "bwd_max_held": engine.bwd_pool.max_held,
                            "fwd_max_bytes": engine.fwd_pool.max_held_bytes,
                            "bwd_max_bytes": engine.bwd_pool.max_held_bytes,
                            "fwd_max_outstanding": engine.fwd_pool.max_outstanding,
                            "bwd_max_outstanding": engine.bwd_pool.max_outstanding,
                            "offloaded_tensors": engine.offloaded_tensors,
                            "offloaded_bytes": engine.offloaded_bytes,
                            "host_enqueue_s": engine.host_enqueue_s,
                            "scope_calls": engine.scope_forward_calls,
                            "d2h_copy_ms": timings["d2h_copy_ms"],
                            "d2h_group_span_ms": timings["d2h_group_span_ms"],
                            "h2d_ms": timings["h2d_ms"],
                            "entry_intervals_ms": timings["entry_intervals_ms"],
                            "mlp_spans_ms": timings["mlp_spans_ms"],
                        }
                    )
                metrics.append(row)
    finally:
        if engine is not None:
            engine.close()
        if entry_timer is not None:
            entry_timer.remove()

    def metric_sum(name):
        return sum(row.get(name, 0) for row in metrics)

    def metric_max(name):
        return max((row.get(name, 0) for row in metrics), default=0)

    def mean(values):
        return sum(values) / len(values) if values else None

    latest = metrics[-1] if metrics else {}
    d2h_copy_ms_total = metric_sum("d2h_copy_ms")
    d2h_group_span_ms_total = metric_sum("d2h_group_span_ms")
    h2d_ms_total = metric_sum("h2d_ms")
    transferred_bytes = metric_sum("offloaded_bytes")
    measured_groups = selected_groups * len(metrics)
    entry_intervals_ms = [interval for row in metrics for interval in row["entry_intervals_ms"]]
    mlp_spans_ms = [span for row in metrics for span in row["mlp_spans_ms"]]
    inter_mlp_entry_mean_ms = mean(entry_intervals_ms)
    mlp_span_mean_ms = mean(mlp_spans_ms)
    d2h_layer_mean_ms = d2h_copy_ms_total / measured_groups if measured_groups else None
    have_ratio_inputs = d2h_layer_mean_ms is not None and mlp_span_mean_ms
    r_d2h_over_mlp_span = d2h_layer_mean_ms / mlp_span_mean_ms if have_ratio_inputs else None
    use_pin = bool(args.pin) if args.mode in ("offload", "remat") else False
    result = {
        "experiment": "saved_activation_offload",
        "mode": args.mode,
        "use_pin_memory": use_pin,
        "pin_backend": "native" if use_pin else None,
        "pin_registered": use_pin,
        "is_pinned_verified": engine.host_pool.is_pinned_verified if engine is not None else None,
        "credits_n": args.credits,
        "prefetch_m": args.prefetch,
        "offload_tensors_k": args.offload_tensors,
        "offload_modules": args.offload_modules,
        "skip_last_layers": args.skip_last_layers,
        "device": accelerator.device_name(),
        "sku": getattr(torch, accelerator.device_name()).get_device_name(),
        "model": args.model or f"synthetic-h{args.hidden}-l{args.layers}",
        "hf_model_type": model_type,
        "transformers": transformers_version,
        "params_b": round(sum(param_numel(parameter) for parameter in model.parameters()) / 1e9, 3),
        "optimizer": "adamw_zero2_cpu" if args.offload_optimizer else "sgd_no_momentum",
        "offload_optimizer": bool(args.offload_optimizer),
        "zero_stage": 2 if args.offload_optimizer else None,
        "layers": args.layers,
        "batch": args.batch,
        "seq": args.seq,
        "tokens_per_step": args.batch * args.seq,
        "step_median_s": statistics.median(step_times),
        "step_p10_s": _percentile(step_times, 0.1),
        "step_p90_s": _percentile(step_times, 0.9),
        "steps": len(step_times),
        "inter_mlp_entry_mean_ms": inter_mlp_entry_mean_ms,
        "inter_mlp_entry_definition": INTER_MLP_ENTRY_DEFINITION,
        "inter_mlp_entry_samples": len(entry_intervals_ms),
        "mlp_span_mean_ms": mlp_span_mean_ms,
        "mlp_span_definition": MLP_SPAN_DEFINITION,
        "t_d_layer_ms": d2h_layer_mean_ms,
        "t_d_definition": T_D_DEFINITION,
        "r_d2h_over_mlp_span": r_d2h_over_mlp_span,
        "t_h2d_layer_ms": h2d_ms_total / measured_groups if measured_groups else None,
        "credit_wait_fwd_ms": metric_sum("fwd_wait") * 1000,
        "credit_wait_bwd_ms": metric_sum("bwd_wait") * 1000,
        "fwd_blocking_acquisitions": metric_sum("fwd_blocking"),
        "bwd_blocking_acquisitions": metric_sum("bwd_blocking"),
        "fwd_max_held": metric_max("fwd_max_held"),
        "fwd_max_held_bytes": metric_max("fwd_max_bytes"),
        "bwd_max_held": metric_max("bwd_max_held"),
        "bwd_max_held_bytes": metric_max("bwd_max_bytes"),
        "fwd_max_outstanding": metric_max("fwd_max_outstanding"),
        "bwd_max_outstanding": metric_max("bwd_max_outstanding"),
        "d2h_copy_ms_total": d2h_copy_ms_total,
        "d2h_group_span_ms_total": d2h_group_span_ms_total,
        "h2d_event_ms_total": h2d_ms_total,
        "d2h_effective_gbps": transferred_bytes / d2h_copy_ms_total / 1e6 if d2h_copy_ms_total else 0.0,
        "h2d_effective_gbps": transferred_bytes / h2d_ms_total / 1e6 if h2d_ms_total else 0.0,
        "host_enqueue_latency_ms": metric_sum("host_enqueue_s") * 1000,
        "hbm_peak_allocated_gb": _max_memory(device_api, "max_memory_allocated"),
        "hbm_peak_reserved_gb": _max_memory(device_api, "max_memory_reserved"),
        "hbm_peak_discovery_gb": discovery_peak_gb,
        "host_offload_bytes": engine.host_pool.allocated_bytes if engine is not None else 0,
        "host_pinned_bytes": engine.host_pool.pinned_bytes if engine is not None else 0,
        "ulimit_l_kb": resource.getrlimit(resource.RLIMIT_MEMLOCK)[0] // 1024,
        "offloaded_tensors": latest.get("offloaded_tensors", 0),
        "offloaded_bytes": latest.get("offloaded_bytes", 0),
        "groups": selected_groups,
        "k_effective_min": k_effective_min,
        "k_effective_max": k_effective_max,
        "manifest_selected_bytes": selected_bytes,
        "manifest_sha256": manifest_hash,
        "scope_forward_calls_per_step": latest.get("scope_calls", args.layers),
        "loss_step0": step0_loss,
        "grad_norm_step0": step0_grad_norm,
        "loss_first_step": first_loss,
        "grad_norm_first_step": first_grad_norm,
        "torch": torch.__version__,
    }
    if args.mode == "offload":
        if not mlp_span_mean_ms or not d2h_layer_mean_ms or not h2d_ms_total:
            raise RuntimeError("accelerator event timing produced no positive MLP-span, D2H, or H2D observation")
        if result["fwd_max_held"] > args.credits or result["bwd_max_held"] > args.prefetch:
            raise RuntimeError("credit cap was exceeded")
        if args.pin and not result["is_pinned_verified"]:
            raise RuntimeError("the pinned arm never confirmed a pinned host buffer")
        if result["offloaded_bytes"] != selected_bytes:
            raise RuntimeError(f"offloaded {result['offloaded_bytes']} bytes, manifest selected {selected_bytes} bytes")
        expected_host_bytes = selected_bytes
        if expected_host_bytes and abs(result["host_offload_bytes"] - expected_host_bytes) / expected_host_bytes > 0.05:
            raise RuntimeError("host allocation differs from the selected manifest by more than 5%")
    print_arm_result(result)


def _passthrough(args, manifest_path, credits=None, seq_override=None):
    result = [
        "--hidden",
        str(args.hidden),
        "--layers",
        str(args.layers),
        "--batch",
        str(args.batch),
        "--seq",
        str(args.seq if seq_override is None else seq_override),
        "--steps",
        str(args.steps),
        "--warmup",
        str(args.warmup),
        "--credits",
        str(args.credits if credits is None else credits),
        "--prefetch",
        str(args.prefetch),
        "--offload-tensors",
        str(args.offload_tensors),
        "--offload-modules",
        args.offload_modules,
        "--min-offload-bytes",
        str(args.min_offload_bytes),
        "--skip-last-layers",
        str(args.skip_last_layers),
        "--credit-timeout-s",
        str(args.credit_timeout_s),
        "--mode",
        args.mode,
        "--manifest-out",
        manifest_path,
    ]
    if args.model:
        result.extend(["--model", args.model])
    if args.offload_optimizer:
        result.append("--offload-optimizer")
    return result


def run_driver(args):
    script = os.path.abspath(__file__)
    cleanup_manifest = args.manifest_out is None
    manifest_path = args.manifest_out
    if manifest_path is None:
        with tempfile.NamedTemporaryFile(prefix="saved-activation-", suffix=".json", delete=False) as handle:
            manifest_path = handle.name
        os.unlink(manifest_path)
    try:
        _discover_via_subprocess(args, manifest_path)
        discovery = {"manifest_sha256": hashlib.sha256(manifest_bytes(_read_manifest(manifest_path))).hexdigest()}
        if args.mode != "offload":
            reference_pin = "1" if args.mode == "remat" else "0"
            result = run_arm_subprocess(script, _passthrough(args, manifest_path) + ["--pin", reference_pin])
            print_driver_result(
                {
                    "experiment": "saved_activation_offload",
                    "manifest_sha256": discovery["manifest_sha256"],
                    "runs": {args.mode: [result]},
                }
            )
            return
        credit_values = (1, 2, 4) if args.n_sweep else (args.credits,)
        all_results = {}
        for credits in credit_values:
            results = []
            for label, pin in (("A", 0), ("B", 1), ("B", 1), ("A", 0)):
                arm = run_arm_subprocess(script, _passthrough(args, manifest_path, credits) + ["--pin", str(pin)])
                arm["arm_order"] = label
                results.append(arm)
            all_results[str(credits)] = results
        primary = all_results[str(args.credits if not args.n_sweep else 2)]
        pageable = [row for row in primary if not row["use_pin_memory"]]
        pinned = [row for row in primary if row["use_pin_memory"]]
        pageable_times = [row["step_median_s"] for row in pageable]
        pinned_times = [row["step_median_s"] for row in pinned]
        pageable_median = statistics.median(pageable_times)
        pinned_median = statistics.median(pinned_times)
        # A pin win is only reportable if the gap between arms is larger than
        # the run-to-run spread within either arm.
        arm_spread_s = max(max(pageable_times) - min(pageable_times), max(pinned_times) - min(pinned_times))
        pin_gap_s = abs(pageable_median - pinned_median)
        print_driver_result(
            {
                "experiment": "saved_activation_offload",
                "manifest_sha256": discovery["manifest_sha256"],
                "runs": all_results,
                "pageable_median_s": pageable_median,
                "pinned_median_s": pinned_median,
                "pin_speedup": pageable_median / pinned_median,
                "pin_gap_s": pin_gap_s,
                "arm_spread_s": arm_spread_s,
                "pin_separated": pin_gap_s > 2 * arm_spread_s,
            }
        )
    finally:
        if cleanup_manifest and os.path.exists(manifest_path):
            os.unlink(manifest_path)


class _FakeEvent:
    def __init__(self, ready=False):
        self.ready = ready

    def query(self):
        return self.ready

    def synchronize(self):
        self.ready = True

    def elapsed_time(self, other):
        if not other.ready:
            raise AssertionError("elapsed_time read before end-event synchronization")
        return 2.5


class _FakePinAccelerator:
    def __init__(self):
        self.unpinned = []

    def pin_memory(self, tensor, make_copy=False):
        return tensor

    def is_pinned(self, tensor):
        return True

    def unpin_memory(self, tensor):
        self.unpinned.append(id(tensor))


class _FailingEvent(_FakeEvent):
    def synchronize(self):
        raise RuntimeError("synthetic event failure")


def run_smoke():
    pool = CreditPool(1, timeout_s=0.03, poll_s=0.001)
    first = pool.acquire("first", 0)
    event = _FakeEvent()
    pool.arm(first, event)
    try:
        pool.acquire("timeout", 1)
    except CreditTimeout:
        pass
    else:
        raise AssertionError("never-recorded event did not time out")
    event.ready = True
    second = pool.acquire("second", 1)
    pool.cancel(second)
    pool.drain()
    if pool.held != 0 or pool.max_held > 1:
        raise AssertionError("credit accounting exceeded its cap")
    if SavedIntermediateOffload._elapsed_ms(_FakeEvent(), _FakeEvent()) != 2.5:
        raise AssertionError("timing event smoke returned an unexpected interval")

    deadlock = CreditPool(1)
    held = deadlock.acquire("held", 0)
    try:
        deadlock.acquire("deadlock", 1)
    except CreditDeadlock:
        pass
    else:
        raise AssertionError("unarmed-credit deadlock was not detected")
    if deadlock.try_acquire("nonblocking", 1) is not None:
        raise AssertionError("try_acquire handed out a credit past the cap")
    deadlock.cancel(held)
    if deadlock.blocking_acquisitions != 0:
        raise AssertionError("an acquire that never waited was counted as blocking")

    def new_generation():
        engine = object.__new__(SavedIntermediateOffload)
        engine._highest_layer = 3
        engine._backward_started = False
        engine._top_hook_seen = False
        engine._backward_generation = 0
        engine._groups_by_layer = {}
        engine._prefetch_order = []
        engine._prefetch_pos = 0
        engine._fill_prefetch_window = lambda: None
        return engine

    generation = new_generation()
    generation._begin_backward(3)
    try:
        generation._begin_backward(3)
    except RuntimeError as exc:
        if "second backward" not in str(exc):
            raise
    else:
        raise AssertionError("second backward generation was not rejected")

    # An unpack that precedes the top module's backward hook must seed the
    # prefetch window rather than turn the real top hook into a second pass.
    unpack_first = new_generation()
    unpack_first._seed_backward()
    unpack_first._begin_backward(3)
    if not unpack_first._backward_started or unpack_first._backward_generation != 1:
        raise AssertionError("unpack-first backward seeding was rejected as a second generation")

    records = [
        {
            "layer_idx": 0,
            "pack_index": pack_index,
            "signature": [[4], [1], "torch.float32", "cuda:0", 0],
            "nbytes": nbytes,
            "storage_id": storage_id,
            "eligible": True,
            "selected": False,
            "slot": None,
            "slot_owner": False,
        }
        for pack_index, storage_id, nbytes in ((0, 10, 16), (1, 20, 32), (2, 20, 32))
    ]
    manifest, selector = freeze_manifest(records, offload_tensors=1, skip_last_layers=1, layers=2)
    if len(selector) != 2 or sum(record["slot_owner"] for record in manifest["records"]) != 1:
        raise AssertionError("repeated selected storage was not mapped to one frozen slot")

    accelerator = _FakePinAccelerator()
    host_pool = HostBufferPool(accelerator, pinned=True)
    source = torch.empty(4)
    pending = host_pool.get_dense(source)
    host_pool.release(pending, _FakeEvent())
    checked_out = host_pool.get_dense(source)
    host_pool.close()
    if sorted(accelerator.unpinned) != sorted((id(pending), id(checked_out))):
        raise AssertionError("close did not unpin both pending and checked-out allocations")

    accelerator = _FakePinAccelerator()
    host_pool = HostBufferPool(accelerator, pinned=True)
    pending = host_pool.get_dense(source)
    host_pool.release(pending, _FailingEvent())
    try:
        host_pool.close()
    except RuntimeError as exc:
        if "synthetic event failure" not in str(exc):
            raise
    else:
        raise AssertionError("host pool suppressed an event cleanup failure")
    if accelerator.unpinned != [id(pending)]:
        raise AssertionError("host pool did not unpin after an event cleanup failure")

    host_pool = HostBufferPool(_FakePinAccelerator(), pinned=False)
    if host_pool.is_pinned_verified is not None:
        raise AssertionError("the pageable arm reported a pin verification it never made")
    host = host_pool.get_dense(source)
    group = Credit(tag="fwd:L0", layer_idx=0, id=1, host_bufs=[host], d2h_event=_FakeEvent(ready=True))
    cleanup = object.__new__(SavedIntermediateOffload)
    cleanup.host_pool = host_pool
    cleanup.s_copy = None
    cleanup._groups_by_id = {group.id: group}
    cleanup._release_group_hosts()
    host_pool.drain()
    if not group.host_released or group.host_bufs or host_pool._checked_out:
        raise AssertionError("D2H-only group host buffers were not reclaimed")

    consumed = object.__new__(SavedIntermediateOffload)
    consumed._groups_by_id = {group.id: group}
    try:
        consumed.assert_backward_consumed()
    except RuntimeError as exc:
        if "not reloaded and consumed" not in str(exc):
            raise
    else:
        raise AssertionError("a group that never reached backward was reported as consumed")

    if has_internal_overlap(torch.zeros(4).expand(3, 4)) is not True:
        raise AssertionError("an expanded tensor was not reported as internally overlapping")
    if has_internal_overlap(torch.zeros(4)):
        raise AssertionError("a dense tensor was reported as internally overlapping")
    print("saved_activation_offload smoke: PASS")


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.smoke:
        run_smoke()
    elif parsed.pin is None:
        run_driver(parsed)
    else:
        run_arm(parsed)
