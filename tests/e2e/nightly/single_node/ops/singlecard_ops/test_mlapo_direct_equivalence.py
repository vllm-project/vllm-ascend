#!/usr/bin/env python3
"""Exact original-MLAPO/Direct checks on real GLM W8A8 layer weights.

Checkpoint slicing follows the production MLAPO weight transforms. Both
operators keep the original fixed epsilon=1e-6. This is
an operator consistency test, not a performance or accuracy-dataset test.
CPU pools use the production memfabric SHARED allocation, not mocked/pinned
torch tensors. A separate NPU stream reads them via offload.sparse_copy.

Run explicitly with the model's NPU Python environment::

    python tests/e2e/nightly/single_node/ops/singlecard_ops/test_mlapo_direct_equivalence.py \
        --model-path /data/weights/GLM-5.2-w8a8 --output /actual/results.json

The pytest entry uses the pre-provisioned ModelScope cache for
Eco-Tech/GLM-5.2-w8a8, never downloads weights, and fails if it is missing.
Nightly retains its JSON evidence under /tmp/test-logs/mlapo-direct/.

This covers in-place scale updates with inference paused. Replacing graph-bound
scale or weight buffers requires recapture.
"""

import argparse
import gc
import hashlib
import json
import socket
import subprocess
import tempfile
import traceback
from pathlib import Path

import torch
import torch_npu
from memfabric_hybrid import offload  # type: ignore[import-untyped]

import vllm_ascend
import vllm_ascend.ops  # noqa: F401 - initialize the registry before attention backends
from vllm_ascend.attention.utils import trans_rope_weight, transdata
from vllm_ascend.compilation.updatable_graph import UpdatableGraph
from vllm_ascend.utils import enable_custom_op

OUTPUTS = ("q_nope", "q_rope", "q_c", "current_k", "current_rope")
SENTINEL = -17.5


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact(actual, expected, label):
    actual, expected = actual.detach().cpu().contiguous(), expected.detach().cpu().contiguous()
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, label
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), (label, "non-finite")
    a, b = actual.view(torch.uint8), expected.view(torch.uint8)
    different = int((a != b).sum())
    assert different == 0, (label, "different_bytes", different)
    return {
        "shape": list(actual.shape),
        "different_bytes": different,
        "sha256": hashlib.sha256(a.numpy().tobytes()).hexdigest(),
    }


def weights_for_heads(root, layer, heads, device, tp_rank=0):
    """Apply the production weight transforms to one real checkpoint layer."""
    from safetensors import safe_open

    raw_config = json.loads((root / "config.json").read_text())
    cfg = raw_config.get("text_config", raw_config)
    expected = {
        "hidden_size": 6144,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "num_attention_heads": 64,
    }
    for key, expected_value in expected.items():
        assert cfg.get(key) == expected_value, (key, cfg.get(key), expected_value)
    assert heads in (16, 32)
    assert 0 <= tp_rank < 64 // heads, (tp_rank, heads)
    d = {
        "hidden": 6144,
        "q": 2048,
        "kv": 512,
        "nope": 192,
        "rope": 64,
        "heads": heads,
        "value": int(cfg.get("v_head_dim", 256)),
    }
    indices = list(root.glob("*.safetensors.index.json"))
    assert len(indices) == 1, ("expected one checkpoint index", indices)
    mapping = json.loads(indices[0].read_text())["weight_map"]
    suffix = f"layers.{layer}.self_attn.q_a_layernorm.weight"
    matches = [name.removesuffix(".q_a_layernorm.weight") for name in mapping if name.endswith(suffix)]
    assert len(matches) == 1, matches
    prefix, loaded = matches[0], []

    def read(name, rows=None, optional=False):
        key = f"{prefix}.{name}"
        if optional and key not in mapping:
            return None
        path = (root / mapping[key]).resolve()
        assert path.is_relative_to(root), path
        with safe_open(str(path), framework="pt", device="cpu") as shard:
            value = shard.get_tensor(key) if rows is None else shard.get_slice(key)[rows]
        value = value.contiguous()
        loaded.append(
            {
                "name": key,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
            }
        )
        return value

    fused = f"{prefix}.fused_qkv_a_proj.weight" in mapping
    w = {}
    for dest, name in (("a", "weight"), ("ad", "deq_scale"), ("ab", "quant_bias")):
        w[dest] = (
            read(f"fused_qkv_a_proj.{name}")
            if fused
            else torch.cat((read(f"q_a_proj.{name}"), read(f"kv_a_proj_with_mqa.{name}")))
        )
    for dest, name in (("as", "input_scale"), ("ao", "input_offset")):
        w[dest] = read(f"{'fused_qkv_a_proj' if fused else 'q_a_proj'}.{name}").reshape(-1)
        if not fused:
            assert torch.equal(w[dest], read(f"kv_a_proj_with_mqa.{name}").reshape(-1)), name
    first_head, last_head = tp_rank * heads, (tp_rank + 1) * heads
    rows = slice(first_head * (d["nope"] + d["rope"]), last_head * (d["nope"] + d["rope"]))
    for dest, name in (("b", "weight"), ("bd", "deq_scale"), ("bb", "quant_bias")):
        w[dest] = read(f"q_b_proj.{name}", rows)
    w["bs"], w["bo"] = read("q_b_proj.input_scale").reshape(-1), read("q_b_proj.input_offset").reshape(-1)
    w["gamma1"], w["gamma2"] = read("q_a_layernorm.weight"), read("kv_a_layernorm.weight")
    w["beta1"] = read("q_a_layernorm.bias", optional=True)
    if w["beta1"] is None:
        w["beta1"] = torch.zeros(d["q"], dtype=torch.bfloat16)
    kv_b = read("kv_b_proj.weight", slice(first_head * (d["nope"] + d["value"]), last_head * (d["nope"] + d["value"])))
    assert kv_b.dtype == torch.bfloat16
    w["uk"] = kv_b.reshape(heads, d["nope"] + d["value"], d["kv"])[:, : d["nope"]].contiguous()
    for name in ("ao", "bo"):
        value = w[name].float()
        assert torch.isfinite(value).all() and (value == value.round()).all(), name
        assert ((value >= -128) & (value <= 127)).all(), name
        w[name] = w[name].to(torch.int8)
    for name in ("as", "bs", "gamma1", "beta1", "gamma2"):
        w[name] = w[name].to(torch.bfloat16)
    for name in ("ad", "bd", "ab", "bb"):
        w[name] = w[name].reshape(-1)
    assert w["a"].dtype == w["b"].dtype == torch.int8
    assert w["ad"].dtype == w["bd"].dtype == torch.float32
    assert w["ab"].dtype == w["bb"].dtype == torch.int32
    for name in ("as", "bs"):
        assert w[name].numel() == 1 and (w[name] > 0).all(), name
    w = {key: value.to(device).contiguous() for key, value in w.items()}
    a = torch.cat((trans_rope_weight(w["a"][d["q"] :], d["rope"]), w["a"][: d["q"]]))
    b = trans_rope_weight(w["b"].view(heads, d["nope"] + d["rope"], d["q"]), d["rope"]).reshape_as(w["b"])
    w["ma"] = torch_npu.npu_format_cast(transdata(a, (16, 32)).unsqueeze(0).contiguous(), 29)
    w["mb"] = torch_npu.npu_format_cast(transdata(b, (16, 32)).unsqueeze(0).contiguous(), 29)
    for original, transformed in (("ad", "mad"), ("ab", "mab")):
        value = w[original].reshape(-1, 1)
        w[transformed] = (
            torch.cat((trans_rope_weight(value[d["q"] :], d["rope"]), value[: d["q"]])).flatten().contiguous()
        )
    for original, transformed in (("bd", "mbd"), ("bb", "mbb")):
        w[transformed] = (
            trans_rope_weight(w[original].view(heads, d["nope"] + d["rope"], 1), d["rope"]).flatten().contiguous()
        )
    w["uk"] = torch_npu.npu_format_cast(w["uk"], 2).contiguous()
    w["one"] = torch.ones(1, device=device, dtype=torch.bfloat16)
    # Both operators use the original scalar tensors, with no padded copy.
    return (
        d,
        w,
        {
            "prefix": prefix,
            "heads": heads,
            "tp": 64 // heads,
            "tp_rank": tp_rank,
            "config_sha256": sha256(root / "config.json"),
            "model_epsilon": cfg.get("rms_norm_eps"),
            "operator_epsilon": 1e-6,
            "weights": loaded,
        },
    )


def original(x, cos, sin, slots, out, w):
    torch.ops._C_ascend.mla_preprocess(
        x,
        w["ma"],
        w["mad"],
        w["gamma1"],
        w["beta1"],
        w["mb"],
        w["mbd"],
        w["gamma2"],
        cos,
        sin,
        w["uk"],
        out["cache_k"],
        out["cache_rope"],
        slots,
        quant_scale0=w["as"],
        quant_offset0=w["ao"],
        bias0=w["mab"],
        quant_scale1=w["bs"],
        quant_offset1=w["bo"],
        bias1=w["mbb"],
        ctkv_scale=w["one"],
        q_nope_scale=w["one"],
        cache_mode="krope_ctkv",
        quant_mode="per_tensor_quant_asymm",
        enable_inner_out=True,
        q_out0=out["q_nope"],
        q_out1=out["q_rope"],
        inner_out=out["q_c"],
        kv_cache_out0=out["cache_k"],
        kv_cache_out1=out["cache_rope"],
    )


def direct(x, cos, sin, out, w, pool=None, slots=None):
    kwargs = {}
    if pool is not None:
        kwargs = {
            "writeback_slots": slots,
            "writeback_k_base": pool[0].data_ptr(),
            "writeback_rope_base": pool[1].data_ptr(),
            "writeback_capacity": pool[0].shape[0],
        }
    torch.ops._C_ascend.mla_preprocess_with_kv_offload(
        hiddenState=x,
        wdqkv=w["ma"],
        descale0=w["mad"],
        gamma1=w["gamma1"],
        beta1=w["beta1"],
        wuq=w["mb"],
        descale1=w["mbd"],
        gamma2=w["gamma2"],
        cos=cos.reshape(x.shape[0], -1),
        sin=sin.reshape(x.shape[0], -1),
        wuk=w["uk"],
        quant_scale0=w["as"],
        quant_offset0=w["ao"],
        bias0=w["mab"],
        quant_scale1=w["bs"],
        quant_offset1=w["bo"],
        bias1=w["mbb"],
        ctkv_scale=w["one"],
        q_nope_scale=w["one"],
        quant_mode="per_tensor_quant_asymm",
        q_out0=out["q_nope"],
        q_out1=out["q_rope"],
        inner_out=out["q_c"],
        current_k=out["current_k"],
        current_rope=out["current_rope"],
        **kwargs,
    )


def new_outputs(n, d, device, paged=False):
    shapes = {
        "q_nope": (n, d["heads"], d["kv"]),
        "q_rope": (n, d["heads"], d["rope"]),
        "q_c": (n, d["q"]),
        "current_k": (n, d["kv"]),
        "current_rope": (n, d["rope"]),
    }
    backing = {
        name: torch.full((shape[0] + 2, *shape[1:]), SENTINEL, device=device, dtype=torch.bfloat16)
        for name, shape in shapes.items()
    }
    out = {name: value[1:-1] for name, value in backing.items()}
    if paged:
        for name, width in (("k", d["kv"]), ("rope", d["rope"])):
            out[f"cache_{name}"] = torch.full((2, 128, 1, width), SENTINEL, device=device, dtype=torch.bfloat16)
            out[f"current_{name}"] = out[f"cache_{name}"].view(-1, width)[:n]
    return out, backing


def shape_case(n, d, w, device):
    scale_addresses = [w[name].data_ptr() for name in ("as", "bs")]
    x = torch.empty(n, d["hidden"], dtype=torch.bfloat16, device=device)
    cos = torch.empty(n, 1, 1, d["rope"], dtype=x.dtype, device=device)
    sin = torch.empty_like(cos)
    compact = torch.arange(n, dtype=torch.int32, device=device)
    slots = torch.empty(n, dtype=torch.int32, device=device)
    old, old_guards = new_outputs(n, d, device, paged=True)
    out, guards = new_outputs(n, d, device)
    pools = tuple(offload.empty([128, width], dtype=torch.bfloat16, pin_memory=True) for width in (512, 64))
    readback = tuple(torch.empty(n, width, dtype=torch.bfloat16, device=device) for width in (512, 64))
    src = torch.empty(2 * n, dtype=torch.int64, device=device)
    dst = torch.tensor(
        [value.data_ptr() + row * width * 2 for value, width in zip(readback, (512, 64)) for row in range(n)],
        dtype=torch.int64,
        device=device,
    )
    sizes = torch.tensor([width * 2 for width in (512, 64) for _ in range(n)], dtype=torch.int32, device=device)
    count = torch.tensor(2 * n, dtype=torch.int32, device=device)
    consumer = torch.npu.Stream()
    producer_done = torch.npu.Event()
    slot_ids: list[int] = []

    def update(iteration):
        # Change activations, RoPE AND business slots between graph replays.
        torch.manual_seed(20261023 + iteration)
        x.copy_(torch.randn(x.shape, dtype=x.dtype, device=device) * (1.0 if iteration % 2 == 0 else 1e-3))
        angle = (
            (torch.arange(n, device=device, dtype=torch.float32).view(n, 1) + iteration + 1)
            * torch.arange(1, 33, device=device, dtype=torch.float32).view(1, 32)
            / 100
        )
        cos.copy_(torch.cat((angle.cos(), angle.cos()), -1).to(x.dtype).view_as(cos))
        sin.copy_(torch.cat((angle.sin(), angle.sin()), -1).to(x.dtype).view_as(sin))
        slot_ids[:] = [3 + iteration % 2 + row * 2 for row in range(n)]
        slots.copy_(torch.tensor(slot_ids, dtype=slots.dtype, device=device))
        src.copy_(
            torch.tensor(
                [pool.data_ptr() + slot * width * 2 for pool, width in zip(pools, (512, 64)) for slot in slot_ids],
                dtype=torch.int64,
                device=device,
            )
        )
        for pool in pools:
            pool.fill_(SENTINEL)
        for value in (*guards.values(), *old_guards.values(), old["cache_k"], old["cache_rope"]):
            value.fill_(SENTINEL)
        torch.npu.synchronize()

    def launch(fused=True):
        original(x, cos, sin, compact, old, w)
        direct(x, cos, sin, out, w, pools if fused else None, slots if fused else None)
        if fused:
            producer = torch.npu.current_stream()
            producer_done.record(producer)
            consumer.wait_event(producer_done)
            with torch.npu.stream(consumer):
                rc = offload.sparse_copy(src, dst, sizes, count, x.device)
                assert rc in (None, 0), rc
            # Join the actual CPU-pool reader into the graph before capture ends.
            producer.wait_stream(consumer)

    def check(fused, label):
        torch.npu.synchronize()
        values = {}
        for name in OUTPUTS:
            assert (old[name] != SENTINEL).any(), (label, name, "unwritten original output")
            values[name] = exact(out[name], old[name], (label, name))
        for name, value in guards.items():
            assert (value[0] == SENTINEL).all() and (value[-1] == SENTINEL).all(), (label, name, "guard")
        for name in ("q_nope", "q_rope", "q_c"):
            value = old_guards[name]
            assert (value[0] == SENTINEL).all() and (value[-1] == SENTINEL).all(), (label, name, "original guard")
        assert (old["cache_k"].view(-1, 512)[n:] == SENTINEL).all(), (label, "original K guard")
        assert (old["cache_rope"].view(-1, 64)[n:] == SENTINEL).all(), (label, "original RoPE guard")
        if fused:
            for pool, value, read, name in zip(
                pools, (out["current_k"], out["current_rope"]), readback, ("current_k", "current_rope")
            ):
                expected = torch.full_like(pool, SENTINEL)
                expected[slot_ids] = value.cpu()
                exact(pool, expected, (label, name, "CPU pool including untouched slots"))
                values[name + "_cpu_readback"] = exact(read, old[name], (label, name, "CPU->NPU consumer"))
        return {
            "label": label,
            "heads": d["heads"],
            "tokens": n,
            "slots": list(slot_ids),
            "fused_cpu_writeback": fused,
            "outputs": values,
        }

    rows = []
    update(0)
    launch(False)
    rows.append(check(False, "eager_direct"))
    launch(True)
    rows.append(check(True, "eager_fused_writeback"))
    streams = (torch.npu.Stream(), torch.npu.Stream())
    graphs = []
    for index, stream in enumerate(streams):
        graph = UpdatableGraph()
        context = torch.npu.graph(graph, stream=stream, capture_error_mode="global")
        # Eager calls above initialized ACLNN; no legacy manual tiling cache
        # or prepare_tiling operator is part of the main-branch ABI.
        with context:
            launch(True)
        graphs.append(graph)
        rows.append(check(True, f"global_capture_stream_{index}"))
    fingerprints = []
    for iteration in (1, 2, 3):
        update(iteration)
        # Replay alternates graph and launch stream; no stream-local stale tiling.
        graph_index = iteration % 2
        with torch.npu.stream(streams[1 - graph_index]):
            graphs[graph_index].replay()
        row = check(True, f"global_replay_{iteration}_graph_{graph_index}")
        # Fresh eager original prevents two captured stale outputs passing together.
        original(x, cos, sin, compact, old, w)
        torch.npu.synchronize()
        for name in OUTPUTS:
            exact(out[name], old[name], (row["label"], name, "fresh original"))
        fingerprints.append(row["outputs"]["q_nope"]["sha256"])
        rows.append(row)
    assert len(set(fingerprints)) > 1, "different replay inputs did not change Q output"

    # Freeze inputs while updating the original scales in place. Both captured
    # operators must consume the new values at the same graph-bound addresses.
    update(0)
    graphs[0].replay()
    before = check(True, "scale_update_before")
    unchanged_inputs = {name: value.cpu().clone() for name, value in (("x", x), ("cos", cos), ("sin", sin))}
    previous_scales = (w["as"].clone(), w["bs"].clone())
    try:
        for name, previous, factor in zip(("as", "bs"), previous_scales, (1.5, 0.75)):
            w[name].mul_(factor)
            assert not torch.equal(w[name].cpu(), previous.cpu()), ("scale value did not change", name)
        assert scale_addresses == [w[name].data_ptr() for name in ("as", "bs")], "scale address changed"
        torch.npu.synchronize()
        for mode, graph in (("eager", None), ("global_graph_0", graphs[0]), ("global_graph_1", graphs[1])):
            update(0)
            for name, value in (("x", x), ("cos", cos), ("sin", sin)):
                exact(value, unchanged_inputs[name], ("scale-only update retained inputs", name))
            if graph is None:
                launch(True)
            else:
                graph.replay()
                torch.npu.synchronize()
                # Compare against a fresh original call as well as the captured
                # call so two stale graph outputs cannot pass together.
                original(x, cos, sin, compact, old, w)
            row = check(True, f"scale_update_{mode}")
            assert any(row["outputs"][name]["sha256"] != before["outputs"][name]["sha256"] for name in OUTPUTS), (
                "changed source scales did not change any output"
            )
            row["scale_addresses"] = scale_addresses
            row["source_scale_values"] = [float(w[name].cpu().item()) for name in ("as", "bs")]
            rows.append(row)
    finally:
        for name, previous in zip(("as", "bs"), previous_scales):
            w[name].copy_(previous)
        torch.npu.synchronize()
    return rows


def loaded_libraries():
    """Record the actual extension/op-api mappings and installed MLAPO artifacts."""
    paths = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if any(x in line for x in ("vllm_ascend_C", "libcust_opapi", "liboptiling")) and ".so" in line
        }
    )
    assert any("vllm_ascend_C" in path for path in paths), ("loaded extension missing", paths)
    assert any("libcust_opapi" in path for path in paths), ("loaded ACLNN op-api missing", paths)
    vendor = Path(vllm_ascend.__file__).resolve().parent / "_cann_ops_custom/vendors/custom_transformer"
    artifacts = sorted(
        path
        for path in vendor.rglob("*")
        if path.is_file()
        and "mlapreprocess" in str(path).lower().replace("_", "")
        and path.suffix in (".so", ".o", ".json")
    )
    assert artifacts, ("installed MLAPO vendor artifacts missing", vendor)

    def describe(path):
        return {"path": str(path), "sha256": sha256(path), "bytes": Path(path).stat().st_size}

    return {
        "mapped_libraries": [describe(path) for path in paths],
        "installed_mlapo_artifacts": [describe(path) for path in artifacts],
    }


def runtime_provenance():
    """Identify the checkout and imported packages separately from binary hashes."""
    import vllm

    root = Path(__file__).resolve().parents[6]
    result = {"test_checkout": str(root), "packages": {}}
    for name, module in (("vllm", vllm), ("vllm_ascend", vllm_ascend)):
        package_path = Path(module.__file__).resolve()
        package = {"path": str(package_path), "version": getattr(module, "__version__", None)}
        result["packages"][name] = package
        try:
            package["commit"] = subprocess.check_output(
                ["git", "-C", str(package_path.parent), "rev-parse", "HEAD"], text=True, timeout=10
            ).strip()
            package["git_status"] = subprocess.check_output(
                ["git", "-C", str(package_path.parent), "status", "--porcelain"], text=True, timeout=10
            ).splitlines()
        except (OSError, subprocess.SubprocessError) as error:
            package["git_error"] = str(error)
    result["commit"] = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=10
    ).strip()
    result["git_status"] = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True, timeout=10
    ).splitlines()
    return result


def resolve_nightly_weights():
    """Require an existing cache; never download a full model for an operator test."""
    from modelscope import snapshot_download

    model = "Eco-Tech/GLM-5.2-w8a8"
    try:
        root = Path(snapshot_download(model, local_files_only=True)).resolve()
    except Exception as error:
        raise RuntimeError(f"Pre-provision the {model} ModelScope cache before running MLAPO nightly") from error
    if not (root / "config.json").is_file() or len(list(root.glob("*.safetensors.index.json"))) != 1:
        raise RuntimeError(f"Incomplete MLAPO nightly checkpoint: {root}")
    return root


def nightly_result_dir(label):
    root = Path("/tmp/test-logs/mlapo-direct") / label
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="run-", dir=root))


def test_mlapo_direct_equivalence():
    output = nightly_result_dir("single-device") / "result.json"
    main(["--model-path", str(resolve_nightly_weights()), "--output", str(output), "--tokens", "1", "6", "32"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--heads", nargs="+", type=int, choices=(16, 32), default=[16, 32])
    parser.add_argument("--tokens", nargs="+", type=int, choices=(1, 6, 32), default=[1, 6])
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args(argv)
    assert not args.output.exists(), ("refusing to overwrite result", args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "failed",
        "test_sha256": sha256(__file__),
        "cases": [],
        "weight_provenance": [],
        "scope": (
            "main ACLNN original MLAPO vs Direct exactness; "
            "real mapped CPU write/read; no performance or dataset accuracy"
        ),
    }
    initialized = False
    try:
        result["runtime"] = runtime_provenance()
        torch.npu.set_device(args.device)
        torch.npu.config.allow_internal_format = True
        enable_custom_op()
        for name in ("mla_preprocess", "mla_preprocess_with_kv_offload"):
            assert hasattr(torch.ops._C_ascend, name), name
        schema = str(torch.ops._C_ascend.mla_preprocess_with_kv_offload.default._schema)
        assert "rmsnorm_epsilon" not in schema and "writeback_slots" in schema, schema
        original_schema = str(torch.ops._C_ascend.mla_preprocess.default._schema)
        assert "rmsnorm_epsilon" not in original_schema, original_schema
        result["schemas"] = {"original": original_schema, "direct": schema}
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        cfg = offload.OffloadConfig()
        cfg.store_url = f"tcp://127.0.0.1:{port}"
        cfg.device_id, cfg.reserve_size, cfg.alloc_size = args.device, 1 << 30, 1 << 30
        cfg.world_size, cfg.rank_id, cfg.scene = 1, 0, offload.Scene.SHARED
        rc = offload.initialize(cfg)
        assert rc == 0, ("offload.initialize", rc)
        initialized = True
        with torch.inference_mode():
            for heads in args.heads:
                d, w, provenance = weights_for_heads(args.model_path.resolve(), args.layer, heads, f"npu:{args.device}")
                result["weight_provenance"].append(provenance)
                for tokens in args.tokens:
                    result["active_case"] = {"heads": heads, "tokens": tokens}
                    rows = shape_case(tokens, d, w, f"npu:{args.device}")
                    result["cases"].extend(rows)
                    print(
                        json.dumps(
                            {
                                "event": "shape_pass",
                                "heads": heads,
                                "tokens": tokens,
                                "checks": len(rows),
                                "outputs": list(OUTPUTS),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    gc.collect()
                del w
        assert len(result["cases"]) == len(args.heads) * len(args.tokens) * 10
        result.pop("active_case")
        result["libraries"] = loaded_libraries()
        result["status"] = "passed"
    except BaseException:
        result["error"] = traceback.format_exc()
        raise
    finally:
        try:
            if initialized:
                torch.npu.synchronize()
                gc.collect()
                offload.uninitialize()
        except BaseException:
            result["status"] = "failed"
            result["cleanup_error"] = traceback.format_exc()
            raise
        finally:
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {"event": "result", "status": result["status"], "cases": len(result["cases"]), "output": str(args.output)},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
