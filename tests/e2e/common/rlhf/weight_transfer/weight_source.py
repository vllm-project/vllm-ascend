# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Deterministic checkpoint payloads for weight-transfer tests.

Two sources share this module, the seeding scheme and the tensor sampler:

``FixedRandomWeightSource``
    Enumerates a Transformers meta model.  That works for every architecture
    Transformers knows, which is what makes the matrix download-free.

``DeepseekV41WeightSource``
    DeepSeek-V4.1 has no Transformers implementation (``AutoConfig`` raises
    ``KeyError: 'deepseek_v41'``), so there is no meta model to enumerate.  It
    derives the payload geometry from the published checkpoint's **own
    safetensors headers** instead: a safetensors file starts with a
    ``<u64 header_len>`` followed by a JSON blob of every ``name -> dtype,
    shape`` in that shard, so an HTTP range request for the leading bytes yields
    the exact checkpoint-facing names and shapes without downloading a weight.
    Headers are cached under ``DSV41_HEADER_CACHE``.
"""

import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import struct
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from vllm.distributed.weight_transfer.base import ParamMeta, WeightSource

from tests.e2e.common.rlhf.weight_transfer.case import WeightUpdateModelCase


def expand_fused_expert_params(name: str, shape: tuple[int, ...]) -> list[tuple[str, tuple[int, ...]]]:
    """Map HF fused routed-expert tensors onto checkpoint-facing per-expert names.

    ``GlmMoeDsaForCausalLM`` keeps routed experts fused as
    ``mlp.experts.gate_up_proj`` ``[E, 2I, H]`` and ``mlp.experts.down_proj``
    ``[E, H, I]``, while the published checkpoint - and therefore vLLM's
    ``expert_params_mapping`` - stores one tensor per expert
    (``mlp.experts.{e}.gate_proj.weight`` etc.).  Without this expansion the
    fused names reach ``load_weights`` unmatched and raise
    ``KeyError: 'layers.N.mlp.experts.gate_up_proj'``.
    """
    for suffix, kind in ((".mlp.experts.gate_up_proj", "gate_up"), (".mlp.experts.down_proj", "down")):
        if not name.endswith(suffix):
            continue
        prefix = name[: -len(suffix.rsplit(".", 1)[-1])]  # keeps the trailing dot
        if kind == "gate_up":
            num_experts, fused_inter, hidden = shape
            inter = fused_inter // 2
            out: list[tuple[str, tuple[int, ...]]] = []
            for expert in range(num_experts):
                out.append((f"{prefix}{expert}.gate_proj.weight", (inter, hidden)))
                out.append((f"{prefix}{expert}.up_proj.weight", (inter, hidden)))
            return out
        num_experts, hidden, inter = shape
        return [(f"{prefix}{expert}.down_proj.weight", (hidden, inter)) for expert in range(num_experts)]
    return [(name, shape)]


def resize_expert_shape(name: str, shape: tuple[int, ...], intermediate_size: int | None) -> tuple[int, ...]:
    """Force expert MLP tensors to the checkpoint's intermediate dimension.

    ``DeepSeek-V4`` builds routed and shared experts from ``intermediate_size``
    in HF's ``from_config`` path but stores ``moe_intermediate_size`` in the
    checkpoint, so the payload shape has to be corrected or vLLM's loader trips
    ``assert args[0].numel() == args[1].numel()`` while copying an expert shard.
    """
    if intermediate_size is None:
        return shape
    if name.endswith((".mlp.shared_experts.gate_proj.weight", ".mlp.shared_experts.up_proj.weight")):
        return (intermediate_size, shape[-1])
    if name.endswith(".mlp.shared_experts.down_proj.weight"):
        return (shape[0], intermediate_size)
    if ".mlp.experts." in name and name.endswith((".gate_proj.weight", ".up_proj.weight")):
        return (intermediate_size, shape[-1])
    if ".mlp.experts." in name and name.endswith(".down_proj.weight"):
        return (shape[0], intermediate_size)
    return shape


def _apply_hf_overrides(config, overrides: dict[str, Any]) -> None:
    for name, value in overrides.items():
        current = getattr(config, name, None)
        if isinstance(value, dict) and current is not None:
            current.update(value)
        else:
            setattr(config, name, value)


# One seed for both sources: the payload has to be reproducible across lanes and
# runs, and the two sources never share a lane, so a single constant is enough.
FIXED_WEIGHT_SEED = 20260915


def _parameter_seed(name: str) -> int:
    """Stable per-name seed, so a payload does not depend on iteration order."""
    digest = hashlib.sha256(f"{FIXED_WEIGHT_SEED}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _sampled_tensor(
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    *,
    num_experts: int,
    sample_on_cpu: bool,
    norm_suffixes: tuple[str, ...],
    integer_topk_names: tuple[str, ...] | None,
) -> torch.Tensor:
    """Deterministically sample one payload tensor from its name.

    ``sample_on_cpu`` is not cosmetic.  The CPU and NPU generators do not produce
    the same stream for one seed - verified on this host: ``identical=False``
    with a max absolute difference equal to the whole sampling range - so a
    source whose payload is compared against a startup load produced in another
    process has to pin *where* it samples.  Sources that only ever materialise on
    the device sample there instead, which avoids a host round trip.

    ``integer_topk_names=None`` means "every integer tensor is a (tokens, top_k)
    router table"; otherwise only tensors with one of those name suffixes are,
    and the rest get a plain bit table.
    """
    seed = _parameter_seed(name)
    torch.manual_seed(seed)
    if not sample_on_cpu:
        torch.npu.manual_seed(seed)
    sample_device = torch.device("cpu") if sample_on_cpu else device

    if not dtype.is_floating_point:
        assert num_experts > 0, f"{name}: integer table needs n_routed_experts"
        if integer_topk_names is None:
            assert len(shape) == 2, f"{name}: expected a (tokens, top_k) integer table, got {shape}"
            topk = True
        else:
            topk = len(shape) == 2 and name.endswith(integer_topk_names)
        if topk:
            # Token -> expert tables (DeepSeek-V4's hash router).  Values must stay
            # inside [0, n_routed_experts) *and* be unique within a row: the MC2
            # dispatch/combine kernels assume a token never routes to the same
            # expert twice, which is also why the served model initialises this
            # table duplicate-free (#16485).  Sampling top-k over per-expert
            # scores guarantees both invariants while still differing from the
            # model's own table.
            scores = torch.rand(shape[0], max(num_experts, shape[1]), device=sample_device)
            return scores.topk(shape[1], dim=1).indices.to(dtype).to(device)
        return torch.randint(0, 2, shape, dtype=dtype, device=sample_device).to(device)

    # Sample in fp32 on the CPU (one code path for every float storage dtype)
    # or directly in the storage dtype on the device.
    tensor = torch.empty(shape, dtype=torch.float32 if sample_on_cpu else dtype, device=sample_device)
    if name.endswith(norm_suffixes):
        tensor = tensor.uniform_(0.9, 1.1)
    else:
        tensor = tensor.uniform_(-0.02, 0.02)
    return tensor.to(device=device, dtype=dtype)


class FixedRandomWeightSource(WeightSource):
    """Generate every reduced-model parameter deterministically on demand.

    A meta-device Transformers model provides the complete checkpoint-facing
    name and shape set without allocating model storage. Each iteration derives
    a stable per-parameter seed and regenerates the same BF16 values, avoiding
    both checkpoint weight downloads and a persistent second model copy.

    ``write_checkpoint`` turns the same iteration into a normally-loadable
    checkpoint, which is how the startup-load reference is built: the reference
    and the transfer must see identical names, shapes, dtypes, values and
    integer tables, so they cannot be generated by two code paths.
    """

    def __init__(self, case: WeightUpdateModelCase, device: torch.device) -> None:
        config = AutoConfig.from_pretrained(case.model, trust_remote_code=True)
        _apply_hf_overrides(config, case.hf_overrides)
        self._case = case
        meta_config = getattr(config, case.meta_config_attribute) if case.meta_config_attribute else config
        with torch.device("meta"):
            meta_model = AutoModelForCausalLM.from_config(meta_config, trust_remote_code=True)

        # Two distinct steps: ``_checkpoint_name`` applies the namespace prefix
        # to the raw meta-model name (once), then the checkpoint rename map runs
        # on the *expanded* names, because the meta model exposes routed experts
        # fused and the per-expert names (gate/up/down -> w1/w3/w2) only exist
        # after ``expand_fused_expert_params`` has split them.
        parameters = [
            (
                self._apply_name_map(expanded_name, case),
                resize_expert_shape(expanded_name, expanded_shape, case.expert_intermediate_size),
                torch.bfloat16,
            )
            for name, parameter in meta_model.named_parameters()
            for expanded_name, expanded_shape in expand_fused_expert_params(
                self._checkpoint_name(name, case), tuple(parameter.shape)
            )
        ]
        # Learned integer tables must travel with the payload too. Some
        # architectures expose them as HF *buffers* while the served model keeps
        # them as parameters: DeepSeek-V4's hash router keeps ``tid2eid`` as a
        # (vocab, topk) token -> expert table, and the real checkpoint ships it
        # (``model.layers.{0,1,2}.mlp.gate.tid2eid`` in the safetensors index).
        # Omitting it leaves the table to be re-materialised with ``torch.empty``
        # during a live update, and the router then indexes experts out of range.
        parameters += [
            (
                self._apply_name_map(self._checkpoint_name(name, case), case),
                tuple(buffer.shape),
                buffer.dtype,
            )
            for name, buffer in meta_model.named_buffers()
            if not buffer.dtype.is_floating_point and not buffer.dtype.is_complex
        ]
        self._num_experts = int(getattr(meta_config, "n_routed_experts", 0) or 0)
        del meta_model

        assert parameters, f"{case.id}: reduced meta model contains no parameters"
        names = [name for name, _, _ in parameters]
        assert len(set(names)) == len(names), f"{case.id}: generated checkpoint parameter names are not unique"
        self._parameters = parameters
        self._device = device

    @staticmethod
    def _checkpoint_name(name: str, case: WeightUpdateModelCase) -> str:
        """Apply the case's checkpoint namespace prefix exactly once."""
        if case.checkpoint_model_prefix is not None and name.startswith("model."):
            return case.checkpoint_model_prefix + name.removeprefix("model.")
        return name

    @staticmethod
    def _apply_name_map(name: str, case: WeightUpdateModelCase) -> str:
        """Apply the case's checkpoint rename map (idempotent per rule)."""
        if case.checkpoint_name_map is not None:
            return case.checkpoint_name_map(name)
        return name

    def _make_tensor(self, name: str, shape: tuple[int, ...], dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return _sampled_tensor(
            name,
            shape,
            dtype,
            self._device,
            num_experts=self._num_experts,
            sample_on_cpu=False,
            norm_suffixes=("norm.weight",),
            integer_topk_names=None,
        )

    @cached_property
    def metadata(self) -> list[ParamMeta]:
        return [ParamMeta(name, dtype, shape) for name, shape, dtype in self._parameters]

    def __iter__(self):
        with torch.no_grad():
            for name, shape, dtype in self._parameters:
                yield name, self._make_tensor(name, shape, dtype)

    def iter_full(self) -> Iterator[tuple[str, torch.Tensor]]:
        """vLLM drives updates through `iter_full()`."""
        yield from self.__iter__()

    @property
    def case_id(self) -> str:
        return self._case.id

    def write_checkpoint(self, directory: str | Path) -> None:
        """Materialize this source's payload as a normally-loadable checkpoint.

        Writes the repository's *own* ``config.json`` (byte for byte) next to one
        safetensors file holding exactly what iteration yields: the same
        checkpoint-facing names — including the language-model prefix, the
        DeepSeek-V4 rename map and the fused-expert expansion — the same
        corrected expert shapes, and the same integer tables. Nothing is
        recomputed from the published checkpoint, so a reference server started
        on this directory holds the same weights a transfer ships.

        The config is copied rather than re-serialized on purpose. The reference
        server takes the same ``--hf-overrides`` as the live-update lane, so both
        get the reduced model through vLLM's own override path, and both parse
        the very same bytes. A re-serialized config would add the derived fields
        a runtime config object carries (``layer_types`` / ``mlp_layer_types``
        for the *original* layer count, for example), which neither matches the
        published file nor survives vLLM's config validation once
        ``num_hidden_layers`` is reduced.
        """
        from huggingface_hub import hf_hub_download
        from safetensors.torch import save_file

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(hf_hub_download(self._case.model, "config.json"), target / "config.json")
        state = {name: tensor.detach().to("cpu").contiguous() for name, tensor in self}
        save_file(state, str(target / "model.safetensors"), metadata={"format": "pt"})


# ---------------------------------------------------------------------------
# DeepSeek-V4.1: geometry from the published checkpoint's safetensors headers
# ---------------------------------------------------------------------------

# Header cache and the checkpoint it is read from.  ``HF_ENDPOINT`` is honoured
# so a mirror needs no bespoke plumbing.
_DSV41_HEADER_CACHE = Path(os.environ.get("DSV41_HEADER_CACHE", "/tmp/dsv41_safetensors_headers"))
_DSV41_HF_REPO = os.environ.get("DSV41_HF_REPO", "deepseek-ai/DeepSeek-V4.1-Flash")
_DSV41_HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")

# Target size of one checkpoint shard written by the V4.1 ``write_checkpoint``.
_DSV41_SHARD_BYTES = 4 << 30

_DSV41_TORCH_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

_DSV41_GATE_TENSOR = re.compile(r"^layers\.\d+\.ffn\.gate\.(weight|bias|bias_vl)$")
_DSV41_LAYER_INDEX = re.compile(r"^layers\.(\d+)\.")
_DSV41_VISION_BLOCK = re.compile(r"^vision\.blocks\.(\d+)\.")
_DSV41_ENGRAM_EMBED = re.compile(r"^layers\.(\d+)\.engram\.embed\.weight$")


@dataclass(frozen=True)
class TensorSpec:
    """One payload tensor: checkpoint-facing name, torch dtype, logical shape."""

    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]


def _dsv41_text_config(case: Any) -> dict[str, Any]:
    """The reduced model's own ``text_config``.

    The case's ``hf_overrides`` deliberately is *not* the authority.  vLLM's
    ``DeepseekV41Config`` flattens the released ``text_config`` onto the
    top-level config object, so a nested ``{"text_config": {...}}`` override
    replaces that sub-config instead of merging into it, dropping every field it
    does not restate — ``num_attention_heads`` among them, which ``ModelConfig``
    then rejects: "The text_config extracted from the model config does not have
    ``num_attention_heads``".  Reading the directory the lane is actually served
    from keeps this source and the server in agreement by construction.
    """
    config = json.loads((Path(case.model) / "config.json").read_text())
    return config.get("text_config", config)


def _dsv41_engram_table_shapes(text_config: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """Exact engram table shapes for the reduced config.

    ``EngramLayout`` draws one prime near ``engram_vocab_size`` per (n-gram size,
    head) pair and never reuses a prime, so the row count is driven by
    ``engram_vocab_size`` and only bounded by ``engram_num_embeddings``
    (``assert sum(head_sizes) <= num_embeddings``).  Asking the real layout is
    the only way to stay in step with ``find_next_prime``.
    """
    from vllm.models.deepseek_v41.common.engram import EngramLayout

    layout = EngramLayout.from_config(SimpleNamespace(**text_config))
    if layout is None:
        return {}
    shapes: dict[str, tuple[int, int]] = {}
    for slot, layer_id in enumerate(layout.layer_ids):
        rows = sum(size for order in layout.primes[slot] for size in order)
        shapes[f"layers.{layer_id}.engram.embed.weight"] = (rows, layout.head_dim)
    return shapes


def _dsv41_http_get(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 180,
    limit: int | None = None,
) -> bytes:
    """GET ``url``, reading at most ``limit`` bytes when one is given.

    ``stream=True`` matters even though a range request is what we send: a mirror
    that ignores ``Range`` answers 200 with the whole shard, and a non-streaming
    ``response.content`` would then pull a multi-gigabyte file into memory.
    """
    import requests

    with requests.get(url, headers=headers, timeout=timeout, stream=True) as response:
        response.raise_for_status()
        if limit is None:
            return response.content
        chunks: list[bytes] = []
        read = 0
        for chunk in response.iter_content(chunk_size=1 << 16):
            chunks.append(chunk)
            read += len(chunk)
            if read >= limit:
                break
        return b"".join(chunks)


def _dsv41_fetch_weight_map() -> dict[str, str]:
    """The checkpoint's ``weight_map`` (tensor name -> shard file)."""
    _DSV41_HEADER_CACHE.mkdir(parents=True, exist_ok=True)
    cached = _DSV41_HEADER_CACHE / "weight_map.json"
    if cached.exists():
        return json.loads(cached.read_text())

    url = f"{_DSV41_HF_ENDPOINT}/{_DSV41_HF_REPO}/resolve/main/model.safetensors.index.json"
    weight_map = json.loads(_dsv41_http_get(url))["weight_map"]
    cached.write_text(json.dumps(weight_map))
    return weight_map


def _dsv41_fetch_header(shard: str) -> dict:
    """Read one shard's header through a range request; never the weights.

    The smallest range that usually fits is requested first and only grown when
    the header genuinely is bigger — asking for a fixed 8 MiB per shard would
    waste hundreds of MiB across a 48-shard checkpoint whose headers are tens of
    KiB.
    """
    _DSV41_HEADER_CACHE.mkdir(parents=True, exist_ok=True)
    cached = _DSV41_HEADER_CACHE / f"{shard}.json"
    if cached.exists():
        return json.loads(cached.read_text())

    url = f"{_DSV41_HF_ENDPOINT}/{_DSV41_HF_REPO}/resolve/main/{shard}"
    header_len = 0
    for limit in (1 << 20, 8 << 20, 64 << 20):
        raw = _dsv41_http_get(url, headers={"Range": f"bytes=0-{limit - 1}"}, limit=limit)
        if len(raw) < 8:
            raise RuntimeError(f"{shard}: short range response ({len(raw)} bytes)")
        header_len = struct.unpack("<Q", raw[:8])[0]
        if 8 + header_len <= len(raw):
            break
    else:
        raise RuntimeError(f"{shard}: header claims {header_len} bytes, more than the largest range fetched")

    header = json.loads(raw[8 : 8 + header_len])
    header.pop("__metadata__", None)
    cached.write_text(json.dumps(header))
    return header


class DeepseekV41WeightSource(WeightSource):
    """Generate the reduced V4.1 payload's tensors deterministically on demand.

    Same contract as ``FixedRandomWeightSource`` — stable per name, regenerated
    rather than stored, and able to materialise itself as a normal checkpoint for
    the startup-load oracle — but the layout comes from the published
    checkpoint's safetensors headers instead of a Transformers meta model, which
    V4.1 does not have.

    The lane keeps every layer and the released ``hidden_size`` / ``head_dim`` /
    ``moe_intermediate_size``, so a surviving tensor has the shape the checkpoint
    gives it; only counts shrink (experts, Engram vocabulary, MTP stages, vision
    blocks), and that reduction lives in the model directory's own ``config.json``.
    Two deliberate exceptions to "headers are the truth":

    * the Engram table's row count is derived from the real ``EngramLayout``
      (primes near ``engram_vocab_size``), not from ``engram_num_embeddings``;
    * the lane is bf16 while the checkpoint is FP8/FP4 with block scales, so
      ``_payload_spec`` maps dtypes and packed shapes instead of copying them.

    No rename map is needed: the header names *are* the checkpoint-facing names,
    because nothing is synthesised through an HF model first.
    """

    def __init__(self, case: Any, device: torch.device) -> None:
        text = _dsv41_text_config(case)
        self._case = case
        self._device = device
        self._num_layers = int(text.get("num_hidden_layers", 40))
        self._num_experts = int(text.get("n_routed_experts", 384))
        self._num_nextn = int(text.get("num_nextn_predict_layers", 0))
        self._engram_layer_ids = tuple(text.get("engram_layer_ids") or ())
        self._engram_shapes = _dsv41_engram_table_shapes(text) if self._engram_layer_ids else {}
        self._num_vision_blocks = int(
            json.loads((Path(case.model) / "config.json").read_text())
            .get("vision_config", {})
            .get("num_hidden_layers", 2)
        )
        # The model directory also carries the config and tokenizer the reference
        # startup-load checkpoint has to reproduce.
        self._model_dir = Path(case.model)
        self._specs = self._read_layout()

    # -- layout -------------------------------------------------------------

    def _in_reduced_lane(self, name: str) -> bool:
        """Keep only what the reduced case actually instantiates.

        The three exclusions are config-driven on purpose: the lane keeps all
        ``num_hidden_layers`` layers, so anything the config still turns on has
        to reach ``load_weights``.
        """
        if name.endswith(".scale"):
            # Block scales have no counterpart in a bf16 model.  This has to be
            # tested *before* the engram branch: an ``engram.embed.scale`` that
            # slips through gets rewritten to ``...embed_tokens.weight_scale_inv``
            # by the model's name mapper, and a bf16 engram table has no such
            # parameter, so ``params_dict[name]`` raises ``KeyError``.
            # ``hc_attn_scale`` ends in ``_scale``, not ``.scale``, and is real.
            return False
        if name.startswith("mtp."):
            # num_nextn_predict_layers == 0 turns the MTP / dSpark stages off.
            return self._num_nextn > 0
        if ".engram." in name:
            # The tables themselves are kept (CPU-offloaded by the server), but
            # only for the layers the config still lists.
            match = _DSV41_ENGRAM_EMBED.match(name) or re.match(r"^layers\.(\d+)\.engram\.", name)
            if match is None:
                return False
            return int(match.group(1)) in self._engram_layer_ids

        if name.startswith("vision.blocks."):
            match = _DSV41_VISION_BLOCK.match(name)
            return match is not None and int(match.group(1)) < self._num_vision_blocks

        if not name.startswith("layers."):
            # embed / norm / head / vision.norm / vision.patch_embed / aligner.*
            return True

        match = _DSV41_LAYER_INDEX.match(name)
        if match is None or int(match.group(1)) >= self._num_layers:
            return False

        parts = name.split(".")
        if "experts" in parts:  # layers.N.ffn.experts.E.<...>
            idx = parts.index("experts")
            if idx + 1 < len(parts) and parts[idx + 1].isdigit():
                return int(parts[idx + 1]) < self._num_experts
        return True

    def _payload_spec(self, name: str, dtype: str, shape: tuple[int, ...]) -> tuple[torch.dtype, tuple[int, ...]]:
        """Map one checkpoint tensor onto the bf16 model's expectation."""
        if name in self._engram_shapes:
            # The reduced engram_vocab_size moved the row count away from the
            # published header; the layout is the authority here.
            return torch.bfloat16, self._engram_shapes[name]
        if dtype == "I8":
            # Only routed experts are FP4-packed in this checkpoint; anything
            # else appearing as int8 would mean the packing assumption is wrong.
            assert ".experts." in name, f"{name}: unexpected int8 tensor outside the experts"
            return torch.bfloat16, (*shape[:-1], shape[-1] * 2)
        if dtype.startswith("F8_"):
            return torch.bfloat16, shape
        if _DSV41_GATE_TENSOR.match(name):
            # The router is shaped by the expert count.
            return _DSV41_TORCH_DTYPES[dtype], (self._num_experts, *shape[1:])
        return _DSV41_TORCH_DTYPES[dtype], shape

    def _read_layout(self) -> list[TensorSpec]:
        weight_map = _dsv41_fetch_weight_map()
        specs: list[TensorSpec] = []
        for shard in sorted(set(weight_map.values())):
            for name, entry in _dsv41_fetch_header(shard).items():
                if not self._in_reduced_lane(name):
                    continue
                dtype, shape = self._payload_spec(name, entry["dtype"], tuple(entry["shape"]))
                specs.append(TensorSpec(name, dtype, shape))
        specs.sort(key=lambda spec: spec.name)
        assert specs, (
            f"{self._case.id}: no tensors survived the reduced filter "
            f"(layers<{self._num_layers}, experts<{self._num_experts})"
        )
        return specs

    # -- WeightSource -------------------------------------------------------

    # ``WeightSource.__init_subclass__`` rejects a plain method here: engines
    # read the metadata repeatedly and plan from it, so it must be computed once
    # and never change. Same contract as ``FixedRandomWeightSource``.
    @cached_property
    def metadata(self) -> list[ParamMeta]:
        return [ParamMeta(spec.name, spec.dtype, spec.shape) for spec in self._specs]

    def _make_tensor(self, spec: TensorSpec) -> torch.Tensor:
        return _sampled_tensor(
            spec.name,
            spec.shape,
            spec.dtype,
            self._device,
            num_experts=self._num_experts,
            sample_on_cpu=True,
            norm_suffixes=("norm.weight", "attn_sink"),
            integer_topk_names=("tid2eid", "eid2tid"),
        )

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        with torch.no_grad():
            for spec in self._specs:
                yield spec.name, self._make_tensor(spec)

    def iter_full(self) -> Iterator[tuple[str, torch.Tensor]]:
        """vLLM drives weight updates through ``iter_full()``."""
        yield from self.__iter__()

    @property
    def case_id(self) -> str:
        return self._case.id

    def max_tensor_bytes(self) -> int:
        return max(int(torch.tensor(spec.shape).prod()) * spec.dtype.itemsize for spec in self._specs)

    def write_checkpoint(self, directory: str | Path) -> None:
        """Materialise this source's payload as a normally-loadable checkpoint.

        Unlike ``FixedRandomWeightSource`` this shards the tensors behind a
        ``model.safetensors.index.json``: the config and tokenizer come from the
        lane's own model directory, and a single ``model.safetensors`` is not
        enough once engram is enabled, because the engram preflight rejects
        anything that is not an *indexed* checkpoint — "Engram table
        'layers.N.engram.embed.weight' is not in an indexed safetensors
        checkpoint ... (found: no index)".

        The config is copied rather than re-serialised on purpose: a
        re-serialised config would carry the derived fields a runtime config
        object holds, which matches neither the published file nor survives
        validation once the layer count is reduced.
        """
        from safetensors.torch import save_file

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
            source = self._model_dir / name
            if source.exists():
                shutil.copyfile(source, target / name)
            elif name == "config.json":
                raise FileNotFoundError(f"{self._case.id}: {source} is required to build the reference")

        weight_map: dict[str, str] = {}
        total_size = 0
        shard: dict[str, torch.Tensor] = {}
        shard_bytes = 0
        shard_index = 0

        def flush() -> None:
            nonlocal shard, shard_bytes, shard_index
            if not shard:
                return
            shard_index += 1
            filename = f"model-{shard_index:05d}.safetensors"
            save_file(shard, str(target / filename), metadata={"format": "pt"})
            for tensor_name in shard:
                weight_map[tensor_name] = filename
            shard = {}
            shard_bytes = 0

        # A few GiB per shard, so the 49 GiB payload never has to sit in host
        # memory all at once.
        for name, tensor in self:
            tensor = tensor.detach().to("cpu").contiguous()
            shard[name] = tensor
            size = tensor.numel() * tensor.element_size()
            shard_bytes += size
            total_size += size
            if shard_bytes >= _DSV41_SHARD_BYTES:
                flush()
        flush()

        (target / "model.safetensors.index.json").write_text(
            json.dumps(
                {"metadata": {"total_size": total_size}, "weight_map": weight_map},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        # Every tensor above was materialised on the NPU and then copied to CPU.
        # The caching allocator keeps those blocks, and at 40 layers the retained
        # total is enough to starve the server that starts next on the same chip
        # ("Free memory on device ... is less than desired GPU memory
        # utilization"), so hand the memory back explicitly.
        if self._device.type == "npu":
            torch.npu.empty_cache()


def packed_buffer_size_for(source: WeightSource) -> int:
    """Size a packed transfer buffer so every single tensor fits.

    Both engines default to 1 GiB, which is smaller than the largest tensor of
    some reduced cases (GLM-5.1's embedding is ~1.9 GB), and the producer then
    raises ``ValueError: Tensor ... exceeds buffer_size_bytes``. Keep the 1 GiB
    floor and add 128 MiB of headroom above the largest tensor.
    """
    metadata = source.metadata
    max_tensor_bytes = max(math.prod(meta.shape) * meta.dtype.itemsize for meta in metadata)
    return max(max_tensor_bytes + 128 * 2**20, 2**30)


def make_weight_source(case: WeightUpdateModelCase, device: torch.device) -> WeightSource:
    """Build the payload source a case declares, or the default random one.

    Cases whose architecture Transformers can enumerate use
    ``FixedRandomWeightSource``. DeepSeek-V4.1 cannot - ``AutoConfig`` raises
    ``KeyError: 'deepseek_v41'`` - so it names the header-driven source that
    lives in this module too. ``packed_buffer_size_for`` works on either, because
    both answer ``metadata()`` with the same ``ParamMeta``.
    """
    if case.weight_source_factory is not None:
        return case.weight_source_factory(case, device)
    return FixedRandomWeightSource(case, device)


@contextlib.contextmanager
def fixed_startup_checkpoint(source: WeightSource) -> Iterator[str]:
    """Expose the source's payload as a throwaway startup-load checkpoint."""
    with tempfile.TemporaryDirectory(prefix=f"fixed-startup-{source.case_id}-") as directory:
        source.write_checkpoint(directory)
        yield directory
