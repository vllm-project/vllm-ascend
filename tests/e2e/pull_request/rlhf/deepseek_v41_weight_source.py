# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""DeepSeek-V4.1 payload source for the weight-update E2Es.

``FixedRandomWeightSource`` builds its payload by enumerating a Transformers
meta model, which is what makes the matrix download-free::

    config = AutoConfig.from_pretrained(case.model)
    meta_model = AutoModelForCausalLM.from_config(config)

DeepSeek-V4.1 breaks exactly that step.  vLLM serves the architecture
(``DeepseekV41ForCausalLM`` is in the registry) but the pinned Transformers does
not know it::

    KeyError: 'deepseek_v41'

so there is no meta model to enumerate.  This module supplies the payload
geometry instead, from the published checkpoint's **own safetensors headers**: a
safetensors file starts with ``<u64 header_len>`` followed by a JSON blob of
every ``name -> dtype, shape`` in that shard, so an HTTP range request for the
leading bytes yields the exact checkpoint-facing names and shapes without
downloading a single weight.  Headers are cached under ``DSV41_HEADER_CACHE``.

No rename map is needed.  ``weight_transfer_test_utils`` carries
``deepseek_v4_checkpoint_name`` because it synthesises an HF model whose names
differ from the served model's; here the names *are* the checkpoint's, so that
translation is the identity.  The only adjustments are arithmetic, in
``_payload_spec``.

What the case reduces, and why the shapes still work
----------------------------------------------------
The lane deliberately keeps **all 40 layers** and the released ``hidden_size`` /
``head_dim`` / ``moe_intermediate_size``, so every surviving tensor has exactly
the shape the published checkpoint gives it.  Only *counts* shrink:

===================  =========  ==============
knob                 released   this lane
===================  =========  ==============
num_hidden_layers    40         40  (unchanged)
n_routed_experts     384        8
engram_vocab_size    16000000   40960
num_nextn_predict... 3          0
vision layers        32         2
===================  =========  ==============

That lands the GPU-side weights at roughly 14 GiB, which is what makes a
single-chip NPU IPC lane — the deployment that backend exists for — possible at
full depth.

The one exception to "headers are the truth" is the engram table.  Its row count
is **not** ``engram_num_embeddings``: ``EngramLayout`` draws one prime of about
``engram_vocab_size`` per (n-gram size, head) pair, so the table is
``(max_ngram_size - 1) * n_heads * engram_vocab_size`` rows in practice.  Since
this lane shrinks ``engram_vocab_size``, those two tensors get their shape from
the real ``EngramLayout`` instead (see ``_engram_table_shapes``) — reaching for
the layout beats reimplementing ``find_next_prime`` and drifting from it.

The lane is bf16 while the checkpoint is FP8 with FP4 experts and MXFP8
(``ue8m0``) block scales, so ``_payload_spec`` maps one onto the other:
``.scale`` partners are dropped, FP8 weights keep their logical shape and only
change dtype, packed FP4 experts get their last dimension doubled back, and the
router shrinks with ``n_routed_experts``.
"""

import hashlib
import json
import os
import re
import shutil
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import requests
import torch
from vllm.distributed.weight_transfer.base import ParamMeta, WeightSource

HEADER_CACHE = Path(os.environ.get("DSV41_HEADER_CACHE", "/tmp/dsv41_safetensors_headers"))
HF_REPO = os.environ.get("DSV41_HF_REPO", "deepseek-ai/DeepSeek-V4.1-Flash")
# Honour the ecosystem-standard variable so a mirror needs no bespoke plumbing.
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")

# Matches ``FixedRandomWeightSource.FIXED_WEIGHT_SEED``'s role, not its value:
# any fixed constant makes the payload reproducible across lanes and runs.
FIXED_WEIGHT_SEED = 20261010

# Target size of one checkpoint shard written by ``write_checkpoint``.
_SHARD_BYTES = 4 << 30

_TORCH_DTYPES = {
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

_GATE_TENSOR = re.compile(r"^layers\.\d+\.ffn\.gate\.(weight|bias|bias_vl)$")
_LAYER_INDEX = re.compile(r"^layers\.(\d+)\.")
_VISION_BLOCK = re.compile(r"^vision\.blocks\.(\d+)\.")
_ENGRAM_EMBED = re.compile(r"^layers\.(\d+)\.engram\.embed\.weight$")


@dataclass(frozen=True)
class TensorSpec:
    """One payload tensor: checkpoint-facing name, torch dtype, logical shape."""

    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]


def _read_config(case: Any) -> dict[str, Any]:
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


def _engram_table_shapes(text_config: dict[str, Any]) -> dict[str, tuple[int, int]]:
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


def _http_get(
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


def _fetch_weight_map() -> dict[str, str]:
    """The checkpoint's ``weight_map`` (tensor name -> shard file)."""
    HEADER_CACHE.mkdir(parents=True, exist_ok=True)
    cached = HEADER_CACHE / "weight_map.json"
    if cached.exists():
        return json.loads(cached.read_text())

    url = f"{HF_ENDPOINT}/{HF_REPO}/resolve/main/model.safetensors.index.json"
    weight_map = json.loads(_http_get(url))["weight_map"]
    cached.write_text(json.dumps(weight_map))
    return weight_map


def _fetch_header(shard: str) -> dict:
    """Read one shard's header through a range request; never the weights.

    The smallest range that usually fits is requested first and only grown when
    the header genuinely is bigger — asking for a fixed 8 MiB per shard would
    waste hundreds of MiB across a 48-shard checkpoint whose headers are tens of
    KiB.
    """
    HEADER_CACHE.mkdir(parents=True, exist_ok=True)
    cached = HEADER_CACHE / f"{shard}.json"
    if cached.exists():
        return json.loads(cached.read_text())

    url = f"{HF_ENDPOINT}/{HF_REPO}/resolve/main/{shard}"
    header_len = 0
    for limit in (1 << 20, 8 << 20, 64 << 20):
        raw = _http_get(url, headers={"Range": f"bytes=0-{limit - 1}"}, limit=limit)
        if len(raw) < 8:
            raise RuntimeError(f"{shard}: short range response ({len(raw)} bytes)")
        header_len = struct.unpack("<Q", raw[:8])[0]
        if 8 + header_len <= len(raw):
            break
    else:
        raise RuntimeError(
            f"{shard}: header claims {header_len} bytes, more than the largest range fetched"
        )

    header = json.loads(raw[8 : 8 + header_len])
    header.pop("__metadata__", None)
    cached.write_text(json.dumps(header))
    return header


class DeepseekV41WeightSource(WeightSource):
    """Generate the reduced V4.1 payload's tensors deterministically on demand.

    Same contract as ``FixedRandomWeightSource`` — stable per name, regenerated
    rather than stored, and able to materialise itself as a normal checkpoint for
    the startup-load oracle — but driven by the checkpoint's header layout
    instead of a Transformers meta model.
    """

    def __init__(self, case: Any, device: torch.device) -> None:
        text = _read_config(case)
        self._case = case
        self._device = device
        self._num_layers = int(text.get("num_hidden_layers", 40))
        self._num_experts = int(text.get("n_routed_experts", 384))
        self._num_nextn = int(text.get("num_nextn_predict_layers", 0))
        self._engram_layer_ids = tuple(text.get("engram_layer_ids") or ())
        self._engram_shapes = _engram_table_shapes(text) if self._engram_layer_ids else {}
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
            match = _ENGRAM_EMBED.match(name) or re.match(r"^layers\.(\d+)\.engram\.", name)
            if match is None:
                return False
            return int(match.group(1)) in self._engram_layer_ids

        if name.startswith("vision.blocks."):
            match = _VISION_BLOCK.match(name)
            return match is not None and int(match.group(1)) < self._num_vision_blocks

        if not name.startswith("layers."):
            # embed / norm / head / vision.norm / vision.patch_embed / aligner.*
            return True

        match = _LAYER_INDEX.match(name)
        if match is None or int(match.group(1)) >= self._num_layers:
            return False

        parts = name.split(".")
        if "experts" in parts:  # layers.N.ffn.experts.E.<...>
            idx = parts.index("experts")
            if idx + 1 < len(parts) and parts[idx + 1].isdigit():
                return int(parts[idx + 1]) < self._num_experts
        return True

    def _payload_spec(
        self, name: str, dtype: str, shape: tuple[int, ...]
    ) -> tuple[torch.dtype, tuple[int, ...]]:
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
        if _GATE_TENSOR.match(name):
            # The router is shaped by the expert count.
            return _TORCH_DTYPES[dtype], (self._num_experts, *shape[1:])
        return _TORCH_DTYPES[dtype], shape

    def _read_layout(self) -> list[TensorSpec]:
        weight_map = _fetch_weight_map()
        specs: list[TensorSpec] = []
        for shard in sorted(set(weight_map.values())):
            for name, entry in _fetch_header(shard).items():
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

    def metadata(self) -> list[ParamMeta]:
        return [ParamMeta(spec.name, spec.dtype, spec.shape) for spec in self._specs]

    @staticmethod
    def _seed(name: str) -> int:
        digest = hashlib.sha256(f"{FIXED_WEIGHT_SEED}:{name}".encode()).digest()
        return int.from_bytes(digest[:8], "little") % (2**63 - 1)

    def _make_tensor(self, spec: TensorSpec) -> torch.Tensor:
        """Sample on the CPU, then move the result to the lane's device.

        The CPU and NPU generators do **not** produce the same stream for a given
        seed — verified on this host: ``identical=False`` with a max absolute
        difference equal to the whole sampling range, i.e. the two draws are
        uncorrelated.  Sampling on the device would therefore make "the same
        payload" depend on *where* it was materialised, and the oracle compares a
        live transfer against a startup load produced in a different process.
        That is not hypothetical: the engram tables the live lane streams from
        disk were written on the CPU while the reference checkpoint was written
        on the NPU, which is exactly how a mismatch appeared.
        """
        seed = self._seed(spec.name)
        torch.manual_seed(seed)

        if not spec.dtype.is_floating_point:
            # Integer tables must stay inside the expert range and be unique
            # within a row, or the dispatch/combine kernels index out of bounds.
            if len(spec.shape) == 2 and spec.name.endswith(("tid2eid", "eid2tid")):
                scores = torch.rand(spec.shape[0], max(self._num_experts, spec.shape[1]))
                return scores.topk(spec.shape[1], dim=1).indices.to(spec.dtype).to(self._device)
            return torch.randint(0, 2, spec.shape, dtype=spec.dtype).to(self._device)

        # Sample in fp32 and cast: one code path for every float storage dtype.
        tensor = torch.empty(spec.shape, dtype=torch.float32)
        if spec.name.endswith(("norm.weight", "attn_sink")):
            tensor = tensor.uniform_(0.9, 1.1)
        else:
            tensor = tensor.uniform_(-0.02, 0.02)
        return tensor.to(device=self._device, dtype=spec.dtype)

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        with torch.no_grad():
            for spec in self._specs:
                yield spec.name, self._make_tensor(spec)

    @property
    def case_id(self) -> str:
        return self._case.id

    def max_tensor_bytes(self) -> int:
        return max(
            int(torch.tensor(spec.shape).prod()) * spec.dtype.itemsize for spec in self._specs
        )

    def write_checkpoint(self, directory: str | Path) -> None:
        """Materialise this source's payload as a normally-loadable checkpoint.

        The config is copied rather than re-serialised on purpose: a
        re-serialised config would carry the derived fields a runtime config
        object holds, which matches neither the published file nor survives
        validation once the layer count is reduced.

        The tensors are sharded behind a ``model.safetensors.index.json``.  A
        single ``model.safetensors`` is not enough once engram is enabled: the
        engram preflight rejects anything that is not an *indexed* checkpoint —
        "Engram table 'layers.N.engram.embed.weight' is not in an indexed
        safetensors checkpoint ... (found: no index)".
        """
        from safetensors.torch import save_file

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
            source = self._model_dir / name
            if source.exists():
                shutil.copyfile(source, target / name)
            elif name == "config.json":
                raise FileNotFoundError(
                    f"{self._case.id}: {source} is required to build the reference"
                )

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
            if shard_bytes >= _SHARD_BYTES:
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
