from __future__ import annotations

import logging
import math
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path

import regex as re
import torch

logger = logging.getLogger(__name__)

WORKER_DIR_PATTERN = re.compile(r"^tp_(\d+)_pp_(\d+)$")
LAYER_ZERO_PATTERN = re.compile(r"-(?:chunk_\d+-)?layer_0\.bin$")
SINK_TOKENS = 4
SLIDING_WINDOW_TOKENS = 128


class SamplingPolicy(Enum):
    STRICT = "strict"
    BEST_EFFORT = "best-effort"
    RELAXED = "relaxed"
    OPEN = "open"


class KV(str, Enum):
    K = "-K-"
    V = "-V-"

    @property
    def matrix_name(self) -> str:
        return "keys" if self is KV.K else "values"


class SequenceBucket(IntEnum):
    IGNORE = 0
    SHORT = 1000
    LONG = 8000

    @classmethod
    def for_length(cls, length: int) -> SequenceBucket:
        if length < cls.SHORT:
            return cls.IGNORE
        if length < cls.LONG:
            return cls.SHORT
        return cls.LONG


@dataclass(frozen=True)
class DumpTensorSet:
    dataset: Path
    request_id: str
    worker: str
    kv: KV
    paths: tuple[Path, ...]
    token_count: int
    bucket: SequenceBucket

    @property
    def usable_tokens(self) -> int:
        return self.token_count - SINK_TOKENS - SLIDING_WINDOW_TOKENS


@dataclass(frozen=True)
class LoadedTensorSet:
    tensor_set: DumpTensorSet
    tensor: torch.Tensor


class Rope:
    rotary_emb = None
    rotary_dim = None
    is_neox_style = None

    @classmethod
    def load_model_config(cls, model_path: str) -> None:
        from transformers import AutoConfig
        from vllm.config import VllmConfig, set_current_vllm_config

        from vllm_ascend.kvtc.rope import get_rotary_embedding

        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=False).get_text_config()
        # vLLM CustomOp construction requires a config even without an engine.
        with set_current_vllm_config(VllmConfig()), torch.device("cpu"):
            cls.rotary_emb = get_rotary_embedding(cfg, dtype=torch.bfloat16)
        cls.rotary_dim = cls.rotary_emb.rotary_dim
        cls.is_neox_style = cls.rotary_emb.is_neox_style
        logger.info("Loaded vLLM RoPE: head_dim=%s", cls.rotary_dim)

    @classmethod
    def invert(cls, tensor: torch.Tensor) -> torch.Tensor:
        tokens = tensor.shape[0]
        assert cls.rotary_emb is not None, "RoPE model config was not loaded"

        cos_sin_cache = cls.rotary_emb.cos_sin_cache
        if tokens > cos_sin_cache.shape[0]:
            raise ValueError(
                f"Dump has {tokens} tokens, but the model RoPE cache has only {cos_sin_cache.shape[0]} positions."
            )

        cos, sin = cos_sin_cache[:tokens].chunk(2, dim=-1)
        cos = cos[:, None, None, :].float()
        sin = sin[:, None, None, :].float()
        rotated = tensor[..., : cls.rotary_dim].float()
        passthrough = tensor[..., cls.rotary_dim :]
        if cls.is_neox_style:
            first, second = torch.chunk(rotated, 2, dim=-1)
            inverted = torch.cat(
                (
                    (first * cos + second * sin),
                    (second * cos - first * sin),
                ),
                dim=-1,
            )
        else:
            first, second = rotated[..., ::2], rotated[..., 1::2]
            inverted = torch.stack(
                (
                    (first * cos + second * sin),
                    (second * cos - first * sin),
                ),
                dim=-1,
            ).flatten(-2)
        return torch.cat((inverted, passthrough), dim=-1).to(tensor.dtype)


def _discover_workers(directory: Path) -> list[str]:
    return sorted(
        (entry.name for entry in directory.iterdir() if entry.is_dir() and WORKER_DIR_PATTERN.fullmatch(entry.name)),
        key=lambda worker: tuple(map(int, WORKER_DIR_PATTERN.fullmatch(worker).groups())),
    )


def discover_dump_directories(
    input_dirs: Path | Sequence[Path],
) -> tuple[list[Path], list[str]]:
    if isinstance(input_dirs, Path):
        input_dirs = (input_dirs,)
    if not input_dirs:
        raise ValueError("At least one input directory is required")

    dump_dirs = []
    worker_sets = {}
    discovered_paths = set()
    for input_dir in input_dirs:
        if not input_dir.is_dir():
            raise ValueError(f"Input directory does not exist: {input_dir}")

        workers = _discover_workers(input_dir)
        candidates = [(input_dir, workers)] if workers else []
        if not candidates:
            candidates = [
                (candidate, candidate_workers)
                for candidate in sorted(input_dir.iterdir())
                if candidate.is_dir() and (candidate_workers := _discover_workers(candidate))
            ]

        if not candidates:
            raise ValueError(f"No dump directories with tp_<X>_pp_<Y> workers found in {input_dir}")

        for dump_dir, workers in candidates:
            canonical_path = dump_dir.resolve()
            if canonical_path in discovered_paths:
                raise ValueError(f"Dump directory was specified more than once: {dump_dir}")
            discovered_paths.add(canonical_path)
            dump_dirs.append(dump_dir)
            worker_sets[dump_dir] = workers
            logger.info("Discovered dump directory %s with workers: %s", dump_dir, workers)

    workers = worker_sets[dump_dirs[0]]
    for dump_dir in dump_dirs[1:]:
        if worker_sets[dump_dir] != workers:
            raise ValueError(
                "Dump directories must have the same worker set. "
                f"Expected {workers} from {dump_dirs[0]}, but found "
                f"{worker_sets[dump_dir]} in {dump_dir}."
            )
    logger.info(
        "Using %d dump directories and %d workers: %s",
        len(dump_dirs),
        len(workers),
        workers,
    )
    return dump_dirs, workers


def scan_dump_manifest(dump_dirs: list[Path], workers: list[str]) -> list[DumpTensorSet]:
    manifest = []
    for dataset in dump_dirs:
        for worker in workers:
            worker_dir = dataset / worker
            for kv in KV:
                sequences = defaultdict(list)
                for path in worker_dir.iterdir():
                    if path.is_file() and kv.value in path.name:
                        sequences[path.name.split(kv.value, 1)[0]].append(path)

                for request_id, paths in sorted(sequences.items()):
                    paths = tuple(sorted(paths))
                    layer_zero = [path for path in paths if LAYER_ZERO_PATTERN.search(path.name)]
                    if not layer_zero:
                        logger.warning(
                            "Skipping %s/%s/%s during discovery: no layer-0 dumps",
                            dataset,
                            worker,
                            request_id,
                        )
                        continue
                    try:
                        token_count = torch.concat(
                            [torch.load(path, map_location="cpu", weights_only=True) for path in layer_zero]
                        ).shape[0]
                    except Exception as error:
                        logger.warning(
                            "Skipping %s/%s/%s during discovery: %s",
                            dataset,
                            worker,
                            request_id,
                            error,
                        )
                        continue
                    bucket = SequenceBucket.for_length(token_count)
                    if bucket is SequenceBucket.IGNORE:
                        logger.debug(
                            "Ignoring too short sequence %s/%s/%s/%s: %s tokens",
                            dataset,
                            worker,
                            request_id,
                            kv.name,
                            token_count,
                        )
                        continue
                    manifest.append(
                        DumpTensorSet(
                            dataset=dataset,
                            request_id=request_id,
                            worker=worker,
                            kv=kv,
                            paths=paths,
                            token_count=token_count,
                            bucket=bucket,
                        )
                    )
    logger.info("Indexed %d eligible worker/KV request entries", len(manifest))
    return manifest


def partition_requests(
    manifest: list[DumpTensorSet],
    canonical_worker: str,
    dp_sample_tokens: int,
    seed: int,
) -> set[tuple[Path, str]]:
    grouped = defaultdict(list)
    for tensor_set in manifest:
        if tensor_set.worker == canonical_worker and tensor_set.kv is KV.K:
            grouped[(tensor_set.dataset, tensor_set.bucket)].append(tensor_set)
    if not grouped:
        raise ValueError("No eligible requests are available for the DP holdout")

    target_per_group = math.ceil(dp_sample_tokens / len(grouped))
    reserved = set()
    for group_index, ((dataset, bucket), candidates) in enumerate(
        sorted(grouped.items(), key=lambda item: (str(item[0][0]), int(item[0][1])))
    ):
        candidates.sort(key=lambda item: item.request_id)
        random.Random(seed + group_index).shuffle(candidates)
        capacity = 0
        selected = 0
        for tensor_set in candidates:
            reserved.add((dataset, tensor_set.request_id))
            capacity += tensor_set.usable_tokens
            selected += 1
            if capacity >= target_per_group:
                break
        if capacity < target_per_group:
            logger.error(
                "DP holdout group %s/%s has only %s usable tokens; %s are required",
                dataset,
                bucket.name.lower(),
                capacity,
                target_per_group,
            )
        logger.info(
            "Reserved %s requests with %s usable tokens for DP in %s/%s",
            selected,
            capacity,
            dataset,
            bucket.name.lower(),
        )
    return reserved


def select_records(
    manifest: list[DumpTensorSet],
    worker: str,
    kv: KV,
    dp_request_ids: set[tuple[Path, str]],
    partition: str,
) -> list[DumpTensorSet]:
    select_dp = partition == "dp"
    return [
        tensor_set
        for tensor_set in manifest
        if tensor_set.worker == worker
        and tensor_set.kv is kv
        and (((tensor_set.dataset, tensor_set.request_id) in dp_request_ids) == select_dp)
    ]


def load_tensor(paths: tuple[Path, ...]) -> tuple[torch.Tensor | None, int | None]:
    """Load one request only when every chunk has the same complete layer set."""
    if not paths:
        logger.warning("Discarding empty dump request")
        return None, None
    request_path = paths[0].parent
    request_name = paths[0].name.split("-K-", 1)[0].split("-V-", 1)[0]
    chunks = defaultdict(dict)
    for path in paths:
        match = re.search(r"-(?:chunk_(\d+)-)?layer_(\d+)\.bin$", path.name)
        if match is None:
            logger.warning("Discarding %s: unrecognized dump filename %s", request_name, path)
            return None, None
        chunk_id = int(match.group(1) or 0)
        layer_id = int(match.group(2))
        if layer_id in chunks[chunk_id]:
            logger.warning(
                "Discarding %s: duplicate layer %s in chunk %s at %s",
                request_name,
                layer_id,
                chunk_id,
                request_path,
            )
            return None, None
        chunks[chunk_id][layer_id] = path

    chunk_ids = sorted(chunks)
    if chunk_ids != list(range(len(chunk_ids))):
        logger.warning(
            "Discarding %s: non-contiguous chunk IDs %s at %s",
            request_name,
            chunk_ids,
            request_path,
        )
        return None, None

    expected_layer_ids = None
    chunk_tensors = []
    for chunk_id in chunk_ids:
        layer_ids = sorted(chunks[chunk_id])
        if expected_layer_ids is None:
            expected_layer_ids = layer_ids
        elif layer_ids != expected_layer_ids:
            logger.warning(
                "Discarding %s: chunk %s has layers %s, expected %s at %s",
                request_name,
                chunk_id,
                layer_ids,
                expected_layer_ids,
                request_path,
            )
            return None, None
        try:
            layer_tensors = [
                torch.load(chunks[chunk_id][layer_id], map_location="cpu", weights_only=True) for layer_id in layer_ids
            ]
            chunk = torch.stack(layer_tensors)
        except Exception as error:
            logger.warning(
                "Discarding %s: cannot load chunk %s at %s: %s",
                request_name,
                chunk_id,
                request_path,
                error,
            )
            return None, None
        if chunk.ndim != 4:
            logger.warning(
                "Discarding %s: chunk %s has shape %s, expected [layer, token, head, head_dim]",
                request_name,
                chunk_id,
                tuple(chunk.shape),
            )
            return None, None
        if chunk_tensors and (
            chunk.shape[0] != chunk_tensors[0].shape[0]
            or chunk.shape[2:] != chunk_tensors[0].shape[2:]
            or chunk.dtype != chunk_tensors[0].dtype
        ):
            logger.warning(
                "Discarding %s: chunk %s shape/dtype %s/%s differs from %s/%s",
                request_name,
                chunk_id,
                tuple(chunk.shape),
                chunk.dtype,
                tuple(chunk_tensors[0].shape),
                chunk_tensors[0].dtype,
            )
            return None, None
        chunk_tensors.append(chunk)

    try:
        tensor = torch.concat(chunk_tensors, dim=1).transpose(0, 1)
    except RuntimeError as error:
        logger.warning(
            "Discarding %s: cannot combine validated chunks at %s: %s",
            request_name,
            request_path,
            error,
        )
        return None, None
    if tensor.dtype != torch.bfloat16:
        logger.warning(
            "Discarding %s: dtype %s, expected torch.bfloat16",
            request_name,
            tensor.dtype,
        )
        return None, None
    logger.debug("Loaded %s from %s", tuple(tensor.shape), request_name)
    torch.cpu.synchronize()
    return tensor, tensor.shape[0]


def load_sample_pool(
    records: list[DumpTensorSet],
    undo_rope: bool,
) -> list[LoadedTensorSet]:
    loaded = []
    for record in records:
        tensor, token_count = load_tensor(record.paths)
        if tensor is None or token_count is None:
            continue
        try:
            if undo_rope:
                tensor = Rope.invert(tensor)
            tensor = tensor[SINK_TOKENS:-SLIDING_WINDOW_TOKENS]
            loaded.append(LoadedTensorSet(record, tensor))
        except (AssertionError, RuntimeError, ValueError) as error:
            logger.warning("Skipping %s: %s", record.paths[0], error)

    if not loaded:
        raise RuntimeError("No valid dump requests remain in the sampling pool")

    feature_shapes = Counter(entry.tensor.shape[1:] for entry in loaded)
    feature_shape, _ = feature_shapes.most_common(1)[0]
    valid = [entry for entry in loaded if entry.tensor.shape[1:] == feature_shape]
    if len(valid) != len(loaded):
        logger.warning(
            "Discarding %s loaded tensors with non-canonical feature shapes; using %s from %s tensors",
            len(loaded) - len(valid),
            feature_shape,
            len(valid),
        )
        logger.warning("Observed loaded feature shapes: %s", dict(feature_shapes))

    logger.info(
        "Loaded %s valid requests with %s usable tokens into the sampling pool",
        len(valid),
        sum(entry.tensor.shape[0] for entry in valid),
    )
    return valid


def sample_pool(
    pool: list[LoadedTensorSet],
    target: int,
    policy: SamplingPolicy,
    seed: int,
    purpose: str,
) -> torch.Tensor:
    grouped = defaultdict(list)
    for entry in pool:
        record = entry.tensor_set
        if policy in (SamplingPolicy.STRICT, SamplingPolicy.BEST_EFFORT):
            group = (record.dataset, record.bucket)
        elif policy is SamplingPolicy.RELAXED:
            group = (record.dataset,)
        else:
            group = ()
        grouped[group].append(entry)
    if not grouped:
        raise RuntimeError(f"No valid dump requests remain for {purpose}")

    tokens_per_group = target // len(grouped)
    if tokens_per_group == 0:
        raise ValueError(f"Cannot split {target} requested tokens evenly across {len(grouped)} sampling pools")

    rng = random.Random(seed)
    samples = []
    collected = {}
    kv = pool[0].tensor_set.kv if pool else "KV"
    sorted_groups = sorted(
        grouped.items(),
        key=lambda item: tuple(str(value) if isinstance(value, Path) else int(value) for value in item[0]),
    )
    for group, entries in sorted_groups:
        capacity = sum(entry.tensor.shape[0] for entry in entries)
        sample_count = tokens_per_group
        if capacity < tokens_per_group:
            label = "/".join(str(value) if isinstance(value, Path) else value.name.lower() for value in group) or "all"
            message = (
                f"Sampling pool {label} has only {capacity} usable {kv} tokens; "
                f"{tokens_per_group} are requested for {purpose}"
            )
            if policy is not SamplingPolicy.BEST_EFFORT:
                raise RuntimeError(message)
            logger.warning(message)
            sample_count = capacity

        indices = sorted(rng.sample(range(capacity), sample_count))
        first = 0
        index_offset = 0
        for entry in entries:
            end = first + entry.tensor.shape[0]
            local_indices = []
            while index_offset < len(indices) and indices[index_offset] < end:
                local_indices.append(indices[index_offset] - first)
                index_offset += 1
            if local_indices:
                samples.append(entry.tensor[local_indices].to(dtype=torch.float32, copy=True))
            first = end
        collected[group] = sample_count

    data = torch.concat(samples, dim=0).flatten(start_dim=1)
    torch.cpu.synchronize()
    for group, count in collected.items():
        label = "/".join(str(value) if isinstance(value, Path) else value.name.lower() for value in group) or "all"
        logger.info(
            "%s sampled %s tokens from pool %s",
            purpose,
            count,
            label,
        )
    logger.info("Collected %s/%s %s tokens for %s", data.shape[0], target, kv, purpose)
    return data


def fit_pca(samples: torch.Tensor, svd_dim: int, svd_iter: int) -> tuple[torch.Tensor, torch.Tensor]:
    logger.info("PCA input shape=%s dtype=%s", tuple(samples.shape), samples.dtype)
    mean = samples.mean(dim=0)
    logger.info("Calculating low-rank SVD q=%s iter=%s", svd_dim, svd_iter)
    _, _, basis = torch.svd_lowrank(samples, q=svd_dim, niter=svd_iter, M=mean)
    return mean, basis
