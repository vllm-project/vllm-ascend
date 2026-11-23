# SPDX-License-Identifier: Apache-2.0
"""Fit attention-aware NOVA-KV artifacts from Ascend capture samples."""

import argparse
import json
from pathlib import Path

import torch

from vllm_ascend.quantization.methods.kv_cache.nova_kv.codec import FORMAT_VERSION, NovaCodec


def hadamard(dim):
    if dim <= 0 or dim & (dim - 1):
        raise ValueError("Hadamard dimension must be a power of two")
    result = torch.ones(1, 1, dtype=torch.float64)
    while result.shape[0] < dim:
        result = torch.cat([torch.cat([result, result], 1), torch.cat([result, -result], 1)], 0) / 2**0.5
    return result


def fit_head(key, value, query_moment, iterations, device):
    key, value = key.double(), value.double()
    dim = key.shape[-1]
    query_moment = query_moment / query_moment.diagonal().mean().clamp_min(1e-12)
    eigenvalues, eigenvectors = torch.linalg.eigh((query_moment + query_moment.T) / 2)
    # A finite eigenvalue floor stabilizes the BF16/NPU reconstruction path.
    query_root = (eigenvectors * eigenvalues.clamp_min(1e-6).sqrt()) @ eigenvectors.T
    mean = key.mean(0)
    centered = key - mean
    covariance = centered.T @ centered / len(key)
    spectrum, basis = torch.linalg.eigh(query_root @ covariance @ query_root)
    basis = basis.flip(-1)
    groups = dim // 4
    order = torch.arange(dim).reshape(4, groups).T.flatten()
    transform = query_root @ basis[:, order]
    inverse = torch.linalg.inv(transform)
    transformed = centered @ transform
    transformed = transformed / transformed.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
    samples = transformed.float().reshape(len(key), groups, 4).transpose(0, 1).contiguous().to(device)
    generator = torch.Generator().manual_seed(1024)
    books = samples[:, torch.randperm(len(key), generator=generator)[:256]].clone()
    for iteration in range(iterations):
        distance = samples.square().sum(-1, keepdim=True) + books.square().sum(-1).unsqueeze(1)
        distance -= 2 * torch.bmm(samples, books.transpose(-1, -2))
        assignment = distance.argmin(-1)
        sums = torch.zeros_like(books)
        sums.scatter_add_(1, assignment[..., None].expand(-1, -1, 4), samples)
        counts = torch.zeros((groups, 256), device=device)
        counts.scatter_add_(1, assignment, torch.ones_like(assignment, dtype=torch.float32))
        new_books = sums / counts.clamp_min(1)[..., None]
        books = torch.where((counts > 0)[..., None], new_books, books)
    weights = ((key @ query_moment) * key).sum(-1).clamp_min(0)
    weights = weights / weights.mean().clamp_min(1e-12)
    value_covariance = (value * weights[:, None]).T @ value / len(value)
    _, value_basis = torch.linalg.eigh((value_covariance + value_covariance.T) / 2)
    value_transform = value_basis @ hadamard(dim)
    return {
        "key_mean": mean.float(),
        "key_transform": transform.float(),
        "key_inverse": inverse.float(),
        "value_transform": value_transform.float(),
        "codebooks": books.cpu().to(torch.float16),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.iterations < 1 or args.threads < 1:
        parser.error("iterations and threads must be positive")
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    layers = sorted({p.stem.split("_rank_")[0] for p in args.capture.glob("layer_*_rank_*.pt")})
    if not layers:
        raise ValueError("No captured layers")
    report = {}
    for layer in layers:
        paths = sorted(args.capture.glob(f"{layer}_rank_*.pt"), key=lambda path: int(path.stem.split("_rank_")[1]))
        if [int(path.stem.split("_rank_")[1]) for path in paths] != list(range(len(paths))):
            raise ValueError(f"Missing tensor parallel capture shard for {layer}")
        shards = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
        if len({(shard["seen"], len(shard["key"])) for shard in shards}) != 1:
            raise ValueError(f"Tensor parallel captures are not synchronized for {layer}")
        key = torch.cat([s["key"] for s in shards], dim=1)
        value = torch.cat([s["value"] for s in shards], dim=1)
        moments = torch.cat([s["query_moment"] for s in shards], dim=0)
        if len(key) < 256:
            raise ValueError("At least 256 calibration tokens are required")
        heads = []
        for h in range(key.shape[1]):
            heads.append(fit_head(key[:, h], value[:, h], moments[h], args.iterations, args.device))
            print(f"{layer} head={h} fit complete", flush=True)
        artifact = {name: torch.stack([h[name] for h in heads]) for name in heads[0]}
        artifact["format_version"] = FORMAT_VERSION
        artifact["calibration_tokens"] = len(key)
        artifact["iterations"] = args.iterations
        artifact["algorithm"] = "NOVA-KV group-VQ K + rotated INT2 V; query eigenvalue floor 1e-6"
        torch.save(artifact, args.output / f"{layer}.pt")
        codec = NovaCodec(**{name: artifact[name] for name in NovaCodec.__dataclass_fields__})
        k, v = codec.decode(*codec.encode(key[:256], value[:256]))
        report[layer] = {
            "key_nmse": float((k - key[:256].float()).square().sum() / key[:256].float().square().sum()),
            "value_nmse": float((v - value[:256].float()).square().sum() / value[:256].float().square().sum()),
            "samples": len(key),
            "seen_per_rank": [s["seen"] for s in shards],
        }
        (args.output / "calibration_metrics.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
