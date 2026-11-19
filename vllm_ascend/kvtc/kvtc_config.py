"""Shared KVTC offload and calibration settings."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from .quant_layout import integer, ratio_key

KVTC_CONNECTOR_NAME = "AscendKVTCOffloadConnector"


@dataclass(frozen=True)
class KVTCConfig:
    kcr: float
    vcr: float
    uncompressed_ratio: float
    sink_blocks: int
    window_blocks: int
    svd_path: str | None
    kvtc_batch_blocks: int
    kvtc_quantization: bool
    kvtc_dump: bool
    kvtc_dump_dir: str

    @property
    def compression_enabled(self) -> bool:
        return self.kcr > 0 and self.vcr > 0

    @classmethod
    def from_extra(cls, extra: Mapping) -> KVTCConfig:
        removed = {
            "kvtc_fast_codec",
            "kvtc_dequant_graph",
            "kvtc_check_status",
            "kvtc_overlap_load",
        }.intersection(extra)
        if removed:
            raise ValueError(
                f"Remove obsolete KVTC settings: {', '.join(sorted(removed))}. "
                "Use kvtc_quantization=true/false; load overlap is always enabled."
            )
        kvtc_quantization = extra.get("kvtc_quantization", False)
        if not isinstance(kvtc_quantization, bool):
            raise ValueError("kvtc_quantization must be a JSON boolean")
        kvtc_dump = extra.get("kvtc_dump", False)
        if not isinstance(kvtc_dump, bool):
            raise ValueError("kvtc_dump must be a JSON boolean")
        kvtc_dump_dir = extra.get("kvtc_dump_dir", "./kvtc_dump")
        if not isinstance(kvtc_dump_dir, str) or not kvtc_dump_dir.strip():
            raise ValueError("kvtc_dump_dir must be a nonempty path")
        values = []
        for name, default in (("kcr", 20.0), ("vcr", 20.0), ("uncompressed_ratio", 0.2)):
            value = extra.get(name, default)
            if isinstance(value, bool):
                raise ValueError(f"{name} must be a finite number")
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a finite number") from exc
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            values.append(value)
        kcr, vcr, uncompressed_ratio = values
        if (kcr > 0) != (vcr > 0) or not 0 <= uncompressed_ratio <= 1:
            raise ValueError("K and V compression must be enabled together; raw ratio must be in [0,1]")
        compressed = kcr > 0
        if compressed and min(kcr, vcr) < 1:
            raise ValueError("Compression ratios must be >= 1")
        if kvtc_quantization and not compressed:
            raise ValueError("Quantization requires PCA compression")
        if kvtc_quantization:
            ratio_key(kcr)
            ratio_key(vcr)
        svd_path = extra.get("svd_path")
        if compressed and (not isinstance(svd_path, str) or not svd_path.strip()):
            raise ValueError("Compression requires an explicit svd_path")
        return cls(
            kcr=kcr,
            vcr=vcr,
            uncompressed_ratio=uncompressed_ratio,
            sink_blocks=integer(extra.get("sink_blocks", 1), "sink_blocks"),
            window_blocks=integer(extra.get("window_blocks", 0), "window_blocks"),
            svd_path=svd_path,
            kvtc_batch_blocks=integer(extra.get("kvtc_batch_blocks", 32), "kvtc_batch_blocks", 1),
            kvtc_quantization=kvtc_quantization,
            kvtc_dump=kvtc_dump,
            kvtc_dump_dir=kvtc_dump_dir,
        )
