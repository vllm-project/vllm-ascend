# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SSD table manifest: artifact identity, layout validation and TableSpecs.

The manifest binds an SSD layout file to the exact model artifact it was
packed from (source weight identity, layers, per-layer rows, quant format).
A path match alone never proves weight identity; the version string is the
artifact identity and must equal what the deployment expects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROW_BYTES = 288
SUPPORTED_SCHEMA = ("complete", "version", "sources", "layer_row_bases")
SUPPORTED_CODE_DTYPE = ("I8",)          # INT8[256]
SUPPORTED_SCALE_DTYPE = ("F8_E8M0", "F32")  # packed as FP32[8] rows


class SSDManifestError(ValueError):
    """Manifest rejected: identity/layout mismatch or malformed file."""


@dataclass(frozen=True)
class TableSpec:
    """One layer's placement inside the SSD table file(s)."""

    layer_id: int
    logical_rows: int
    packed_global_base: int
    file_path: str
    file_row_origin: int      # 行偏移基准: 完整 packed=0; L14 切片=packed_global_base
    row_bytes: int = ROW_BYTES
    codes_dtype: str = "I8"
    scale_dtype: str = "F32"

    def row_offset(self, packed_global_id: int) -> int:
        """文件内字节偏移; 调用方保证 id 在本层逻辑区间内。"""
        return (packed_global_id - self.file_row_origin) * self.row_bytes

    @property
    def logical_interval(self) -> tuple[int, int]:
        return (self.packed_global_base,
                self.packed_global_base + self.logical_rows)


class SSDManifest:
    """Validated view of a packed-table manifest plus explicit table files."""

    def __init__(self, specs: list[TableSpec], raw: dict[str, Any],
                 version: str, file_bytes: dict[str, int]):
        self.specs = specs
        self.raw = raw
        self.version = version
        self.file_bytes = file_bytes
        self.total_logical_rows = sum(s.logical_rows for s in specs)

    @classmethod
    def load(cls, path: str | Path, model_layout: dict[str, Any],
             table_paths: list[str]) -> SSDManifest:
        """Load and fully validate against the expected model layout.

        ``model_layout`` keys (what the model deployment expects):
          ``version``: artifact identity string (must match manifest).
          ``layers``: {layer_id: {"rows": r, "codes_dtype": "I8",
          "codes_dim": 256, "scale_dtype": "F8_E8M0"}}.
        """
        p = Path(path)
        if not p.exists():
            raise SSDManifestError(f"manifest not found: {p}")
        try:
            raw = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            raise SSDManifestError(f"manifest is not valid JSON: {e}") from e
        for key in SUPPORTED_SCHEMA:
            if key not in raw:
                raise SSDManifestError(f"manifest missing key: {key}")
        if raw.get("complete") is not True:
            raise SSDManifestError("manifest is not complete")
        version = str(raw.get("version", ""))
        expected_version = str(model_layout.get("version", ""))
        if not expected_version:
            raise SSDManifestError("model_layout must pin an artifact version")
        if version != expected_version:
            raise SSDManifestError(
                f"manifest identity mismatch: manifest={version!r} "
                f"expected={expected_version!r} (same path does not prove "
                f"same weights)")
        bases = raw["layer_row_bases"]
        sources = raw["sources"]
        if not isinstance(bases, list) or not isinstance(sources, list) \
                or len(bases) != len(sources):
            raise SSDManifestError("layer_row_bases/sources shape mismatch")
        layers_expected: dict[int, dict[str, Any]] = {
            int(k): v for k, v in model_layout.get("layers", {}).items()}
        if not layers_expected:
            raise SSDManifestError("model_layout has no layers")

        specs: list[TableSpec] = []
        for base, src in zip(bases, sources):
            layer = int(src.get("layer", -1))
            if layer not in layers_expected:
                raise SSDManifestError(f"manifest layer {layer} not expected by model")
            exp = layers_expected[layer]
            rows = int(src.get("rows", -1))
            if rows != int(exp["rows"]):
                raise SSDManifestError(
                    f"layer {layer} rows mismatch: manifest={rows} expected={exp['rows']}")
            weight = src.get("weight", {})
            if weight.get("dtype") != exp.get("codes_dtype", "I8"):
                raise SSDManifestError(
                    f"layer {layer} codes dtype {weight.get('dtype')} not supported "
                    f"(expected {exp.get('codes_dtype', 'I8')}; BF16/MXFP8 explicitly rejected)")
            shape = weight.get("shape", [])
            if len(shape) != 2 or shape[1] != int(exp.get("codes_dim", 256)) or shape[0] != rows:
                raise SSDManifestError(f"layer {layer} codes shape invalid: {shape}")
            scale = src.get("scale", {})
            sshape = scale.get("shape", [])
            if len(sshape) != 2 or sshape[0] != rows or sshape[1] != 8:
                raise SSDManifestError(f"layer {layer} scale shape invalid: {sshape}")
            specs.append(TableSpec(
                layer_id=layer, logical_rows=rows, packed_global_base=int(base),
                file_path="", file_row_origin=0))

        # 基址单调且区间不重叠; padding 行只允许出现在区间之间
        intervals = sorted(s.logical_interval for s in specs)
        for (lo0, hi0), (lo1, hi1) in zip(intervals, intervals[1:]):
            if hi0 > lo1:
                raise SSDManifestError(f"layer intervals overlap: [{lo0},{hi0}) [{lo1},{hi1})")

        # 表文件: 1 个完整 packed 或 2 个(第二文件为 L14 精确切片)
        n = len(table_paths)
        if n == 1:
            fb = Path(table_paths[0]).stat().st_size
            total_padded_rows = bases[-1] + specs[-1].logical_rows
            if fb < (total_padded_rows - 1) * ROW_BYTES + ROW_BYTES:
                raise SSDManifestError(
                    f"table file too small: {fb} < {total_padded_rows * ROW_BYTES}")
            for s in specs:
                object.__setattr__(s, "file_path", str(table_paths[0]))
                object.__setattr__(s, "file_row_origin", 0)
            file_bytes = {str(table_paths[0]): fb}
        elif n == 2:
            l1, l2 = specs[0], specs[1]
            fb1 = Path(table_paths[0]).stat().st_size
            fb2 = Path(table_paths[1]).stat().st_size
            need1 = (l1.packed_global_base + l1.logical_rows) * ROW_BYTES
            if fb1 < need1:
                raise SSDManifestError(f"first file too small: {fb1} < {need1}")
            need2 = l2.logical_rows * ROW_BYTES
            if fb2 < need2:
                raise SSDManifestError(
                    f"second file must be an exact L14 slice: {fb2} < {need2}")
            object.__setattr__(l1, "file_path", str(table_paths[0]))
            object.__setattr__(l1, "file_row_origin", 0)
            object.__setattr__(l2, "file_path", str(table_paths[1]))
            object.__setattr__(l2, "file_row_origin", l2.packed_global_base)
            file_bytes = {str(table_paths[0]): fb1, str(table_paths[1]): fb2}
        else:
            raise SSDManifestError("table_paths must have 1 or 2 entries")
        return cls(specs, raw, version, file_bytes)

    def intervals(self) -> list[tuple[int, int]]:
        """有效逻辑行区间(padding gap 不在其中, 引擎按此拒绝 padding 行)。"""
        return [s.logical_interval for s in self.specs]

    def engine_layer_args(self) -> tuple[list[int], list[int]]:
        """(layer_rows, layer_bases) for SSDFillEngine."""
        return ([s.logical_rows for s in self.specs],
                [s.packed_global_base for s in self.specs])
