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
"""Model loader for the Engram SSD offload backend.

The 2×~101 GiB engram table shards must never be materialized on the
loading path: the SSD runtime reads rows from the packed table files at
serving time. This loader subclasses the default loader and only overrides
weight enumeration — table keys (``layers.<L>.engram.embed.weight`` and
``.scale``) are skipped *before* any ``get_tensor`` call, whole-table-only
shard files are never opened, and every other parameter (gate, projections,
norms, experts) loads through the default machinery unchanged.

Skipped layers are recorded in an ``external_tables`` ledger the model
consults at load end: both engram layers must be accounted for by a
validated SSD manifest, while all remaining parameters must still arrive
through the normal mechanism.

Only local safetensors checkpoints are supported in this first version;
remote/pt/np formats are rejected explicitly.
"""

from __future__ import annotations

from typing import Generator

import torch
from safetensors import safe_open
from vllm.config import LoadConfig
from vllm.logger import logger
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import _natural_sort_key, enable_tqdm
from tqdm import tqdm

# 表参数名(相对 layer 命名空间; 加载名可能带 model./source 前缀)
_ENGRAM_TABLE_SUFFIXES = (
    ".engram.embed.weight",
    ".engram.embed.scale",
)

# 全进程账本: loader 记录、模型 load 结束断言(单模型实例场景)
external_tables: dict[str, dict] = {}


def is_engram_table_key(name: str) -> bool:
    """精确匹配 engram 表 weight/scale; gate/投影/归一化不在其中。"""
    return any(name.endswith(s) for s in _ENGRAM_TABLE_SUFFIXES)


def _table_only_file(st_file: str) -> bool:
    """整个分片只含表 key 时直接跳过(不 open, 不读字节)。"""
    with safe_open(st_file, framework="pt") as f:
        return all(is_engram_table_key(k) for k in f.keys())


@register_model_loader("ascend_engram_ssd")
class EngramSSDModelLoader(DefaultModelLoader):
    """DefaultModelLoader that skips engram tables before materialization."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

    def _get_weights_iterator(
        self, source: DefaultModelLoader.Source
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        # _prepare_weights 按已知 load_format 选 allow_patterns; 复用默认准备
        # 逻辑时用 safetensors 副本(不修改共享 load_config)。
        saved = self.load_config.load_format
        try:
            self.load_config.load_format = "safetensors"
            hf_folder, hf_weights_files, use_safetensors, _ = self._prepare_weights(
                source.model_or_path,
                source.subfolder,
                source.revision,
                source.fall_back_to_pt,
                source.allow_patterns_overrides,
            )
        finally:
            self.load_config.load_format = saved
        if not use_safetensors:
            raise ValueError(
                "ascend_engram_ssd loader only supports local safetensors "
                f"checkpoints; got non-safetensors files under {hf_folder}")
        table_only_files = self._table_only_files_from_index(hf_folder)
        skipped_layers: dict[int, set[str]] = {}
        skipped_bytes = 0
        sorted_files = sorted(hf_weights_files, key=_natural_sort_key)
        with tqdm(sorted_files, desc="Loading non-table weights (SSD offload)",
                  disable=not enable_tqdm(self.load_config.use_tqdm_on_load)) as bar:
            for st_file in bar:
                if table_only_files is not None:
                    import os

                    base = os.path.basename(st_file)
                    if base in table_only_files:
                        skipped_bytes += _file_size(st_file)
                        for name in table_only_files[base]:
                            skipped_layers.setdefault(_layer_of(name), set()).add(
                                name.rsplit(".", 1)[-1])
                        continue  # 表专属分片: 不 open, 不读任何字节
                with safe_open(st_file, framework="pt") as f:
                    for name in f.keys():
                        if is_engram_table_key(name):
                            # 表 key: 记账且绝不 get_tensor
                            layer = _layer_of(name)
                            skipped_layers.setdefault(layer, set()).add(
                                name.rsplit(".", 1)[-1])
                            skipped_bytes += _tensor_bytes(f, name)
                            continue
                        tensor = f.get_tensor(name)
                        yield source.prefix + name, tensor
        if skipped_layers:
            external_tables[source.model_or_path] = {
                "layers": {k: sorted(v) for k, v in skipped_layers.items()},
                "skipped_bytes": skipped_bytes,
            }
            logger.info(
                "EngramSSDModelLoader skipped table tensors for layers %s "
                "(%.2f GiB left on SSD)",
                sorted(skipped_layers), skipped_bytes / 1024 ** 3)

    @staticmethod
    def _table_only_files_from_index(hf_folder: str) -> dict[str, list[str]] | None:
        """index weight_map → {文件名: [全部为表 key 的文件之 key 列表]}。

        无 index 时返回 None(退化为逐文件 header 探查)。
        """
        import glob
        import json
        import os

        candidates = glob.glob(os.path.join(hf_folder, "*.index.json")) + glob.glob(
            os.path.join(hf_folder, "*safetensors*.json"))
        index_file = next((c for c in candidates
                           if os.path.basename(c).endswith(".index.json")), None)
        if index_file is None:
            return None
        try:
            weight_map = json.load(open(index_file))["weight_map"]
        except Exception:
            return None
        by_file: dict[str, list[str]] = {}
        for key, fname in weight_map.items():
            by_file.setdefault(os.path.basename(fname), []).append(key)
        return {f: keys for f, keys in by_file.items()
                if keys and all(is_engram_table_key(k) for k in keys)}


def _layer_of(name: str) -> int:
    """`...layers.14.engram.embed.weight` -> 14。"""
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return -1


def _file_size(path: str) -> int:
    import os

    return os.path.getsize(path)


_DTYPE_ITEMSIZE = {"I8": 1, "U8": 1, "F8_E8M0": 1, "F8_E4M3": 1, "I16": 2, "F16": 2,
                   "BF16": 2, "I32": 4, "F32": 4, "I64": 8, "F64": 8}


def _tensor_bytes(f, name: str) -> int:
    sl = f.get_slice(name)
    shape = sl.get_shape()
    dtype = str(sl.get_dtype())
    item = _DTYPE_ITEMSIZE.get(dtype, 4)
    return int(torch.Size(shape).numel()) * item


def assert_external_tables_complete(model_path: str,
                                    expected_layers: list[int]) -> None:
    """模型 load 结束时的完整性断言(由 DeepseekV41 SSD 分支调用)。"""
    ledger = external_tables.get(model_path)
    if ledger is None:
        raise RuntimeError(
            "engram SSD model loaded without table ledger; the loader "
            "did not run or no table keys were present")
    got = set(ledger["layers"])
    want = set(expected_layers)
    if got != want:
        raise RuntimeError(
            f"engram SSD table ledger mismatch: loaded-offline layers {sorted(got)} "
            f"!= expected engram layers {sorted(want)}")
    for layer, keys in ledger["layers"].items():
        if set(keys) != {"weight", "scale"}:
            raise RuntimeError(
                f"engram layer {layer} ledger incomplete: {keys}")
