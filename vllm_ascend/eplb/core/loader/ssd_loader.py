#
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
# This file is a part of the vllm-ascend project.
#

import os
import json
import re
import time
import queue
import math
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Generator, Any

import torch
import torch_npu                       
from safetensors import safe_open
from vllm_ascend.eplb.core.loader.abstract_loader import ExpertWeightLoader


def log_debug(msg: str):
    print(f"[DEBUG] {msg}")


class SSDExpertWeightLoader:

        """
    Load all tensors that belong to one (layer, expert_id) pair, CPU only.
    """

    def __init__(self, load_config: Optional[dict] = None):
        self.load_config = load_config or {}
        self.counter_before_loading_weights = 0.0


    def _prepare_index(
        self,
        model_dir: str,
        index_file: str
    ) -> Dict[str, str]:
        index_path = os.path.join(model_dir, index_file)
        if not os.path.exists(index_path):
            raise FileNotFoundError(index_path)
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "weight_map" not in data:
            raise KeyError("weight_map missing in index file")
        return data["weight_map"]


    def load_expert_weight_from_ssd(
        self,
        model_dir: str,
        layer_prefix: str,
        expert_id: int,
        index_file: str = "model.safetensors.index.json"
    ) -> Dict[str, torch.Tensor]:
        index_map = self._prepare_index(model_dir, index_file)

        # collect parameter names under the target expert
        prefix = f"{layer_prefix}.experts.{expert_id}."
        param_to_shard = {
            name: shard for name, shard in index_map.items()
            if name.startswith(prefix)
        }
        if not param_to_shard:
            raise KeyError(f"expert {expert_id} not found under {layer_prefix}")

        # group by shard
        shard_to_params: Dict[str, List[str]] = {}
        for p, s in param_to_shard.items():
            shard_to_params.setdefault(s, []).append(p)

        result: Dict[str, torch.Tensor] = {}
        for shard, params in shard_to_params.items():
            shard_path = os.path.join(model_dir, shard)
            if not os.path.exists(shard_path):
                raise FileNotFoundError(shard_path)
            with safe_open(shard_path, framework="pt", device="cpu") as reader:
                for p in params:
                    result[p] = reader.get_tensor(p)
        return result

class EplbLoaderProcess(mp.Process):
    """
    Independent process for blocking SSD reads.
    """

    def __init__(
        self,
        model_dir: str,
        req_q: mp.Queue,
        res_q: mp.Queue,
        quit_evt: mp.Event,
        index_file: str = "quant_model_weight_w8a8_dynamic.safetensors.index.json"
    ):
        super().__init__(daemon=True)
        self.model_dir = model_dir
        self.req_q = req_q
        self.res_q = res_q
        self.quit_evt = quit_evt
        self.index_file = index_file
        self.loader = SSDExpertWeightLoader()

    # ---------- process loop -------------------------------------------------#
    def run(self) -> None:
        while not self.quit_evt.is_set():
            try:
                job = self.req_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if job["type"] == "shutdown":
                break

            try:
                job_id: int = job["job_id"]
                layer_prefix, expert_id = job["layer_prefix"], job["expert_id"]

                tensors = self.loader.load_expert_weight_from_ssd(
                    self.model_dir,
                    layer_prefix=layer_prefix,
                    expert_id=expert_id,
                    index_file=self.index_file
                )
                self.res_q.put((job_id, tensors, None))
            except Exception as exc:
                self.res_q.put((job["job_id"], None, str(exc)))

    @staticmethod
    def _rpc(
        layer_prefix: str,
        expert_id: int,
        model_dir: str,
        index_file: str,
        req_q: mp.Queue,
        res_q: mp.Queue,
        timeout: float = 30.0
    ) -> Dict[str, torch.Tensor]:
        job_id = time.time_ns()
        req_q.put(dict(
            type="load",
            job_id=job_id,
            layer_prefix=layer_prefix,
            expert_id=expert_id
        ))
        t0 = time.time()
        while True:
            try:
                jid, tensors, err = res_q.get(timeout=0.1)
            except queue.Empty:
                if time.time() - t0 > timeout:
                    raise RuntimeError("EPLB I/O RPC timeout")
                continue
            if jid != job_id:
                res_q.put((jid, tensors, err))  
                continue
            if err:
                raise RuntimeError(err)
            return tensors


@dataclass
class BatchMeta:
    layer_ids: List[int]
    tensors: List[torch.Tensor]      # CPU tensors in this batch


class EplbRebalanceLoader:

    def __init__(
        self,
        model_dir: str,
        num_layers: int,
        first_dense_layers: int,
        experts_per_layer: int,
        layer_group_size: int = 8,
        index_file: str = "quant_model_weight_w8a8_dynamic.safetensors.index.json"
        enable_subprocess: bool = True
    ):
        self.model_dir = model_dir
        self.first_dense = first_dense_layers
        self.num_moe_layers = num_layers - first_dense_layers
        self.experts_per_layer = experts_per_layer
        self.layer_group_size = layer_group_size
        self.index_file = index_file

        self.cpu_buf: List[torch.Tensor] = []
        self.npu_buf: List[torch.Tensor] = []

        # multi-process I/O
        self.enable_mp = enable_subprocess
        if self.enable_mp:
            self.req_q: mp.Queue = mp.Queue()
            self.res_q: mp.Queue = mp.Queue()
            self.quit_evt = mp.Event()
            self.proc = EplbLoaderProcess(
                model_dir,
                self.req_q,
                self.res_q,
                self.quit_evt
            )
            self.proc.start()

    def _load_one_expert_cpu(self, layer_prefix: str, expert_id: int) -> Dict[str, torch.Tensor]:
        if self.enable_mp:
            return EplbLoaderProcess._rpc(
                layer_prefix=layer_prefix,
                expert_id=expert_id,
                model_dir=self.model_dir,
                index_file=index_file,
                req_q=self.req_q,
                res_q=self.res_q
            )

        loader = SSDExpertWeightLoader()
        return loader.load_expert_weight_from_ssd(
            self.model_dir,
            layer_prefix=layer_prefix,
            expert_id=expert_id
        )

    def load_batch(self, batch_idx: int, expert_ids: List[int]) -> BatchMeta:

        start_layer = batch_idx * self.layer_group_size + self.first_dense
        end_layer = min(start_layer + self.layer_group_size,
                        self.first_dense + self.num_moe_layers)

        tensors: List[torch.Tensor] = []
        layer_ids: List[int] = list(range(start_layer, end_layer))

        for lid in layer_ids:
            prefix = f"model.layers.{lid}.mlp"
            for eid in expert_ids:
                layer_pref = prefix
                loaded = self._load_one_expert_cpu(layer_pref, eid)
                tensors.extend(loaded.values())
        # keep CPU buffer for later copy
        self.cpu_buf = tensors
        return BatchMeta(layer_ids, tensors)

    def host_to_npu(self):
        if not self.cpu_buf:
            raise RuntimeError("CPU buffer is empty, call load_batch first")

        if not self.npu_buf:
            self.npu_buf = [t.npu() for t in self.cpu_buf]
        else:
            for dst, src in zip(self.npu_buf, self.cpu_buf):
                dst.copy_(src)

    def shutdown(self):
        if self.enable_mp:
            self.quit_evt.set()
            self.req_q.put(dict(type="shutdown"))
            self.proc.join(timeout=2.0)
            log_debug("EplbLoaderProcess terminated")


if __name__ == "__main__":

    index_file = 'quant_model_weight_w8a8_dynamic.safetensors.index.json'

    loader = EplbRebalanceLoader(
        model_dir="/home/data/eplb_vllm/DeepSeek-V3-W8A8",
        num_layers=24,
        first_dense_layers=3,
        experts_per_layer=4,
        layer_group_size=4,
        enable_subprocess=True,
        index_file = index_file
    )

    # load batch 0 (layers 2-5) for experts 0 and 2
    meta = loader.load_batch(batch_idx=0, expert_ids=[0, 2])
    log_debug(f"CPU tensors in batch: {len(meta.tensors)}")

    # copy to NPU
    loader.host_to_npu()
    log_debug(f"NPU buffer tensors: {len(loader.npu_buf)}")

    loader.shutdown()
