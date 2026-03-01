#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/examples/offline_inference/save_sharded_state.py
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
#
"""
Saves each worker's model state dict directly to a checkpoint, which enables a
fast load path for large tensor-parallel models where each worker only needs to
read its own shard rather than the entire checkpoint.
Sparse-Compress-Quantization state dict could also be saved via this script.

Example usage:

python save_sharded_state.py \
    --model /path/to/load \
    --tensor-parallel-size 8 \
    --output /path/to/save \
    --enable-compress \
    --compress-process-num 8

Then, the model can be loaded with

llm = LLM(
    model="/path/to/save",
    load_format="sharded_state",
    tensor_parallel_size=8,
    quantization="ascend",
)
"""

import dataclasses
import json
import multiprocessing as mp
import os
import shutil
from multiprocessing import Queue
from pathlib import Path
from typing import Any

import torch
from vllm import LLM, EngineArgs
from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
from vllm.utils.argparse_utils import FlexibleArgumentParser

SUPPORTED_COMPRESS_QUANT_TYPE = ["W8A8S", "W16A16S"]
DEFAULT_PATTERN = "model-rank-{rank}-part-{part}.safetensors"
QUANTIZATION_UPDATE_MAP = {"W8A8S": "W8A8SC", "W16A16S": "W16A16SC"}


class FileHandler:
    @staticmethod
    def validate_path(path: str, must_exist: bool = True, check_writable: bool = False) -> Path:
        """
        Comprehensive path validation.
        - Checks existence
        - Checks write permissions for the target or its parent
        """
        p = Path(path)
        if must_exist and not p.exists():
            raise FileNotFoundError(f"Error: Path '{path}' does not exist.")

        if check_writable:
            # Check the directory itself if it exists, otherwise check the parent
            target = p if p.exists() else p.parent
            if not os.access(target, os.W_OK):
                raise PermissionError(f"Permission Denied: No write access to '{target}'.")
        return p

    @staticmethod
    def safe_copy(src: Path, dst: Path):
        """Copies files or directories with permission handling."""
        try:
            if src.is_dir():
                # dirs_exist_ok=True prevents errors if the destination directory exists
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                # copy2 preserves metadata (timestamps, permissions)
                shutil.copy2(src, dst)
        except (PermissionError, OSError) as e:
            print(f"Warning: Failed to copy {src} due to: {e}")


def clean_up():
    """Clean up VLLM resources"""
    destroy_model_parallel()
    destroy_distributed_environment()
    torch.npu.empty_cache()


def parse_args():
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--output", "-o", required=True, type=str, help="path to output checkpoint")
    parser.add_argument(
        "--enable-compress",
        action="store_true",
    )
    parser.add_argument(
        "--compress-process-num",
        type=int,
        default=1,
    )
    return parser.parse_args()


def get_quant_description(json_file: str) -> dict:
    """
    Extract quantization description from JSON configuration file.

    Args:
        json_file: Path to the JSON configuration file

    Returns:
        dict: Quantization descriptor dictionary

    Raises:
        FileNotFoundError: If the JSON file does not exist
        RuntimeError: If JSON parsing fails or required keys are missing
    """
    config_path = Path(json_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Model configuration file not found: {json_file}")
    try:
        with config_path.open("r", encoding="utf-8") as file:
            quant_desc = json.load(file)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON format in {json_file}: {e}")

    return quant_desc


def update_quant_description(json_file: str) -> None:
    """
    Update quantization types in JSON configuration file based on update mapping.

    Args:
        json_file: Path to the JSON configuration file

    Raises:
        FileNotFoundError: If the JSON file does not exist
        RuntimeError: If JSON parsing fails or required keys are missing
    """
    config_path = Path(json_file)
    try:
        with config_path.open("r", encoding="utf-8") as file:
            json_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to read configuration file {json_file}: {e}")

    original_quant_type = json_data.get("model_quant_type")
    if not original_quant_type or original_quant_type not in QUANTIZATION_UPDATE_MAP:
        raise RuntimeError(
            f"Cannot update quantization type. "
            f"Original type '{original_quant_type}' not found or not supported for update in {json_file}."
        )
    updated_quant_type = QUANTIZATION_UPDATE_MAP[original_quant_type]

    updated_config = {"model_quant_type": updated_quant_type, "version": "1.0.0"}

    for key, value in json_data.items():
        if key.endswith(".weight") and value == original_quant_type:
            updated_config[key] = updated_quant_type
        elif key not in ("model_quant_type", "version"):
            updated_config[key] = value

    try:
        with config_path.open("w", encoding="utf-8") as file:
            json.dump(updated_config, file, indent=2, ensure_ascii=False)
    except OSError as e:
        raise RuntimeError(f"Failed to write updated configuration to {json_file}: {e}")


def weight_compress_worker(file_path: str, quant_desc: dict, process_num: int) -> bool:
    """
    Worker logic for multiprocessing.
    Note: Imports are inside the worker to save memory in the main process.

    Returns:
        bool: True if processing succeeded, False otherwise.
    """
    import safetensors
    import safetensors.torch
    from msmodelslim.pytorch.weight_compression import CompressConfig, Compressor

    p = Path(file_path)
    if not p.exists():
        print(f"Error: File not found, failed to compress: {file_path}")
        return False

    try:
        state_dict = safetensors.torch.load_file(str(p))

        compress_config = CompressConfig(
            do_pseudo_sparse=False,
            sparse_ratio=1,
            is_debug=True,
            record_detail_root=str(p.parent),
            multiprocess_num=process_num,
        )
        compressor = Compressor(compress_config, weight=state_dict, quant_model_description=quant_desc)
        compressor.run()
        os.remove(p)
        compressor.export_safetensors(str(p.parent), safetensors_name=p.name)
        return True
    except Exception as e:
        print(f"Error processing Rank file {file_path}: {e}")
        return False


def main(args):
    # 1. Initial Validation
    # Validate early so the script doesn't fail after hours of inference
    output_dir = FileHandler.validate_path(args.output, must_exist=False, check_writable=True)
    model_dir = FileHandler.validate_path(args.model, must_exist=True)

    # 2. Run VLLM Engine and save sharded states
    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM(**dataclasses.asdict(engine_args))

    output_dir.mkdir(parents=True, exist_ok=True)
    llm.llm_engine.engine_core.save_sharded_state(path=str(output_dir))

    del llm
    clean_up()

    # 3. Migrate Metadata (Excluding large weights)
    for item in model_dir.iterdir():
        if item.suffix not in (".bin", ".pt", ".safetensors"):
            FileHandler.safe_copy(item, output_dir / item.name)

    # 4. Compression Logic
    parameters_map_fpath = output_dir / "parameters_type_map.json"
    if args.enable_compress:
        quant_desc = get_quant_description(str(parameters_map_fpath))
        quant_type = quant_desc["model_quant_type"]
        if quant_type in SUPPORTED_COMPRESS_QUANT_TYPE:
            # TODO: Implement w16a16sc
            if quant_type == "W16A16S":
                raise NotImplementedError("W16A16SC is not supported yet.")

            # Use Queue to collect worker results
            result_queue: Queue[Any] = mp.Queue()
            tasks = []
            for i in range(args.tensor_parallel_size):
                file_name = DEFAULT_PATTERN.format(rank=i, part="0")
                full_path = output_dir / file_name

                def worker_wrapper(file_path: str, qd: dict, pn: int, q: mp.Queue):
                    """Wrapper to put result into queue."""
                    result = weight_compress_worker(file_path, qd, pn)
                    q.put(result)

                p = mp.Process(
                    target=worker_wrapper, args=(str(full_path), quant_desc, args.compress_process_num, result_queue)
                )
                tasks.append(p)
                p.start()

            res = []
            for p in tasks:
                p.join()
                res.append(result_queue.get())

            if not all(res):
                raise RuntimeError("Compression failed. Check logs above for details.")
            ori_quant_desc_file = args.output / "quant_model_description.json"
            update_quant_description(str(ori_quant_desc_file))
            print("Compression completed successfully.")
        else:
            print(f"Skipping compression: Unsupported type {quant_type}")
    if parameters_map_fpath.exists():
        os.remove(parameters_map_fpath)


if __name__ == "__main__":
    args = parse_args()
    main(args)
