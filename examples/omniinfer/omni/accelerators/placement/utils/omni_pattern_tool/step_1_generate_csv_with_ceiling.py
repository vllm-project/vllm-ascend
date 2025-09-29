#!/usr/bin/env python
# coding: utf-8
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.


import os
import csv
import glob
import numpy as np
import math
import re
from datetime import datetime
import logging
from typing import List, Optional, Tuple

# Configure font to support Chinese characters
try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    print("Warning: matplotlib is not installed, Chinese font settings will be skipped, but this does not affect CSV generation")

# Precompile regex pattern for log files
LOG_PATTERN = re.compile(
    r'\[dump activation\] (prefill|decode) step (\d+) in rank (\d+) for layer (\d+) get (\d+) experts data: ([\d\s]+)'
)

def parse_recordstep_range(recordstep_range: Optional[str]) -> Optional[Tuple[int, int]]:
    """Parse recordstep_range parameter (format: start:end)."""
    if recordstep_range is None or recordstep_range == "":
        return None
    try:
        start, end = map(int, recordstep_range.split(':'))
        if start < 0 or end < start:
            raise ValueError
        return start, end
    except ValueError:
        raise ValueError("recordstep_range must be in format 'start:end' where start and end are non-negative integers and start <= end.")

def validate_inputs(
    input_log_files: Optional[List[str]],
    input_txt_folders: Optional[List[str]], 
    input_mode: str,
    collecting_modes: str,
    num_ranks_of_collecting_data: Optional[int],
    num_positions_of_routed_experts: int,
    num_layers: int,
    recordstep_range: Optional[str]
) -> None:
    """Validate input parameters."""
    if input_mode not in ['log', 'txt']:
        raise ValueError("input_mode must be 'log' or 'txt'.")
    
    if input_mode == 'log':
        if not isinstance(input_log_files, list) or not input_log_files:
            raise ValueError("When input_mode='log', input_log_files must be a non-empty list of log file paths.")
        for log_file in input_log_files:
            if not os.path.exists(log_file):
                raise ValueError(f"Log file {log_file} does not exist.")
            if not log_file.endswith('.log'):
                logger.warning(f"Input file {log_file} does not have a .log extension, please verify the file type")
    else:
        if not isinstance(input_txt_folders, list) or not input_txt_folders:
            raise ValueError("When input_mode='txt', input_txt_folders must be a non-empty list of directory paths.")
        for folder in input_txt_folders:
            if not os.path.isdir(folder):
                raise ValueError(f"Text folder {folder} is not a valid directory.")
    
    if collecting_modes not in ['prefill', 'decode', 'all']:
        raise ValueError("collecting_modes must be 'prefill', 'decode', or 'all'.")
    
    if num_ranks_of_collecting_data is None:
        raise ValueError("num_ranks_of_collecting_data must be provided.")
    if num_ranks_of_collecting_data <= 0:
        raise ValueError("num_ranks_of_collecting_data must be a positive integer.")
    if num_positions_of_routed_experts % num_ranks_of_collecting_data != 0:
        raise ValueError(
            f"num_positions_of_routed_experts must be divisible by num_ranks_of_collecting_data. "
            f"Current num_positions_of_routed_experts={num_positions_of_routed_experts}, "
            f"num_ranks_of_collecting_data={num_ranks_of_collecting_data}"
        )
    if num_layers <= 0:
        raise ValueError("num_layers must be a positive integer.")
    
    parse_recordstep_range(recordstep_range)  # Validate recordstep_range format

def process_log_line(
    line: str,
    valid_modes: set,
    num_ranks_of_collecting_data: int,
    num_layers: int,
    numbers_per_rank: int,
    line_count: int,
    file_name: str,
    step_range: Optional[Tuple[int, int]]
) -> Optional[Tuple[int, int, List[int]]]:
    """Process a single log line and extract relevant data."""
    line = line.strip()
    if not line:
        return None

    match = LOG_PATTERN.match(line)
    if not match:
        return None

    mode, step, rank_id, layer_id, expert_count, expert_data = match.groups()
    if mode not in valid_modes:
        return None

    try:
        step = int(step)
        rank_id = int(rank_id)
        layer_id = int(layer_id)
        expert_count = int(expert_count)
    except ValueError:
        logger.warning(f"Unable to parse step, rank_id, layer_id, or expert_count, file {file_name} line {line_count}: {line}")
        return None

    # Filter by step range
    if step_range and not (step_range[0] <= step <= step_range[1]):
        return None

    if not (0 <= rank_id < num_ranks_of_collecting_data):
        logger.warning(f"rank_id {rank_id} out of range [0, {num_ranks_of_collecting_data-1}], file {file_name} line {line_count}: {line}")
        return None

    if not (0 <= layer_id < num_layers):
        logger.warning(f"layer_id {layer_id} out of range [0, {num_layers-1}], file {file_name} line {line_count}: {line}")
        return None

    try:
        expert_values = [int(x) for x in expert_data.strip().split('\t') if x.strip()]
    except ValueError:
        logger.warning(f"Invalid expert data format '{expert_data}', file {file_name} line {line_count}: {line}")
        return None

    if len(expert_values) != expert_count:
        logger.warning(f"Expert count {expert_count} does not match data count {len(expert_values)}, file {file_name} line {line_count}: {line}")
        return None

    if expert_count != numbers_per_rank:
        logger.warning(
            f"Expert count {expert_count} does not equal the number of experts per rank {numbers_per_rank}, "
            f"file {file_name} line {line_count}: {line}"
        )
        return None

    try:
        values = [math.ceil(float(num) / 128) for num in expert_values]
        return rank_id, layer_id, values
    except ValueError as e:
        logger.warning(f"Unable to process expert data '{expert_data}', file {file_name} line {line_count}, error: {e}")
        return None

def process_log_file(
    log_file: str,
    valid_modes: set,
    num_ranks_of_collecting_data: int,
    num_layers: int,
    numbers_per_rank: int,
    csv_data: np.ndarray,
    step_range: Optional[Tuple[int, int]]
) -> int:
    """Process a single log file and update csv_data."""
    processed_lines = 0
    line_count = 0

    for encoding in ['utf-8', 'gbk']:
        try:
            with open(log_file, 'r', encoding=encoding) as f:
                for line in f:
                    line_count += 1

                    result = process_log_line(
                        line, valid_modes, num_ranks_of_collecting_data, num_layers,
                        numbers_per_rank, line_count, log_file, step_range
                    )
                    if result:
                        rank_id, layer_id, values = result
                        csv_data[layer_id, rank_id*numbers_per_rank:(rank_id+1)*numbers_per_rank] += values
                        processed_lines += 1
            logger.info(f"Finished processing log file {log_file}, total lines: {line_count}, valid data lines: {processed_lines}")
            return processed_lines
        except UnicodeDecodeError:
            logger.warning(f"Unable to read {log_file} with {encoding} encoding, trying next encoding")
        except Exception as e:
            logger.error(f"Error occurred while reading {log_file}: {e}")
            raise
    logger.error(f"Unable to read {log_file}, all encodings failed")
    raise ValueError(f"Unable to read {log_file}")

def process_txt_file(
    txt_file: str,
    num_layers: int,
    numbers_per_rank: int,
    rank_id: int,
    csv_data: np.ndarray
) -> int:
    """Process a single txt file and update csv_data."""
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if len(lines) != num_layers:
            logger.warning(f"{txt_file} has {len(lines)} lines, expected {num_layers} lines")
            return 0

        processed_lines = 0
        for layer_idx, line in enumerate(lines):
            numbers = line.strip().split('\t')
            if len(numbers) != numbers_per_rank:
                logger.warning(f"Line {layer_idx+1} in {txt_file} has {len(numbers)} numbers, expected {numbers_per_rank}")
                continue

            try:
                values = [math.ceil(float(num) / 128) for num in numbers]
                csv_data[layer_idx, rank_id*numbers_per_rank:(rank_id+1)*numbers_per_rank] += values
                processed_lines += 1
            except ValueError:
                logger.warning(f"Invalid number format in line {layer_idx+1} of {txt_file}")
                continue

    logger.info(f"Finished processing text file {txt_file}, total lines: {len(lines)}, valid lines: {processed_lines}")
    return processed_lines

def generate_csv(
    input_log_files: Optional[List[str]] = None,
    input_txt_folders: Optional[List[str]] = None,
    input_mode: str = 'log',
    output_dir: str = './topk_id_count',
    collecting_modes: str = 'all',
    output_csv: Optional[str] = None,
    num_layers: int = 58,
    num_ranks_of_collecting_data: Optional[int] = None,
    num_positions_of_routed_experts: int = 256,
    log_timestamp: Optional[str] = None,
    recordstep_range: Optional[str] = None
) -> str:
    """
    Extract activation data from input files and generate a summarized CSV file.

    Parameters:
        input_log_files: List of log file paths (.log files, used when input_mode='log')
        input_txt_folders: List of folders containing .txt files (used when input_mode='txt')
        input_mode: Input mode, 'log' or 'txt', default 'log'
        output_dir: Directory for output CSV file
        collecting_modes: Data collection mode, 'prefill', 'decode', or 'all'
        output_csv: Output CSV filename (optional, default: topk_ids_count_<timestamp>_<collecting_modes>_step<start>to<end>.csv if recordstep_range is set)
        num_layers: Number of layers
        num_ranks_of_collecting_data: Number of ranks for data collection
        num_positions_of_routed_experts: Number of routed expert positions
        log_timestamp: Timestamp for log file naming (optional)
        recordstep_range: Range of recordstep or step values to process (format: start:end, optional)

    Returns:
        Path to the generated CSV file
    """
    # Set up logging
    if log_timestamp is None:
        log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"pattern_generation_pipeline_{log_timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    global logger
    logger = logging.getLogger(__name__)

    # Validate inputs
    validate_inputs(
        input_log_files, input_txt_folders, input_mode, collecting_modes,
        num_ranks_of_collecting_data, num_positions_of_routed_experts, num_layers,
        recordstep_range
    )

    # Parse recordstep_range
    step_range = parse_recordstep_range(recordstep_range)
    if step_range:
        logger.info(f"Processing steps in range: {step_range[0]} to {step_range[1]}")
    else:
        logger.info("No recordstep_range specified, processing all steps.")

    # Set default output_csv
    if not output_csv:
        step_suffix = f"_step{step_range[0]}to{step_range[1]}" if step_range else ""
        output_csv = f"topk_ids_count_{log_timestamp}_{collecting_modes}{step_suffix}.csv"

    # Ensure output directory exists
    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(output_dir):
        raise OSError(f"Unable to create output directory: {output_dir}")

    # Construct output path
    output_csv_path = os.path.join(output_dir, output_csv)
    output_csv_path = os.path.normpath(output_csv_path)
    logger.info(f"Attempting to write CSV file: {output_csv_path}")

    numbers_per_rank = num_positions_of_routed_experts // num_ranks_of_collecting_data
    csv_data = np.zeros((num_layers, num_ranks_of_collecting_data * numbers_per_rank), dtype=int)

    total_processed_lines = 0
    total_processed_files = 0

    if input_mode == 'log':
        valid_modes = {'prefill', 'decode'} if collecting_modes == 'all' else {collecting_modes}
        for log_file in input_log_files:
            processed_lines = process_log_file(
                log_file, valid_modes, num_ranks_of_collecting_data, num_layers,
                numbers_per_rank, csv_data, step_range
            )
            total_processed_lines += processed_lines
            total_processed_files += 1
    else:
        filtered_txt_files = []
        for txt_folder in input_txt_folders:
            txt_files = glob.glob(os.path.join(txt_folder, "activation_counts_recordstep_*.txt"))
            if not txt_files:
                logger.warning(f"No matching .txt files found in folder {txt_folder} "
                               f"(expected format: activation_counts_recordstep_*_rank_<rank_id>.txt)")
                continue

            for txt_file in txt_files:
                filename = os.path.basename(txt_file)
                try:
                    recordstep_str = filename.split('recordstep_')[1].split('_rank_')[0]
                    recordstep = int(recordstep_str)
                    if step_range is None or (step_range[0] <= recordstep <= step_range[1]):
                        filtered_txt_files.append(txt_file)
                except (IndexError, ValueError):
                    logger.warning(f"Skipping file with unexpected filename format: {filename}")
                    continue

        if not filtered_txt_files:
            logger.error(f"No .txt files across folders {input_txt_folders} match the recordstep range {step_range}")
            raise ValueError(f"No valid .txt files found in folders {input_txt_folders}, check recordstep range or folder contents")

        for txt_file in filtered_txt_files:
            filename = os.path.basename(txt_file)
            try:
                rank_id = int(filename.split('_rank_')[1].split('.txt')[0])
            except (IndexError, ValueError):
                logger.warning(f"Skipping file with unexpected filename format: {filename}")
                continue

            if not (0 <= rank_id < num_ranks_of_collecting_data):
                logger.warning(f"Skipping rank_id {rank_id}, not in range [0, {num_ranks_of_collecting_data-1}]")
                continue

            processed_lines = process_txt_file(txt_file, num_layers, numbers_per_rank, rank_id, csv_data)
            total_processed_lines += processed_lines
            total_processed_files += 1

        logger.info(f"Processed {total_processed_files} .txt files across {len(input_txt_folders)} folders")

    if total_processed_lines == 0:
        raise ValueError("No valid data extracted from any input files, please check input content, parameter configuration, or recordstep range")

    # Generate CSV
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            header = [''] + [f'ep_{i}' for i in range(num_ranks_of_collecting_data * numbers_per_rank)]
            writer.writerow(header)
            for layer_idx in range(num_layers):
                row = [f'layer_{layer_idx}'] + csv_data[layer_idx].tolist()
                writer.writerow(row)
        logger.info(f"CSV file generated successfully: {output_csv_path}")
        logger.info(f"Processed {total_processed_files} input files, total valid data lines: {total_processed_lines}")
        print(f"CSV file generated successfully: {output_csv_path}")
    except Exception as e:
        logger.error(f"Unable to write CSV file {output_csv_path}, error: {e}")
        raise

    return output_csv_path

if __name__ == "__main__":
    # parameters
    num_ranks = 4
    num_recordsteps = 2
    num_experts = 8
    num_layers = 4

    # make txt files folder
    output_dir = './input/txt/folder'
    os.makedirs(output_dir, exist_ok=True)

    # creat txt files
    for recordstep in range(num_recordsteps):
        for rank in range(num_ranks):
            # file names
            file_name = f'{output_dir}/activation_counts_recordstep_{recordstep}_rank_{rank}.txt'

            with open(file_name, 'w') as file:
                for _ in range(num_layers):
                    # random activation data
                    activation_data = [random.randint(10, 1000) for _ in range(2)]

                    line = '\t'.join(map(str, activation_data))

                    #write to file
                    file.write(line + '\n')

    print("files created successfully")
    # Test Example: Text file mode with multiple folders merged into one CSV
    
    generate_csv(
        input_txt_folders=["./input/txt/folder"],
        input_mode='txt',
        output_dir="./topk_id_count",
        collecting_modes="decode",
        output_csv="dataset.csv",
        num_layers=4,
        num_ranks_of_collecting_data=4,
        num_positions_of_routed_experts=8,
        recordstep_range=""
    )