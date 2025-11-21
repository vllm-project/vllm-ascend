#!/usr/bin/env python
# coding: utf-8
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import argparse
import io
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from step_1_generate_csv_with_ceiling import generate_csv
from step_2_placement_pattern_generation import process_expert_deployments
from step_3_placement_pattern_checking_and_plot import (test_expert_mapping,
                                                        view_patterns)
from step_4_load_analysis_and_plot import analyze_and_plot_deployments

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def parse_recordstep_range(recordstep_range: str) -> tuple:
    """Parse recordstep_range into start and end integers."""
    if not recordstep_range:
        return None
    try:
        start, end = map(int, recordstep_range.split(':'))
        if start < 0 or end < start:
            raise ValueError
        return (start, end)
    except ValueError:
        raise ValueError(
            "recordstep_range must be in format 'start:end' where start and end are non-negative integers and start <= end."
        )


def main():
    parser = argparse.ArgumentParser(
        description=
        "Pipeline for processing expert deployment and load analysis.")
    parser.add_argument(
        '--input_log_files',
        type=str,
        nargs='+',
        default=['./activation_data.log'],
        help=
        'List of paths to input log files (e.g., activation_data.log), used when input_mode="log"'
    )
    parser.add_argument(
        '--input_txt_folders',
        type=str,
        nargs='+',
        default=['./decode'],
        help='List of folders containing .txt files, used when input_mode="txt"'
    )
    parser.add_argument(
        '--input_mode',
        type=str,
        default='txt',
        choices=['log', 'txt'],
        help='Input mode: "log" for log files, "txt" for text files')
    parser.add_argument('--topk_id_count_dir',
                        type=str,
                        default='./topk_id_count',
                        help='Directory for output CSV files')
    parser.add_argument('--placement_pattern_dir',
                        type=str,
                        default='./placement_pattern',
                        help='Directory for placement pattern files')
    parser.add_argument(
        '--placement_pattern_view_dir',
        type=str,
        default='./placement_pattern_view',
        help='Directory for placement pattern visualization files')
    parser.add_argument('--placement_pattern_analysis_dir',
                        type=str,
                        default='./placement_pattern_analysis',
                        help='Directory for load analysis plots')
    parser.add_argument(
        '--output_csv',
        type=str,
        default=None,
        help=
        'Name of the output CSV file in topk_id_count_dir (default: topk_ids_count_<timestamp>_<collecting_modes>_step<start>to<end>.csv)'
    )
    parser.add_argument('--num_layers',
                        type=int,
                        default=58,
                        help='Number of layers')
    parser.add_argument('--num_ranks_of_collecting_data',
                        type=int,
                        required=True,
                        help='Number of rank IDs for data collection')
    parser.add_argument('--num_positions_of_routed_experts',
                        type=int,
                        default=256,
                        help='Number of routed expert positions')
    parser.add_argument('--num_ranks_target_pattern',
                        type=int,
                        required=True,
                        help='Number of ranks for placement pattern')
    parser.add_argument(
        '--num_redundant_layers',
        type=int,
        nargs='*',
        default=[0, 10, 20, 30, 58],
        help='List of redundant or rearrange layers for batch processing')
    parser.add_argument('--expert_redundant_limit',
                        type=int,
                        default=11,
                        help='Maximum additional deployments per expert')
    parser.add_argument('--num_layers_target_pattern',
                        type=int,
                        default=58,
                        help='Number of layers for target pattern')
    parser.add_argument('--num_eps_target_pattern',
                        type=int,
                        default=256,
                        help='Number of experts per layer')
    parser.add_argument('--dataset_name',
                        type=str,
                        default='sharegpt',
                        help='Dataset name for plotting')
    parser.add_argument('--output_file_prefix',
                        type=str,
                        default='DSV3_0418_share_gpt_RedFullLays',
                        help='Prefix for output placement pattern filenames')
    parser.add_argument(
        '--pattern_mode',
        type=str,
        default='all',
        choices=['rearrange', 'redundant', 'all'],
        help=
        'Pattern generation mode: rearrange, redundant, or all (both modes)')
    parser.add_argument('--collecting_modes',
                        type=str,
                        default='all',
                        choices=['prefill', 'decode', 'all'],
                        help='Data collection mode: prefill, decode, or all')
    parser.add_argument(
        '--recordstep_range',
        type=str,
        default=None,
        help=
        'Range of recordstep or step values to process (format: start:end, e.g., 400:500)'
    )
    parser.add_argument(
        '--timestamp',
        type=str,
        default=None,
        help='Unified timestamp for file naming (format: YYYYMMDD_HHMMSS)')

    args = parser.parse_args()
    if args.recordstep_range == "":
        args.recordstep_range = None

    pattern_modes = ['rearrange', 'redundant'
                     ] if args.pattern_mode == 'all' else [args.pattern_mode]

    # Use the passed timestamp; generate a new one if not provided
    timestamp = args.timestamp if args.timestamp else datetime.now().strftime(
        "%Y%m%d_%H%M%S")

    # If dataset_name or output_file_prefix is an empty string, use the unified timestamp
    if args.dataset_name == "":
        args.dataset_name = timestamp
    if args.output_file_prefix == "":
        args.output_file_prefix = timestamp

    # Parse recordstep_range for suffix
    step_range = parse_recordstep_range(args.recordstep_range)
    step_suffix = f"_step{step_range[0]}to{step_range[1]}" if step_range else ""

    for dir_path in [
            args.topk_id_count_dir, args.placement_pattern_dir,
            args.placement_pattern_view_dir,
            args.placement_pattern_analysis_dir
    ]:
        dir_path = os.path.normpath(dir_path)
        if not dir_path:
            raise ValueError(f"Directory path cannot be empty: {dir_path}")
        print(f"Ensuring directory exists: {dir_path}")
        os.makedirs(dir_path, exist_ok=True)
        if not os.path.exists(dir_path):
            raise OSError(f"Failed to create directory: {dir_path}")

    # Validate input parameters
    if args.input_mode == 'log':
        input_log_files = [os.path.normpath(f) for f in args.input_log_files]
        for log_file in input_log_files:
            if not os.path.exists(log_file):
                raise ValueError(f"Log file {log_file} does not exist")
        input_txt_folders = None
    else:  # input_mode == 'txt'
        input_txt_folders = [
            os.path.normpath(f) for f in args.input_txt_folders
        ]
        for folder in input_txt_folders:
            if not os.path.isdir(folder):
                raise ValueError(
                    f"Text folder {folder} is not a valid folder path")
        input_log_files = None

    if not args.output_csv:
        args.output_csv = f"topk_ids_count_{timestamp}_{args.collecting_modes}{step_suffix}.csv"

    print(
        f"Step 1: Generating layer-EP count matrix to {args.topk_id_count_dir}"
    )
    generate_csv(
        input_log_files=input_log_files,
        input_txt_folders=input_txt_folders,
        input_mode=args.input_mode,
        output_dir=os.path.normpath(args.topk_id_count_dir),
        output_csv=args.output_csv,
        num_layers=args.num_layers,
        num_ranks_of_collecting_data=args.num_ranks_of_collecting_data,
        num_positions_of_routed_experts=args.num_positions_of_routed_experts,
        collecting_modes=args.collecting_modes,
        log_timestamp=timestamp,
        recordstep_range=args.recordstep_range)

    output_csv_path = Path(args.topk_id_count_dir) / args.output_csv
    output_csv_path = str(output_csv_path)

    pp_path_lis = []
    ppname_lis = ['Baseline']
    for num_red_layers in args.num_redundant_layers:
        for mode in pattern_modes:
            mode_suffix = f"_{args.collecting_modes}{step_suffix}"
            is_redundant = (mode == 'redundant')
            output_file = (
                f"placement_pattern_{timestamp}_{num_red_layers}_{mode}_layers_"
                f"{args.num_layers_target_pattern}_layers_{args.num_ranks_target_pattern}_ranks_"
                f"epmaxdeploy_{args.expert_redundant_limit+1}{mode_suffix}.npy"
                if is_redundant else
                f"placement_pattern_{timestamp}_{num_red_layers}_{mode}_layers_"
                f"{args.num_layers_target_pattern}_layers_{args.num_ranks_target_pattern}_ranks{mode_suffix}.npy"
            )

            output_path = os.path.normpath(
                os.path.join(args.placement_pattern_dir, output_file))
            pp_path_lis.append(output_path)
            ppname_lis.append(f'Pattern_{num_red_layers}_{mode}')

            print(
                f"Step 2: Generating placement pattern for num_special_layers={num_red_layers}, mode={mode}, saving to {output_path}"
            )
            process_expert_deployments(
                input_file=output_csv_path,
                output_dir=os.path.normpath(args.placement_pattern_dir),
                num_ranks_target_pattern=args.num_ranks_target_pattern,
                num_special_layers=num_red_layers,
                expert_redundant_limit=args.expert_redundant_limit,
                num_layers_target_pattern=args.num_layers_target_pattern,
                num_eps_target_pattern=args.num_eps_target_pattern,
                output_file=output_file,
                is_redundant=is_redundant,
                collecting_modes=args.collecting_modes,
                log_timestamp=timestamp,
                recordstep_range=args.recordstep_range)

    for pp_path, ppname in zip(pp_path_lis, ppname_lis[1:]):
        print(f"Step 3: Testing and visualizing pattern {pp_path}")
        if not os.path.exists(pp_path):
            print(f"Error: Placement pattern file {pp_path} does not exist.")
            raise OSError(f"Placement pattern file {pp_path} does not exist.")
        is_valid, test_result = test_expert_mapping(pp_path,
                                                    log_timestamp=timestamp)
        print(f"Pattern {ppname} valid: {is_valid}")
        print(f"Test result: {test_result}")

        fig_save_path = os.path.normpath(
            os.path.join(args.placement_pattern_view_dir,
                         f"{os.path.basename(pp_path)[:-4]}.png"))
        view_patterns(pp_path,
                      ppname=ppname,
                      fig_save_path=fig_save_path,
                      log_timestamp=timestamp)

    print(f"Step 4: Analyzing and plotting load distributions")
    analyze_and_plot_deployments(
        load_file=output_csv_path,
        pp_path_lis=pp_path_lis,
        ppname_lis=ppname_lis,
        fig_save_path=os.path.normpath(args.placement_pattern_analysis_dir),
        num_ranks=args.num_ranks_target_pattern,
        dataset_name=
        f"{args.num_ranks_target_pattern}_Ranks_{args.dataset_name}{step_suffix}",
        log_timestamp=timestamp)

    # Print log file location
    log_file_path = os.path.normpath(
        f"pattern_generation_pipeline_{timestamp}.log")
    print(f"All process logs are recorded in the log file at: {log_file_path}")


if __name__ == "__main__":
    main()
