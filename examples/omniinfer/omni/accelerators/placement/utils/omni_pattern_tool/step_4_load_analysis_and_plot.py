#!/usr/bin/env python
# coding: utf-8
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime

# Configure font to support Unicode characters for plotting
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    logging.warning("matplotlib is not installed. Unicode font settings are skipped, but this does not affect load analysis or plotting.")

def setup_logging(log_timestamp=None):
    """Set up logging configuration with a specified timestamp."""
    if log_timestamp is None:
        log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"pattern_generation_pipeline_{log_timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8', mode='a')
        ]
    )
    return logging.getLogger(__name__)

def analyze_device_load(placement_pattern: np.ndarray, load_array: np.ndarray, log_timestamp=None) -> pd.DataFrame:
    """
    Analyze device load based on placement pattern and load array.

    Args:
        placement_pattern: 3D numpy array of placement pattern.
        load_array: 2D numpy array of load data.
        log_timestamp: Timestamp for log file naming (optional).

    Returns:
        Pandas DataFrame with load data pivoted by layer and rank.
    """
    logger = setup_logging(log_timestamp)
    logger.info("Starting device load analysis.")

    num_ranks, num_layers, num_experts = placement_pattern.shape
    load_records = []

    for layer in range(num_layers):
        for rank in range(num_ranks):
            total_load = 0
            for ep in range(num_experts):
                if placement_pattern[rank, layer, ep] == 1:
                    ranks_with_ep = np.where(placement_pattern[:, layer, ep] == 1)[0]
                    num_deployment = len(ranks_with_ep)
                    if num_deployment > 0:
                        load_for_ep = load_array[layer, ep] / num_deployment
                        total_load += load_for_ep
            load_records.append({
                'layer': layer,
                'rank_id': rank,
                'load': total_load
            })

    df_load = pd.DataFrame(load_records)
    df_pivot = df_load.pivot(index='layer', columns='rank_id', values='load')
    df_pivot.rename(columns=lambda x: f"rank_{x}", inplace=True)
    df_pivot = df_pivot.reset_index(drop=True)
    logger.info(f"Device load analysis completed. DataFrame shape: {df_pivot.shape}")
    return df_pivot

def calculate_best_ep_per_layer(load_array: np.ndarray, num_ranks: int, log_timestamp=None) -> np.ndarray:
    """
    Calculate the best expert placement load per layer.

    Args:
        load_array: 2D numpy array of load data.
        num_ranks: Number of ranks.
        log_timestamp: Timestamp for log file naming (optional).

    Returns:
        Numpy array of best expert loads per layer.
    """
    logger = setup_logging(log_timestamp)
    num_layers, num_experts = load_array.shape
    total_load_per_layer = np.sum(load_array, axis=1)
    best_ep_per_layer = total_load_per_layer / num_ranks
    logger.info(f"Calculated best expert placement load per layer. Shape: {best_ep_per_layer.shape}")
    return best_ep_per_layer

def analyze_default_deployment_load(load_array: np.ndarray, num_ranks: int, log_timestamp=None) -> pd.DataFrame:
    """
    Analyze default deployment load.

    Args:
        load_array: 2D numpy array of load data.
        num_ranks: Number of ranks.
        log_timestamp: Timestamp for log file naming (optional).

    Returns:
        Pandas DataFrame with default deployment load data.
    """
    logger = setup_logging(log_timestamp)
    num_layers, num_experts = load_array.shape
    if num_experts % num_ranks != 0:
        logger.error("Number of experts must be divisible by number of ranks.")
        raise ValueError("Number of experts must be divisible by number of ranks.")

    experts_per_rank = num_experts // num_ranks
    default_load_records = []

    for layer in range(num_layers):
        layer_record = {}
        for rank in range(num_ranks):
            start = rank * experts_per_rank
            end = (rank + 1) * experts_per_rank
            total_load = np.sum(load_array[layer, start:end])
            layer_record[f"rank_{rank}"] = total_load
        default_load_records.append(layer_record)

    df_default = pd.DataFrame(default_load_records)
    df_default = df_default.reset_index(drop=True)
    logger.info(f"Default deployment load analysis completed. DataFrame shape: {df_default.shape}")
    return df_default

def plot_load_comparison_heatmaps_multi(
    optimized_df_lis: list,
    ppname_lis: list,
    figsize=(18, 8),
    num_ranks=32,
    dataset_name='Rand',
    save_path=None,
    log_timestamp=None
) -> None:
    """
    Plot multiple heatmaps comparing load distributions.

    Args:
        optimized_df_lis: List of DataFrames with optimized load data.
        ppname_lis: List of pattern names.
        figsize: Figure size.
        num_ranks: Number of ranks.
        dataset_name: Dataset name for plot title.
        save_path: Path to save the plot (optional).
        log_timestamp: Timestamp for log file naming (optional).
    """
    logger = setup_logging(log_timestamp)
    logger.info(f"Generating load comparison heatmaps for {len(optimized_df_lis)} patterns: {ppname_lis}")

    vmin = optimized_df_lis[0].min().min()
    vmax = vmin

    for df in optimized_df_lis:
        vmin = min(vmin, df.min().min())
        vmax = max(vmax, df.max().max())

    num_lis = len(optimized_df_lis)
    fig, axes = plt.subplots(1, num_lis, figsize=figsize)

    for i in range(num_lis):
        sns.heatmap(
            optimized_df_lis[i],
            ax=axes[i] if num_lis > 1 else axes,
            cmap="YlOrRd",
            vmin=vmin,
            vmax=vmax,
            cbar=True,
            cbar_kws={'shrink': 0.7}
        )
        title = ppname_lis[i] + '\n' + f'Dataset: {dataset_name}'
        (axes[i] if num_lis > 1 else axes).set_title(title, fontsize=10, pad=10)
        (axes[i] if num_lis > 1 else axes).set_xlabel("Rank ID", fontsize=10)
        (axes[i] if num_lis > 1 else axes).set_ylabel("Layer ID", fontsize=10)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.3)

    if save_path is not None:
        filename = f'Heat_{dataset_name}.png'
        save_file_path = os.path.join(save_path, filename)
        try:
            plt.savefig(save_file_path, bbox_inches='tight', dpi=100)
            plt.close()
            logger.info(f"Heatmap image saved to: {save_file_path}")
            print(f"Image generated successfully, saved to: {save_file_path}")
        except Exception as e:
            logger.error(f"Failed to save heatmap image to {save_file_path}: {e}")
            raise
    else:
        plt.show()
        logger.info("Heatmap displayed (not saved).")

def plot_max_load_comparison_lis(
    optimized_df_lis: list,
    ppname_lis: list,
    num_ranks=32,
    dataset_name='Rand',
    save_path=None,
    load_array=None,
    log_timestamp=None
) -> None:
    """
    Plot bar chart comparing maximum loads across layers.

    Args:
        optimized_df_lis: List of DataFrames with optimized load data.
        ppname_lis: List of pattern names.
        num_ranks: Number of ranks.
        dataset_name: Dataset name for plot title.
        save_path: Path to save the plot (optional).
        load_array: Load data array for best EP calculation (optional).
        log_timestamp: Timestamp for log file naming (optional).
    """
    logger = setup_logging(log_timestamp)
    logger.info(f"Generating max load comparison bar chart for {len(optimized_df_lis)} patterns: {ppname_lis}")

    max_lis = [df.max(axis=1) for df in optimized_df_lis]

    n_layers = optimized_df_lis[0].shape[0]
    layers = optimized_df_lis[0].index.astype(str)
    indices = np.arange(n_layers)
    bar_width = 0.85

    fig_width = max(12, n_layers * 0.3)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    bar_pos = -bar_width/2 + np.array([i * bar_width/len(optimized_df_lis) for i in range(len(optimized_df_lis))])

    for i in range(len(optimized_df_lis)):
        ax.bar(indices + bar_pos[i], max_lis[i], bar_width/len(optimized_df_lis),
               label=ppname_lis[i], color=f'C{i}')

    if load_array is not None:
        best_ep_per_layer = calculate_best_ep_per_layer(load_array, num_ranks, log_timestamp)
        min_best_ep = np.min(best_ep_per_layer)
        ax.axhline(
            y=min_best_ep,
            color='gray',
            linestyle='--',
            linewidth=1.9,
            alpha=0.9,
            label='Best EP'
        )
        max_y = max(max_lis[i].max() for i in range(len(max_lis)))
        line_y = min_best_ep
        ax.set_ylim(0, max(max_y, line_y) * 1.02)

    ax.set_xlabel('Layer ID')
    ax.set_ylabel('Load Balance Degree')
    ax.set_title(f'Load Balance Degree Comparison per Layer, Dataset: {dataset_name}')
    ax.set_xticks(indices)
    ax.set_xticklabels(layers, rotation=45)
    ax.legend()

    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    if save_path is not None:
        filename = f'Bars_{dataset_name}.png'
        save_file_path = os.path.join(save_path, filename)
        try:
            plt.savefig(save_file_path, bbox_inches='tight', dpi=100)
            plt.close()
            logger.info(f"Bar chart image saved to: {save_file_path}")
            print(f"Image generated successfully, saved to: {save_file_path}")
        except Exception as e:
            logger.error(f"Failed to save bar chart image to {save_file_path}: {e}")
            raise
    else:
        plt.show()
        logger.info("Bar chart displayed (not saved).")

def plot_unbalanced_ratio_comparison_lis(
    optimized_df_lis: list,
    ppname_lis: list,
    num_ranks=32,
    dataset_name='Rand',
    save_path=None,
    load_array=None,
    log_timestamp=None
) -> None:
    """
    Plot bar chart comparing unbalanced ratios (max_rank_load / avg_rank_load) across layers.
    """
    logger = setup_logging(log_timestamp)
    logger.info(f"Generating unbalanced ratio comparison bar chart for {len(optimized_df_lis)} patterns: {ppname_lis}")

    # Calculate unbalanced ratio: max_rank_load / avg_rank_load for each layer
    ratio_lis = []
    for df in optimized_df_lis:
        max_loads = df.max(axis=1)
        avg_loads = df.mean(axis=1)
        ratios = np.where(avg_loads != 0, max_loads / avg_loads, 1.0)
        ratio_lis.append(ratios)

    n_layers = optimized_df_lis[0].shape[0]
    layers = optimized_df_lis[0].index.astype(str)
    indices = np.arange(n_layers)
    bar_width = 0.85

    fig_width = max(12, n_layers * 0.3)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    bar_pos = - bar_width / 2 + np.array([i * bar_width / len(optimized_df_lis) for i in range(len(optimized_df_lis))])

    for i in range(len(optimized_df_lis)):
        ax.bar(indices + bar_pos[i], ratio_lis[i], bar_width / len(optimized_df_lis),
               label=ppname_lis[i], color=f'C{i}')

    # Add ideal line at y=1 (perfect balance)
    ax.axhline(
        y=1.0,
        color='gray',
        linestyle='--',
        linewidth=1.9,
        alpha=0.9,
        label='Best EP'
    )

    # Set y-axis limits
    max_y = max(ratio_lis[i].max() for i in range(len(ratio_lis)))
    ax.set_ylim(0, max_y * 1.02)

    ax.set_xlabel('Layer ID')
    ax.set_ylabel('Load Imbalance Degree (Max/Avg)')
    ax.set_title(f'Load Imbalance Degree Comparison per Layer, Dataset: {dataset_name}')
    ax.set_xticks(indices)
    ax.set_xticklabels(layers, rotation=45)
    ax.legend()

    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    if save_path is not None:
        filename = f'Unbalanced_Ratio_{dataset_name}.png'
        save_file_path = os.path.join(save_path, filename)
        try:
            plt.savefig(save_file_path, bbox_inches='tight', dpi=100)
            plt.close()
            logger.info(f"Unbalanced ratio bar chart image saved to: {save_file_path}")
            print(f"Image generated successfully, saved to: {save_file_path}")
        except Exception as e:
            logger.error(f"Failed to save unbalanced ratio bar chart image to {save_file_path}: {e}")
            raise
    else:
        plt.show()
        logger.info("Unbalanced ratio bar chart displayed (not saved).")

def calculate_max_load_reduction(
    optimized_df_lis: list,
    ppname_lis: list,
    save_path: str,
    dataset_name: str,
    log_timestamp=None
) -> pd.DataFrame:
    """
    Calculate maximum load reduction and save results to CSV.

    Args:
        optimized_df_lis: List of DataFrames with optimized load data.
        ppname_lis: List of pattern names.
        save_path: Path to save the CSV.
        dataset_name: Dataset name for CSV filename.
        log_timestamp: Timestamp for log file naming (optional).

    Returns:
        Pandas DataFrame with load reduction results.
    """
    logger = setup_logging(log_timestamp)
    logger.info("Calculating maximum load reduction.")

    max_loads = [df.max(axis=1) for df in optimized_df_lis]
    total_max_loads = [np.sum(max_load) for max_load in max_loads]
    default_total_max_load = total_max_loads[0]
    percentage_reductions = [
        ((default_total_max_load - total_max_load) / default_total_max_load) * 100
        for total_max_load in total_max_loads
    ]
    results_df = pd.DataFrame({
        'Placement Method': ppname_lis,
        'Load Balance Degree': total_max_loads,
        'Reduction Percentage': percentage_reductions
    })
    filename = f'Max_Load_Reduction_{dataset_name}.csv'
    save_file_path = os.path.join(save_path, filename)
    try:
        results_df.to_csv(save_file_path, index=False)
        logger.info(f"Max load reduction CSV saved to: {save_file_path}")
        print(f"CSV generated successfully, saved to: {save_file_path}")
    except Exception as e:
        logger.error(f"Failed to save max load reduction CSV to {save_file_path}: {e}")
        raise
    return results_df

def analyze_and_plot_deployments(
    load_file: str,
    pp_path_lis: list,
    ppname_lis: list,
    fig_save_path: str,
    num_ranks: int,
    dataset_name: str,
    log_timestamp=None
) -> None:
    """
    Analyze deployments and generate load distribution plots.

    Args:
        load_file: Path to load data CSV file.
        pp_path_lis: List of placement pattern file paths.
        ppname_lis: List of pattern names.
        fig_save_path: Path to save plots.
        num_ranks: Number of ranks.
        dataset_name: Dataset name for plot titles.
        log_timestamp: Timestamp for log file naming (optional).
    """
    logger = setup_logging(log_timestamp)
    logger.info(f"Starting deployment analysis for load file: {load_file}, patterns: {ppname_lis}")

    try:
        load_array = np.genfromtxt(load_file, delimiter=',', skip_header=1)[:, 1:]
        logger.info(f"Loaded load array with shape: {load_array.shape}")
    except Exception as e:
        logger.error(f"Failed to load CSV file {load_file}: {e}")
        raise

    placement_pattern_lis = []
    for path in pp_path_lis:
        if not os.path.exists(path):
            logger.error(f"Placement pattern file {path} does not exist.")
            raise ValueError(f"Placement pattern file {path} does not exist.")
        try:
            placement_pattern_lis.append(np.load(path))
            logger.info(f"Loaded placement pattern: {path}")
        except Exception as e:
            logger.error(f"Failed to load placement pattern {path}: {e}")
            raise

    df_lis = [analyze_default_deployment_load(load_array, num_ranks=num_ranks, log_timestamp=log_timestamp)]
    for placement_pattern in placement_pattern_lis:
        df_lis.append(analyze_device_load(placement_pattern, load_array, log_timestamp))

    plot_load_comparison_heatmaps_multi(
        optimized_df_lis=df_lis,
        ppname_lis=ppname_lis,
        figsize=(23, 10),
        num_ranks=num_ranks,
        dataset_name=dataset_name,
        save_path=fig_save_path,
        log_timestamp=log_timestamp
    )
    
    plot_unbalanced_ratio_comparison_lis(
        optimized_df_lis=df_lis,
        ppname_lis=ppname_lis,
        num_ranks=num_ranks,
        dataset_name=dataset_name,
        save_path=fig_save_path,
        load_array=load_array,
        log_timestamp=log_timestamp
    )

    plot_max_load_comparison_lis(
        optimized_df_lis=df_lis,
        ppname_lis=ppname_lis,
        num_ranks=num_ranks,
        dataset_name=dataset_name,
        save_path=fig_save_path,
        load_array=load_array,
        log_timestamp=log_timestamp
    )

    calculate_max_load_reduction(
        optimized_df_lis=df_lis,
        ppname_lis=ppname_lis,
        save_path=fig_save_path,
        dataset_name=dataset_name,
        log_timestamp=log_timestamp
    )
    logger.info("Deployment analysis and plotting completed.")

if __name__ == "__main__":
    main()