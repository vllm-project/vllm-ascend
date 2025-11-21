#!/usr/bin/env python
# coding: utf-8
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.


import argparse
import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

# Configure font to support Unicode characters for plotting
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    logging.warning("matplotlib is not installed. Unicode font settings are skipped, but this does not affect validation or visualization.")

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

def test_expert_mapping(placement_pattern, log_timestamp=None):
    """
    Test if the placement pattern satisfies the following conditions:
    1. For each layer, every expert is assigned at least once.
    2. For each layer, every rank has the same number of experts.

    Args:
        placement_pattern: 3D numpy array (shape: rank_id, layer_id, ep_id) or .npy file path,
                          value 1 indicates the expert exists on that rank and layer, 0 otherwise.
        log_timestamp: Timestamp for log file naming (optional).

    Returns:
        bool: True if conditions are satisfied, False otherwise.
        dict: Detailed test result information.
    """
    logger = setup_logging(log_timestamp)

    # Load placement_pattern
    if isinstance(placement_pattern, str):
        if not os.path.exists(placement_pattern):
            logger.error(f"Placement pattern file {placement_pattern} does not exist.")
            raise ValueError(f"Placement pattern file {placement_pattern} does not exist.")
        placement_pattern = np.load(placement_pattern)
    elif not isinstance(placement_pattern, np.ndarray):
        logger.error("placement_pattern must be a numpy array or .npy file path.")
        raise TypeError("placement_pattern must be a numpy array or .npy file path.")

    n_ranks, n_layers, n_experts = placement_pattern.shape
    result = {"Condition 1 satisfied": True, "Condition 2 satisfied": True, "Details": {}}

    for layer_id in range(n_layers):
        layer_result = {}
        expert_assigned = np.sum(placement_pattern[:, layer_id, :], axis=0)
        missing_experts = np.where(expert_assigned == 0)[0]

        if len(missing_experts) > 0:
            result["Condition 1 satisfied"] = False
            layer_result["Unassigned EPs"] = missing_experts.tolist()
            logger.warning(f"Layer {layer_id}: Unassigned experts detected: {missing_experts.tolist()}")

        experts_per_rank = np.sum(placement_pattern[:, layer_id, :], axis=1)
        if not np.all(experts_per_rank == experts_per_rank[0]):
            result["Condition 2 satisfied"] = False
            layer_result["EP counts per rank"] = experts_per_rank.tolist()
            layer_result["Inconsistent rank EP counts"] = True
            logger.warning(f"Layer {layer_id}: Inconsistent expert counts per rank: {experts_per_rank.tolist()}")

        if layer_result:
            result["Details"][f"layer_{layer_id}"] = layer_result
            logger.info(f"Layer {layer_id} test details: {layer_result}")

    logger.info(f"Placement pattern validation result: Condition 1 satisfied = {result['Condition 1 satisfied']}, "
                f"Condition 2 satisfied = {result['Condition 2 satisfied']}")
    return result["Condition 1 satisfied"] and result["Condition 2 satisfied"], result

def view_patterns(placement_pattern, ppname='', fig_save_path=None, log_timestamp=None):
    """
    Visualize summed views of a 3D placement pattern using a discrete color map.

    Args:
        placement_pattern: 3D numpy array (shape: rank_id, layer_id, ep_id) or .npy file path.
        ppname: Optional pattern name to include in the plot title.
        fig_save_path: Optional file path to save the generated image; if provided, the image is not displayed.
        log_timestamp: Timestamp for log file naming (optional).

    Returns:
        None (saves the image and logs a message).
    """
    logger = setup_logging(log_timestamp)

    # Load placement_pattern
    if isinstance(placement_pattern, str):
        if not os.path.exists(placement_pattern):
            logger.error(f"Placement pattern file {placement_pattern} does not exist.")
            raise ValueError(f"Placement pattern file {placement_pattern} does not exist.")
        placement_pattern = np.load(placement_pattern)
    elif not isinstance(placement_pattern, np.ndarray):
        logger.error("placement_pattern must be a numpy array or .npy file path.")
        raise TypeError("placement_pattern must be a numpy array or .npy file path.")

    matrix = placement_pattern
    dim_x, dim_y, dim_z = matrix.shape[0], matrix.shape[1], matrix.shape[2]
    sum_axis0 = np.sum(matrix, axis=0)
    sum_axis2 = np.sum(matrix, axis=2)

    logger.info(f"Generating visualization for pattern {ppname} with shape: {matrix.shape}")
    logger.info(f"Sum over Rank_ID dimension shape: {sum_axis0.shape}")
    logger.info(f"Sum over EP_ID dimension shape: {sum_axis2.shape}")

    fig2d, axs = plt.subplots(1, 2, figsize=(18, 5))
    cmap0 = plt.cm.get_cmap('plasma', dim_x + 1)
    cmap2 = plt.cm.get_cmap('plasma', dim_z + 1)

    im0 = axs[0].imshow(sum_axis0, origin='lower', cmap=cmap0, interpolation='nearest')
    axs[0].set_title(f'Sum over Rank_ID dimension {ppname}\n(Resulting shape: Layer_ID x EP_ID): View the number of times each expert is deployed')
    axs[0].set_xlabel('Expert ID')
    axs[0].set_ylabel('MoE Layer ID')
    fig2d.colorbar(im0, ax=axs[0], ticks=range(dim_x + 1))

    im2 = axs[1].imshow(sum_axis2.T, origin='lower', cmap=cmap2, interpolation='nearest')
    axs[1].set_title(f'Sum over EP_ID dimension {ppname}\n(Resulting shape: Layer_ID x Rank_ID): View the number of experts deployed per Rank')
    axs[1].set_xlabel('Rank ID')
    axs[1].set_ylabel('MoE Layer ID')
    fig2d.colorbar(im2, ax=axs[1], ticks=range(dim_z + 1))

    plt.tight_layout()

    if fig_save_path is not None:
        try:
            plt.savefig(fig_save_path, bbox_inches='tight', dpi=300)
            plt.close()
            logger.info(f"Visualization image saved to: {fig_save_path}")
            print(f"Image generated successfully, saved to: {fig_save_path}")
        except Exception as e:
            logger.error(f"Failed to save visualization image to {fig_save_path}: {e}")
            raise
    else:
        plt.show()
        logger.info("Visualization displayed (not saved).")

def main():
    parser = argparse.ArgumentParser(description="Test and visualize a placement pattern from an .npy file.")
    parser.add_argument('--placement_pattern_file', type=str, required=True,
                        help='Path to the .npy file containing the placement pattern.')
    parser.add_argument('--pattern_name', type=str, default='Sample Pattern',
                        help='Name of the placement pattern for visualization title.')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save the visualization image; if not provided, image is displayed.')
    
    args = parser.parse_args()

    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(log_timestamp)

    # Load and validate the placement pattern from .npy file
    if not os.path.exists(args.placement_pattern_file):
        logger.error(f"Placement pattern file {args.placement_pattern_file} does not exist.")
        raise ValueError(f"Placement pattern file {args.placement_pattern_file} does not exist.")
    
    try:
        placement_pattern = np.load(args.placement_pattern_file)
        logger.info(f"Loaded placement pattern from {args.placement_pattern_file} with shape: {placement_pattern.shape}")
    except Exception as e:
        logger.error(f"Failed to load placement pattern from {args.placement_pattern_file}: {e}")
        raise

    # Test the placement pattern
    is_valid, test_result = test_expert_mapping(placement_pattern, log_timestamp)
    logger.info(f"Placement pattern validation: Is valid = {is_valid}, Test result = {test_result}")
    print(f"Placement pattern validation: Is valid = {is_valid}, Test result = {test_result}")

    # Visualize the placement pattern
    view_patterns(
        placement_pattern,
        ppname=args.pattern_name,
        fig_save_path=args.save_path,
        log_timestamp=log_timestamp
    )

if __name__ == "__main__":
    main()