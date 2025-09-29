# OmniPlacement SDK

OmniPlacement is an advanced Software Development Kit (SDK) designed for dynamic expert placement in Mixture of Experts (MoE) systems for NPU-accelerated environments. It enables efficient, near-realtime expert placement with minimal impact on inference performance. Key features include:

1. **Expert Rearrangement**: Dynamically reconfigures experts to optimize resource utilization.
2. **Layer-Wise Uneven Expert Placement**: Supports non-uniform distribution of experts across layers for enhanced efficiency.
3. **Near-Realtime Placement**: Updates expert placements with negligible performance overhead.
4. **Near-Realtime Activation Capturing**: Captures expert activation in realtime

## Framework Capabilities
OmniPlacement provides a robust framework to streamline MoE operations:

1. **Realtime Expert Activation Monitoring**: Captures activation data to inform optimization strategies.
2. **Optimized Expert Placement**: Utilizes advanced algorithms to minimize MoE latency by determining optimal expert configurations.
3. **Dynamic Placement Updates**: Refreshes expert placements in near-realtime to adapt to changing workloads.

## Placement Optimization Objectives
The SDK focuses on two primary goals to enhance performance:

1. **Minimize Maximum Activation Load**: Balances expert activation across all NPUs to prevent bottlenecks.
2. **Reduce Communication Overhead**: Optimizes placement based on NPU cluster topology to lower inter-device communication costs.

## Expert Selection
1. **[Placeholder] Expert Selection Criteria**: Section to be completed, focusing on criteria for selecting experts based on workload and NPU capabilities.
2. **[Placeholder] Selection Algorithm**: Section to be completed, detailing the algorithm for prioritizing experts during placement.

## High Availability via Dynamic Redundant Expert Placement
OmniPlacement enhances system reliability by supporting dynamic redundant expert placement on existing NPU devices without service interruptions. This capability significantly reduces **Recovery Time Objective (RTO)** for large-scale MoE systems, ensuring rapid recovery from failures and maintaining service continuity.

## Deployment and Configuration Guide
[OmniPlacement Guideline](./Guideline.md) provides a clear, structured guide for deploying, enabling, and configuration expert load balancing for OmniPlacement, ensuring optimal performance, scalability, and dependability across diverse technical environments.