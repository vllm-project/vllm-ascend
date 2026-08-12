# APACE

## Project Overview

**APACE** (**A**scend **PA**rallel **C**ommunication-compute **E**ngine) is the architectural foundation for fused communication-compute operators on Ascend NPUs. It provides reusable modules for these operators and reduces development costs.

**Key benefits**:

- Easier development: Layered interfaces and reference implementations let developers compose fused operators without implementing communication-compute orchestration from scratch.
- Better performance: Tightly coupled communication and computation support pipeline overlap and maximize hardware utilization.
- Modular reuse: Reusable code modules can be called directly or used as reference implementations.

---

## Directory Structure

```
apace/
├── kernel/               # Fused communication-compute operator implementations
├── block/
│   ├── blaze_ext/        #   Blaze paradigm extensions
│   ├── aiv_comm/         #   AIV communication interfaces
│   └── aiv_compute/      #   AIV compute interfaces
├── basic/                # Basic data structures and abstractions
│   └── fragment_tensor/  #   Unified abstraction for multiple fragments
├── tiling/               # Tiling algorithms
├── utils/                # Common utilities and constants
├── tests/                # Tests
├── docs/                 # Design documentation
└── README.md
```

---

## Modules

### Kernel Operator Layer

The kernel layer implements fused communication-compute operators, including communication-compute pipeline orchestration, coordinated AIV-AIC scheduling, and synchronization. These implementations can be called directly or used as references.

### Block Interface Layer

Reusable single-core communication-compute interfaces, organized by implementation paradigm.

| Subdirectory | Description |
|:---|:---|
| `blaze_ext` | Extensions to the [ops-tensor](https://gitcode.com/cann/ops-tensor) Blaze library for fused communication-compute scenarios. |
| `aiv_comm` | Communication interfaces that use an AIV core as the initiating engine and support cross-device data transfers. |
| `aiv_compute` | AIV vector-compute interfaces, including quantization/dequantization, type conversion, and reduction, for processing data before and after communication. |

### Basic Layer

Provides common data structures and abstractions for the block and kernel layers.

| Subdirectory | Description |
|:---|:---|
| `fragment_tensor` | A unified abstraction for multiple GM fragments. It supports axis-based Slice, Copy, and Scatter operations to simplify cross-fragment data access. |

### tiling

Provides partitioning algorithms for Matmul and communication data.

### utils

A collection of common utilities, constants, and data structures.

---

## Relationship to MC2 Operators

Specific fused communication-compute operators under `mc2/`, such as `all_gather_matmul`, `matmul_all_reduce`, and `matmul_all_to_all`, can be implemented using APACE interfaces:

- The operator's `op_host/op_tiling` layer can use the partitioning algorithms in `apace/tiling` to determine tiling parameters.
- The operator's `op_kernel` layer can call an `apace/kernel` implementation directly or compose a fused kernel from `apace/block` interfaces.

```
mc2/<op>/op_host    ──┐
                      └──▶ apace/tiling       (tiling interfaces and partitioning algorithms)
mc2/<op>/op_kernel  ──┐
                      ├──▶ apace/kernel       (kernel interfaces for direct use or reference)
                      └──▶ apace/block        (block interfaces for composition)
```
