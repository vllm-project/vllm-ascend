# Triton operator layout

Place implementations in their functional directory under this package. Determine ownership from production callers. Runner operators may live at the package root or in a functional subdirectory. Shared model-layer kernels and utilities retain their functional directories. The shared upstream JIT registry and the generic warmup code serve these paths.

- [Triton operator reference](../../../docs/source/developer_guide/operator_reference/triton/index.md)
- [Contributing Triton operators](../../../docs/source/developer_guide/contribution/triton_operators.md)
