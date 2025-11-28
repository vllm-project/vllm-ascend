# Adding a custom aclnn operation

This document describes how to add a custom aclnn operation to vllm-ascend.

## How custom aclnn operation works in vllm-ascend?

Custom aclnn operations are built and installed into `vllm_ascend/cann_ops_custom` directory during the build process of vllm-ascend. Then the aclnn operators are bound to `torch.ops._C_ascend` module, enabling users to invoke them in vllm-ascend python code.

To enable custom operations, use the following code:

```python
from vllm_ascend.utils import enable_custom_op

enable_custom_op()
```

## How to add a custom aclnn operation?

- Create a new operation folder under `csrc` directory
- Create `op_host` and `op_kernel` directories for host and kernel source code
- Add build options in `csrc/build_aclnn.sh` for supported SOC. Note that multiple ops should be separated with `;`, i.e. `CUSTOM_OPS=op1;op2;op3`
- Bind aclnn operators to torch.ops._C_ascend module in `csrc/torch_binding.cpp`
- Write a meta implementation in `csrc/torch_binding_meta.cpp` for op being captured into aclgraph

After a successful build of vllm-ascend, the custom aclnn operation can be invoked in python code.
