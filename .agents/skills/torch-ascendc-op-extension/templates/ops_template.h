#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

// _torch 后缀区分 PyTorch 接入层函数和 _kernel 后缀的 kernel 入口
namespace ascend_kernel {

at::Tensor xxx_torch(const at::Tensor& x1, const at::Tensor& x2);

} // namespace ascend_kernel

#endif // OPS_H
