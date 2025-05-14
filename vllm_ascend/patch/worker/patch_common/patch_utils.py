from typing import Callable, List, Optional, Tuple

import torch
from torch.library import Library
from vllm import utils
from vllm.utils import vllm_lib


def ascend_direct_register_custom_op(
        op_name: str,
        op_func: Callable,
        mutates_args: list[str],
        fake_impl: Optional[Callable] = None,
        target_lib: Optional[Library] = None,
        dispatch_key: str = "CUDA",
        tags: Tuple[torch.Tag, ...] = (),
):
    # In pytorch 2.5.1, torch.library.infer_schema require the input function to
    # have annotations supported by typing library. But in pytorch 2.7.0 which
    # vllm using, torch.library.infer_schema require the python builtin type. In
    # this case, we should revert built type to typing type for 2.5.1 backward
    # compatibility.
    for k, v in op_func.__annotations__.items():
        if v == list[int]:
            op_func.__annotations__[k] = List[int]
        if v == Optional[list[int]]:
            op_func.__annotations__[k] = Optional[List[int]]
        # TODO: add more type convert here if needed.
    import torch.library
    schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
    my_lib = target_lib or vllm_lib
    my_lib.define(op_name + schema_str, tags=tags)
    my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)


utils.direct_register_custom_op = ascend_direct_register_custom_op
