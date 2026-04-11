#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""函数基础库。"""

import operator
from typing import Callable, Iterator, TypeVar


A = TypeVar('A')


def constant(value: A) -> Callable[..., A]:
    """常量值。"""
    def constant_inner(*_args, **_kwargs) -> A:
        return value

    return constant_inner


def dispatch(*funcs):
    """分派应用。"""
    def dispatch_inner(*args, **kwargs) -> Iterator:
        return (func(*args, **kwargs) for func in funcs)

    return dispatch_inner


def pipe(*funcs):
    """串联多个函数。"""
    def pipe_func(*args, **k_args):
        result = funcs[0](*args, **k_args)
        for func in funcs[1:]:
            result = func(result)
        return result

    return pipe_func


def identity(value: A) -> A:
    """同一。"""
    return value


def invoke(func, *args, **kwargs):
    """调用。"""
    return func(*args, **kwargs)


def side_effect(*funcs):
    """调用函数，产生副作用，但不影响管道结果。"""
    def side_effect_func(arg):
        for func in funcs:
            # 不保留结果
            func(arg)
        return arg

    return side_effect_func


def star_apply(func):
    """列表展开再应用。"""
    def star_apply_func(arg):
        return func(*arg)

    return star_apply_func


def any_(*funcs) -> Callable:
    """高阶any。
    注意，any有短路效果。"""
    return pipe(
        dispatch(*funcs),
        any,
    )


def not_(func) -> Callable:
    """高阶not。"""
    return pipe(func, operator.not_)
