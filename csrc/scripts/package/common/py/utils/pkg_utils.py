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

"""基础构件。"""

import os 
from functools import partial
from itertools import chain, tee
from operator import methodcaller
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, TypeVar, Set

TOP_DIR = str(Path(__file__).resolve().parents[5])
TOP_SOURCE_DIR = TOP_DIR + '/scripts/'
DELIVERY_PATH = "build/_CPack_Packages/makeself_staging"
CONFIG_SCRIPT_PATH = 'package'
BLOCK_CONFIG_PATH = 'package/module'

SUCCESS = 0
FAIL = -1


A = TypeVar('A')


class PackageError(Exception):
    """打包异常基类。"""


class PackageConfigError(PackageError):
    """打包配置错误异常。"""


class BlockConfigError(PackageError):
    """块配置错误异常。"""


class ParseOsArchError(PackageError):
    """解析os_arch失败异常。"""


class EnvNotSupported(PackageError):
    """环境变量不支持异常。"""


class ContainAsteriskError(PackageError):
    """包含星号异常。"""

    def __init__(self, value: str):
        super().__init__()
        self.value = value


class FilelistError(PackageError):
    """文件列表异常。"""


class UnknownOperateTypeError(PackageError):
    """未知的操作类型。"""


class PackageNameEmptyError(PackageError):
    """包名为空错误。"""


class GenerateFilelistError(PackageError):
    """生成文件列表文件异常。"""

    def __init__(self, filename: str):
        super().__init__()
        self.filename = filename


class IllegalVersionDir(PackageError):
    """version_dir配置错误。"""


class CompressError(PackageError):
    """打包错误。"""

    def __init__(self, package_name: Optional[str]):
        super().__init__(package_name)
        self.package_name = package_name


def flatten(list_of_lists):
    """Flatten one level of nesting"""
    return chain.from_iterable(list_of_lists)


def merge_dict(base: Dict, *news: Dict):
    """合并两个字典。"""
    result = base.copy()
    for new in news:
        result.update(new)
    return result


def star_pipe(*funcs):
    """串联多个函数。解包结果。"""
    def pipe_func(*args, **k_args):
        result = funcs[0](*args, **k_args)
        for func in funcs[1:]:
            # 解包元组或列表结果
            result = func(*result)
        return result

    return pipe_func


def swap_args(func):
    """交换函数前两个参数。"""
    def inner(fst, snd, *args, **k_args):
        return func(snd, fst, *args, **k_args)
    return inner


def conditional_apply(predicate, func):
    """条件下应用函数。"""
    def conditional_apply_func(arg):
        if predicate(arg):
            return func(arg)
        return arg

    return conditional_apply_func


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def path_join(base: Optional, *others: str) -> Optional:
    """路径联合。"""
    if base is None:
        return None
    return os.path.join(base, *others)


def yield_if(data, predicate: Callable) -> Iterator:
    """条件满足则产生。"""
    if predicate(data):
        yield data


def config_feature_to_set(feature_str: str, feature_type: str = 'feature') -> Set[str]:
    """配置feature转换为集合。"""
    if feature_str is None:
        return set()

    if isinstance(feature_str, set):
        return feature_str

    if feature_str == '':
        raise PackageConfigError(f"Not allow to config {feature_type} empty.")

    features = set(feature_str.split(';'))
    if 'all' in features:
        raise PackageConfigError(f"Not allow to config {feature_type} all.")
    return features


def config_feature_to_string(features: Set[str]) -> str:
    """配置feature集合转换为字符串。"""
    if not features:
        return 'all'
    return ';'.join(sorted(features))
