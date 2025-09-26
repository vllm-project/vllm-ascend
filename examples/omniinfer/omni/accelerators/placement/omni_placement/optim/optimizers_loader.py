# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib

def create_optimizer_by_name(name: str, cluster_params, **kwargs):
    """
    根据自定义优化器类名动态创建优化器实例。

    参数:
        name (str): 自定义优化器类名（如 "TokenBalance", "AdaRouter"）。
        **kwargs: optimizer的参数 。

    返回:
        optimizer优化器实例。

    异常:
        ValueError: 如果优化器类不存在。
    """
    try:
        parts = name.split('.')
        module_name = f"omni_placement.optim.{parts[0]}"
        class_name = parts[1]

        # 动态导入模块
        module = importlib.import_module(module_name)
        # 从模块中获取类
        optimizer_class = getattr(module, class_name)

        return optimizer_class(cluster_params, **kwargs)
    except KeyError:
        raise ValueError(f"Unsupported optimizer: {name}")

def _create_optimizers(optimizer_config, cluster_params):
    """
    根据配置文件创建优化器列表。

    返回:
        list: 自定义优化器实例列表。
    """
    optimizers = []
    for optimizer_info in optimizer_config:
        opt_name = list(optimizer_info.keys())[0]
        opt_param = optimizer_info[opt_name]
        
        optimizer = create_optimizer_by_name(opt_name, cluster_params, **opt_param)
        optimizers.append(optimizer)
    return optimizers        