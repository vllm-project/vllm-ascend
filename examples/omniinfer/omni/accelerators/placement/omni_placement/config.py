# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from pathlib import Path

import yaml


class Config:
    def __init__(self, config_yaml_path):
        omni_config = self.load_and_validate_config(config_yaml_path)
        if omni_config:
            self._convert_dict_to_obj(omni_config)

    @staticmethod
    def load_and_validate_config(config_yaml_path):
        try:
            # 获取文件的绝对路径
            config_yaml_path = Path(config_yaml_path).absolute()
            # 打开文件并读取内容
            with open(config_yaml_path, mode='r', encoding='utf-8') as fh:
                omni_config = yaml.safe_load(fh)
            return omni_config
        except FileNotFoundError:
            print(f"文件 {config_yaml_path} 未找到。")
        except yaml.YAMLError as e:
            print(f"YAML 解析错误: {e}")
        return None

    def _convert_dict_to_obj(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                # 如果值是字典，递归转换为具有属性的对象
                sub_obj = Config.__new__(Config)
                sub_obj._convert_dict_to_obj(value)
                setattr(self, key, sub_obj)
            else:
                setattr(self, key, value)
    
    def getattr(self, key, default_value):
        return getattr(self, key, default_value)
