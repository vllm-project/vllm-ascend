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

import copy
import glob
import hashlib
import itertools
import re
import xml.etree.ElementTree as ET
import os
import sys
from argparse import Namespace
from functools import partial
from io import StringIO
from itertools import chain
from operator import attrgetter, itemgetter, methodcaller
from typing import (
    Any, Callable, Dict, Iterable, Iterator, List, NamedTuple, Optional, Set, Tuple, Union
)

from .utils import pkg_utils
from .filelist import FileItem, FileList, fill_is_common_path
from .utils.pkg_utils import (
    ContainAsteriskError, FAIL, BLOCK_CONFIG_PATH,
    BlockConfigError, EnvNotSupported, IllegalVersionDir, PackageError,
    ParseOsArchError, config_feature_to_set,
    flatten, star_pipe, merge_dict, yield_if
)
from .utils.funcbase import constant, dispatch, invoke, pipe, star_apply
from .utils.comm_log import CommLog
from .version_info import (
    VersionInfo, VersionXml, VersionFormatNotMatch, is_multi_version
)

# 环境变量字典
EnvDict = Dict[str, str]

# 文件信息
FileInfo = Dict[str, str]

# 包属性
PackageAttr = Dict[str, Union[str, bool]]

# 生成信息
GenerateInfo = Dict[str, str]


class ParseOption(NamedTuple):
    """解析参数。"""
    os_arch: Optional[str]
    pkg_version: Optional[str]
    build_type: Optional[str]
    package_check: bool
    ext_name: str = ''


def parse_os_arch(os_arch: str) -> Tuple[str, str, str]:
    """解析系统和架构。"""
    match = re.match("^([a-z]+)(\\d+(\\.\\d+)*)?[.-]?(\\S*)", os_arch)
    if match:
        os_name = match.group(1)
        os_ver = match.group(2)
        if match.group(4):
            arch = match.group(4)
        else:
            # 如果os_arch中没有配置ARCH，ARCH默认值为aarch64
            arch = 'aarch64'

        return os_name, os_ver, arch

    raise ParseOsArchError()


def replace_env(env_dict: EnvDict, in_str: str):
    """替换环境变量为实际值。"""
    env_list = re.findall(".*?\\$\\((.*?)\\).*?", in_str)
    for env in env_list:
        if env == 'FILE':
            continue
        if env in env_dict:
            if env_dict[env] is not None:
                in_str = in_str.replace(f"$({env})", env_dict[env])
            else:
                in_str = in_str.replace(f"$({env})", '')
        else:
            raise EnvNotSupported(f"Error: {env} not supported.")
    return in_str


class ParseEnv(NamedTuple):
    """解析上下文环境。"""
    env_dict: EnvDict
    parse_option: ParseOption
    delivery_dir: str
    top_dir: str


class BlockElement(NamedTuple):
    """块配置。"""
    name: str
    block_conf_path: str
    dst_path: str
    chips: Set[str]
    features: Set[str]
    attrs: Dict[str, str]


# BlockElement直接透传给LoadedBlockElement的参数列表
BLOCK_ELEMENT_PASS_THROUGH_ARGS = ['dst_path', 'chips', 'features', 'attrs']


class LoadedBlockElement(NamedTuple):
    """加载后的块配置。"""
    root_ele: ET.Element
    use_move: bool
    dst_path: str
    chips: Set[str]
    features: Set[str]
    attrs: Dict[str, str]


class FileInfoParsedResult(NamedTuple):
    """file_info元素解析结果。"""
    file_info: FileInfo
    move_infos: List[FileInfo]
    dir_infos: List[Dict[str, str]]
    expand_infos: List[Dict[str, str]]


class BlockConfig(NamedTuple):
    """块配置。"""

    dir_install_list: List[Dict]
    move_files: List[FileInfo]
    expand_content_list: List[Dict]
    package_content_list: List[Dict]
    generate_infos: List[GenerateInfo]


class PackerConfig(NamedTuple):
    """安装相关配置。"""
    fill_is_common_path: Callable[[FileList], Iterator[FileItem]]


class XmlConfig(NamedTuple):
    """安装xml配置。"""
    default_config: Dict[str, str]
    package_attr: PackageAttr
    version_info: VersionInfo
    blocks: List[BlockConfig]
    version: str
    version_xml: Optional[VersionXml]
    packer_config: PackerConfig

    def _collect_list(self, list_name):
        result = []
        for block in self.blocks:
            result.extend(getattr(block, list_name))
        return result

    @property
    def dir_install_list(self):
        return self._collect_list('dir_install_list')

    @property
    def move_content_list(self):
        return self._collect_list('move_files')

    @property
    def expand_content_list(self):
        return self._collect_list('expand_content_list')

    @property
    def package_content_list(self):
        return self._collect_list('package_content_list')

    @property
    def generate_infos(self) -> List[GenerateInfo]:
        return self._collect_list('generate_infos')


# 默认包属性
DEFAULT_PACKAGE_ATTR = {
    'gen_version_info': True,
}


def parse_package_info(package_info_ele: Optional[ET.Element]) -> Dict:
    """解析package_info元素。"""

    def get_package_info_attrs(ele: ET.Element) -> Iterator[Tuple[str, Union[str, bool]]]:
        # expand_asterisk: 展开配置中星号
        # parallel: 并行复制文件
        # parallel_limit: 限制并发数
        # package_check: 检查filelist.csv中配置目录是否完整
        # check_features: 检查filelist.csv中所有feature是否符合package_check
        # gen_version_info: 是否生成version.info文件
        bool_attrs = (
            'expand_asterisk', 'parallel', 'parallel_limit', 'package_check', 'check_features',
            'use_move', 'gen_version_info'
        )
        bool_values = ('t', 'true', 'y', 'yes')
        if ele.tag in bool_attrs:
            if ele.text.lower() in bool_values:
                yield ele.tag, True
            else:
                yield ele.tag, False
        else:
            yield ele.tag, ele.text

    if not package_info_ele:
        return {}

    attr = dict(
        chain.from_iterable(map(get_package_info_attrs, list(package_info_ele)))
    )

    return attr


def parse_package_attr_by_args(args: Namespace) -> Dict:
    """通过命令行参数解析"""

    def pairs():
        if hasattr(args, 'chip_name') and args.chip_name:
            yield 'chip_name', args.chip_name
        if hasattr(args, 'suffix') and args.suffix:
            yield 'suffix', args.suffix
        if hasattr(args, 'func_name') and args.func_name:
            yield 'func_name', args.func_name

    return dict(pairs())


def parse_package_attr(root_ele: ET.Element, args: Namespace) -> Dict:
    """通过根元素解析package_info元素。"""
    package_info_ele = root_ele.find("package_info")
    return merge_dict(
        DEFAULT_PACKAGE_ATTR,
        parse_package_info(package_info_ele),
        parse_package_attr_by_args(args),
    )


def render_cann_version(a_ver: int,
                        b_ver: int,
                        c_ver: Optional[int],
                        d_ver: Optional[int],
                        e_ver: Optional[int],
                        f_ver: Optional[int]) -> str:
    """渲染CANN版本号。"""
    buffer = StringIO()
    buffer.write('(')
    buffer.write(f'({a_ver + 1} * 100000000) + ({b_ver + 1} * 1000000)')
    if c_ver is not None:
        buffer.write(f' + ({c_ver + 1} * 10000)')
    if d_ver is not None:
        buffer.write(f' + (({d_ver + 1} * 100) + 5000)')
    if e_ver is not None:
        buffer.write(f' + ({e_ver + 1} * 100)')
    if f_ver is not None:
        buffer.write(f' + {f_ver}')
    buffer.write(')')
    return buffer.getvalue()


def render_semver(package_name: str, version: str) -> Iterator[Tuple[str, str]]:
    """
    将语义化版本号转换为可比较的整数表达式，严格遵循SemVer规范

    排序规则：
    1. 正式版本 > 所有对应预发布版本（如 8.0.5 > 8.0.5-rc.1）
    2. 预发布类型优先级：rc > beta > alpha > 其他类型（如 rc.1 > beta.100）
    3. 同类型预发布版本：序号越大优先级越高（如 alpha.10 > alpha.2）
    4. 多段序号比较：从左到右逐段比较（如 alpha.1.2 > alpha.1.1）
    """
    expr_buffer = StringIO()
    expr_buffer.write('(')

    # 移除构建元数据（+后面的内容不影响版本优先级）
    version = version.split('+')[0]

    # 分离正式版本和预发布版本
    pre_release = None
    if '-' in version:
        release_part, pre_release = version.split('-', 1)
        release_part = release_part.split('.')
    else:
        release_part = version.split('.')
        if len(release_part) > 3:
            pre_release = '.'.join(release_part[3:])
            release_part = release_part[:3]

    # 解析正式版本号（主版本.次版本.修订号）
    try:
        major, minor, patch = map(int, release_part)
    except (ValueError, TypeError) as ex:
        raise IllegalVersionDir(f"无效的版本号格式: {version}") from ex

    yield f'{package_name}_VERSION_STR', f'"{version}"'
    yield f'{package_name}_MAJOR', str(major)
    yield f'{package_name}_MINOR', str(minor)
    yield f'{package_name}_PATCH', str(patch)
    # 计算基础版本值（主版本*10^7 + 次版本*10^5 + 修订号*10^3）
    # 预留10^3空间用于预发布版本，确保不同正式版本区间不重叠
    expr_buffer.write(f'({major} * 10000000) + ({minor} * 100000) + ({patch} * 1000)')

    # 处理正式版本（无预发布部分）
    if not pre_release:
        expr_buffer.write(')')
        yield f'{package_name}_PRERELEASE', '""'
        yield f'{package_name}_VERSION_NUM', expr_buffer.getvalue()
        return

    yield f'{package_name}_PRERELEASE', f'"{pre_release}"'

    # 预发布类型权重（值越小优先级越高）
    type_weights = {
        'rc': 100,  # rc优先级最高
        'beta': 200,  # beta次之
        'alpha': 300,  # alpha最低
    }

    def calc_pre_release() -> Tuple[int, int]:
        """计算预发布版本。"""
        if '.' in pre_release:
            pre_parts = pre_release.split('.')
            pre_type = pre_parts[0]  # 提取预发布类型（rc/beta/alpha等）

            # 提取序号部分（支持多段序号，非数字部分忽略）
            pre_nums = []
            for part in pre_parts[1:]:
                if part.isdigit():
                    pre_nums.append(int(part))
            if not pre_nums:  # 无序号时默认0
                pre_nums = [0]

            # 未知类型权重设为400（优先级低于alpha）
            pre_type_weight = type_weights.get(pre_type, 400)

            # 计算序号值（支持多段和多位数）
            num_str = ''.join(map(str, pre_nums))
            # 转换为整数并返回
            num_value = int(num_str)
            return pre_type_weight, num_value

        for pre_type in type_weights:
            if pre_release.startswith(pre_type):
                pre_type_weight = type_weights[pre_type]
                num_value = int(pre_release[len(pre_type):])
                return pre_type_weight, num_value

        return None, None

    try:
        pre_type_weight, num_value = calc_pre_release()
    except (ValueError, TypeError) as ex:
        raise IllegalVersionDir(f"无效的预发布版本: {pre_release}") from ex

    if not pre_type_weight:
        raise IllegalVersionDir(f"无效的预发布版本: {pre_release}")

    # 预发布版本最终值 = 基础值 - 类型权重 + 序号值
    # 确保：预发布值 < 基础值（正式版本）
    expr_buffer.write(f' - {pre_type_weight} + {num_value}')
    expr_buffer.write(')')

    yield f'{package_name}_VERSION_NUM', expr_buffer.getvalue()


def get_cann_version_info(name: str, version: str) -> Iterator[Tuple[str, str]]:
    """获取CANN版本号信息。"""
    version_info = []

    # 删除字符串中的_VERSION
    package_name = name[:-8]

    if not version:
        yield f'{package_name}_VERSION_STR', '"0"'
        return

    yield from render_semver(package_name, version)


def get_default_env_items() -> Iterator[Tuple[str, str]]:
    """获取默认环境字典条目。"""
    yield 'VERSION_DIR', ''
    yield 'HOME', os.environ.get('HOME')


def get_env_items_by_version(version: Optional[str]) -> Iterator[Tuple[str, str]]:
    """根据version获取环境字典条目。"""
    if version:
        yield 'ASCEND_VER', version

        version_parts = version.split('.')
        for idx in range(1, len(version_parts) + 1):
            yield f'CUR_VER[{idx}]', '.'.join(version_parts[:idx])
        yield 'CUR_VER', version
        yield 'LOWER_CUR_VER', version.lower()


def get_env_items_by_version_dir(version_dir: Optional[str]) -> Iterator[Tuple[str, str]]:
    """根据version_dir获取环境字典条目。"""
    if version_dir:
        yield 'VERSION_DIR', version_dir


def get_os_arch_default_env_items() -> Iterator[Tuple[str, str]]:
    """获取系统相关默认环境字典条目。"""
    yield 'OS_NAME', 'linux'
    yield 'OS_VER', ''
    yield 'ARM', 'aarch64'
    yield 'TARGET_ENV', '$(TARGET_ENV)'


def get_env_items_by_os_arch(os_arch: str) -> Iterator[Tuple[str, str]]:
    """根据os_arch获取环境字典条目。"""
    if os_arch:
        os_name, os_ver, arch = parse_os_arch(os_arch)
        yield 'OS_NAME', os_name
        yield 'OS_VER', os_ver
        yield 'ARCH', arch
        yield 'OS_ARCH', os_arch
        if arch in ('arm', 'sw_64'):
            yield 'ARM', arch
        else:
            yield 'ARM', 'aarch64'
        yield 'TARGET_ENV', f"{arch}-linux"
    else:
        yield from get_os_arch_default_env_items()


def get_env_items_by_timestamp(timestamp: Optional[str]) -> Iterator[Tuple[str, str]]:
    """根据timestamp获取环境字典条目。"""
    if timestamp:
        yield 'TIMESTAMP', timestamp
        yield 'TIMESTAMP_NO', timestamp.replace('_', '')
    else:
        yield 'TIMESTAMP', '0'
        yield 'TIMESTAMP_NO', '0'


def parse_env_dict(os_arch: str,
                   package_attr: PackageAttr,
                   version: Optional[str],
                   version_dir: Optional[str],
                   timestamp: Optional[str]) -> EnvDict:
    """解析环境变量字典。"""
    env_dict = dict(
        chain(
            get_default_env_items(),
            yield_if(('ARCH', package_attr.get('default_arch')), itemgetter(1)),
            get_env_items_by_os_arch(os_arch),
            get_env_items_by_version(version),
            get_env_items_by_version_dir(version_dir),
            yield_if(('VERSION_DIR', version_dir), constant(version_dir)),
            get_env_items_by_timestamp(timestamp),
        )
    )

    return env_dict


def get_timestamp(args: Namespace) -> Optional[str]:
    """获取触发时间戳。"""
    if 'tag' not in args:
        return None

    tag = args.tag
    if tag:
        timestamp_re = r"\d{8}_\d{9}"
        timestamp_list = re.findall(timestamp_re, tag)
        if not timestamp_list:
            raise PackageError("The {} format is incorrect.".format(tag))
        timestamp = timestamp_list[-1]
    else:
        timestamp = None
    return timestamp


def extract_element_attrib(ele: ET.Element) -> Dict:
    """提取元素属性。"""
    return ele.attrib.copy()


def extract_generate_info_content(generate_info_ele: ET.Element, env_dict: EnvDict) -> Dict:
    """提取生成信息内容。"""
    file_content = {
        sub_item.tag: replace_env(env_dict, sub_item.text)
        for sub_item in list(generate_info_ele)
    }
    return {
        'content': file_content
    }


def parse_generate_infos_by_loaded_block(loaded_block: LoadedBlockElement,
                                         default_config: Dict[str, str],
                                         env_dict: EnvDict) -> List[Dict]:
    """根据根元素解析生成信息列表。"""
    return invoke(
        pipe(
            partial(
                map, pipe(
                    dispatch(
                        pipe(
                            extract_element_attrib,
                            partial(merge_dict, default_config),
                            partial(evaluate_info, loaded_block=loaded_block, env_dict=env_dict),
                        ),
                        partial(extract_generate_info_content, env_dict=env_dict),
                    ),
                    star_apply(merge_dict),
                )
            ),
            list,
        ),
        loaded_block.root_ele.findall('generate_info')
    )


def join_pkg_inner_softlink(link_str_list: List[str]) -> str:
    """合并pkg_inner_softlink"""
    path = "/".join(link_str_list)
    return os.path.normpath(path)


def check_contain_asterisk(value: str) -> bool:
    """检查串是否包含星号。"""
    if '*' in value:
        return True
    return False


def check_value(value: str,
                package_check: bool,
                package_attr: PackageAttr):
    """检查元素value属性。"""
    if package_check and package_attr.get('suffix') == 'run':
        if check_contain_asterisk(value):
            raise ContainAsteriskError(value)


def get_dst_prefix(file_info: FileInfo, env: ParseEnv) -> str:
    """获取文件的前缀。"""
    return os.path.join(env.delivery_dir, file_info['dst_path'])


def get_dst_target(file_info: FileInfo, env: ParseEnv) -> str:
    """获取文件的实际路径。"""
    dst_prefix = get_dst_prefix(file_info, env)
    return os.path.join(dst_prefix, os.path.basename(file_info.get('value')))


def make_hash(filepath: str) -> str:
    """计算文件的hash(sha256)值。"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as file:
        sha256_hash.update(file.read())

    return sha256_hash.hexdigest()


def config_hash(parsed_result: FileInfoParsedResult, env: ParseEnv):
    """配置hash值。"""
    file_info = parsed_result.file_info
    # 如果配置了configurable，需要计算文件的hash值
    if file_info and file_info['configurable'] == 'TRUE':
        src_target = get_dst_target(file_info, env)
        hash_value = make_hash(src_target)
        file_info['hash'] = hash_value
    return parsed_result


def apply_func(func: Callable[[str], str],
               value: Union[List[str], Set[str], str]
               ) -> Union[List[str], Set[str], str]:
    """对一个字符串，或字符串序列，应用函数。"""
    # 如：pkg_softlink列表
    if isinstance(value, list):
        return list(map(func, value))
    # 如：feature集合
    if isinstance(value, set):
        return set(map(func, value))
    return func(value)


REAL_PREFIX = 'real:'


def join_dst_path(base: str, other: str) -> str:
    """联结dst_path。"""
    if other.startswith('real:'):
        other = other[len(REAL_PREFIX):]
        return other
    return os.path.join(base, other)


def evaluate_info(info: Dict[str, str],
                  loaded_block: LoadedBlockElement,
                  env_dict: EnvDict) -> Dict[str, str]:
    """info元素求值。"""
    dst_keys = ('dst_path',)

    replace_env_func = partial(replace_env, env_dict)
    add_dst_path_func = partial(join_dst_path, loaded_block.dst_path)

    def upper_value(key: str, value: str) -> Tuple[str, str]:
        if key == 'configurable':
            return key, value.upper()
        return key, value

    def add_dst_path(key: str, value: str) -> Tuple[str, str]:
        if key in dst_keys:
            return key, apply_func(add_dst_path_func, value)
        return key, value

    def replace_pkg_inner_softlink(key: str, value: str) -> Tuple[str, str]:
        if key == 'pkg_inner_softlink':
            return key, 'NA'
        return key, value

    def merge_feature(key: str, value: str) -> Tuple[str, str]:
        if key in ('chip', 'feature'):
            config_features = config_feature_to_set(value, key)
            return key, config_features | getattr(loaded_block, f'{key}s')
        return key, value

    def eval_value(_key: str, value: str) -> str:
        if value is not None:
            return apply_func(replace_env_func, value)

    eval_value_func = star_pipe(
        upper_value, add_dst_path, replace_pkg_inner_softlink,
        merge_feature, eval_value,
    )

    return {
        key: eval_value_func(key, value)
        for key, value in
        itertools.chain(
            # 默认值配置
            [
                ('dst_path', ''), ('configurable', 'FALSE'),
                ('chip', None), ('feature', None), ('pkg_feature', None),
            ],
            info.items()
        )
    }


def parse_dir_info_elements(loaded_block: LoadedBlockElement,
                            default_config: Dict[str, str],
                            package_attr: PackageAttr,
                            env: ParseEnv) -> List[Dict[str, str]]:
    """解析dir_info元素。"""
    dir_info_elements: List[ET.Element] = loaded_block.root_ele.findall('dir_info')
    dir_infos = []
    for item in dir_info_elements:
        dir_config = default_config.copy()
        dir_config.update(item.attrib)
        dir_config['module'] = dir_config.get('value')
        for sub_item in list(item):
            dir_info = dir_config.copy()
            dir_info.update(sub_item.attrib)
            dir_info = evaluate_info(dir_info, loaded_block, env.env_dict)
            check_value(
                dir_info['value'], env.parse_option.package_check, package_attr
            )
            dir_infos.append(dir_info)

    return dir_infos


def expand_dir(file_info: FileInfo, get_dst_target_func: Callable[[FileInfo], str]):
    """
    如果file_info中配置的路径是文件夹，需要展开到文件
    """
    file_info_list = []
    dir_info_list = []
    dst_target = get_dst_target_func(file_info)

    value_list = file_info.get('value').split('/')
    target_name = value_list[-1] if value_list[-1] else value_list[-2]

    # 这里把当前目录也加入到dir_info_list中
    dir_info_copy = file_info.copy()
    dir_info_copy['module'] = file_info.get('value')
    dir_info_copy['value'] = os.path.join(
        file_info.get('install_path', ''), target_name
    )

    # 子目录的权限按照xml中subdir_mod配置，如果没有配置subdir_mod按照install_mod配置
    subdir_mod = file_info.get("subdir_mod", None)
    if subdir_mod is not None:
        dir_info_copy['install_mod'] = subdir_mod
    # 被展开的当前目录不需要设置softlink
    dir_info_copy['install_softlink'] = 'NA'
    dir_info_copy['pkg_inner_softlink'] = 'NA'
    dir_info_list.append(dir_info_copy)

    for root, dirs, files in os.walk(dst_target, followlinks=True):
        # 不同操作系统上，os.walk遍历的结果顺序会略有不同，这里按字母排序，保证不同系统一致
        dirs.sort()
        files.sort()

        dirs_to_remove = []
        for name in dirs:
            dirname = os.path.join(root, name)
            # 如果是指向目录的软连接，则按照文件处理，无需在安装时创建目录，只需要卸载时删除就行
            if os.path.islink(dirname) and not need_dereference(file_info):
                copy_file_info = create_file_info(dirname, dst_target, file_info, name, target_name)
                # 被展开的子文件不需要设置softlink
                copy_file_info['install_softlink'] = 'NA'
                copy_file_info['pkg_inner_softlink'] = 'NA'
                file_info_list.append(copy_file_info)
                dirs_to_remove.append(name)
                continue
            relative_dirname = os.path.relpath(dirname, dst_target)
            dir_info_copy = file_info.copy()
            dir_info_copy['module'] = file_info.get('value')
            dir_info_copy['value'] = os.path.join(
                file_info.get('install_path', ''), target_name, relative_dirname
            )
            # 被展开的子目录不需要设置softlink
            dir_info_copy['install_softlink'] = 'NA'
            dir_info_copy['pkg_inner_softlink'] = 'NA'
            # 子目录的权限按照xml中subdir_mod配置，如果没有配置subdir_mod按照install_mod配置
            subdir_mod = file_info.get("subdir_mod", None)
            if subdir_mod is not None:
                dir_info_copy['install_mod'] = subdir_mod
            dir_info_list.append(dir_info_copy)
        for name in files:
            filename = os.path.join(root, name)
            copy_file_info = create_file_info(filename, dst_target, file_info, name, target_name)
            file_info_list.append(copy_file_info)

        for name in dirs_to_remove:
            dirs.remove(name)
    return file_info_list, dir_info_list


def create_file_info(dirname, dst_target, file_info, name, target_name):
    relative_filename = os.path.relpath(dirname, dst_target)
    relative_dir_name = os.path.split(relative_filename)[0]
    copy_file_info = file_info.copy()
    copy_file_info['value'] = name
    copy_file_info['src_path'] = os.path.join(
        file_info['src_path'], file_info['value'], relative_dir_name
    )
    copy_file_info['dst_path'] = os.path.join(
        file_info['dst_path'], target_name, relative_dir_name
    )
    copy_file_info['install_path'] = os.path.join(
        file_info.get('install_path', ''), target_name, relative_dir_name
    )
    return copy_file_info


def expand_file_info_asterisk(parsed_result: FileInfoParsedResult,
                              env: ParseEnv) -> Iterator[FileInfoParsedResult]:
    """展开FileInfoParsedResult中的星号。"""
    file_info = parsed_result.file_info
    if check_contain_asterisk(file_info.get('value', '')):
        dst_prefix = get_dst_prefix(file_info, env)
        dst_targets = sorted(glob.glob(get_dst_target(file_info, env)))
        if 'exclude' in file_info:
            exclude = list(map(methodcaller('strip'), file_info['exclude'].split(';')))
        else:
            exclude = []
        for dst_target in dst_targets:
            value = os.path.relpath(dst_target, dst_prefix)
            if value in exclude:
                continue
            new_file_info = file_info.copy()
            new_file_info['value'] = value
            if 'pkg_inner_softlink' in new_file_info:
                # pkg_inner_softlink中的特殊变量$(FILE)替换为展开后的文件名
                pkg_inner_softlink = new_file_info['pkg_inner_softlink']
                new_file_info['pkg_inner_softlink'] = pkg_inner_softlink.replace(
                    '$(FILE)', os.path.basename(dst_target)
                )
            yield parsed_result._replace(file_info=new_file_info)
    else:
        yield parsed_result


def trans_to_stream(item: Any) -> Iterator[Any]:
    """转换为流。"""
    yield item


def need_dereference(file_info: FileInfo) -> bool:
    """是否需要解引用。"""
    if 'dereference' in file_info:
        return True
    return False


def need_expand(file_info: FileInfo, get_dst_target_func: Callable[[FileInfo], str]) -> bool:
    """是否需要展开子目录。"""
    if file_info.get('entity') == 'true':
        return False
    dst_target = get_dst_target_func(file_info)
    if os.path.isdir(dst_target):
        if need_dereference(file_info):
            return True
        if os.path.islink(dst_target):
            return False
        return True
    return False


def expand_file_info(parsed_result: FileInfoParsedResult,
                     use_move: bool,
                     get_dst_target_func: Callable[[FileInfo], str]) -> FileInfoParsedResult:
    """展开FileInfoParsedResult中的目录。"""
    file_info = parsed_result.file_info
    if need_expand(file_info, get_dst_target_func):
        # 如果当前是文件夹，需要展开计算
        expand_infos, dir_infos = expand_dir(file_info, get_dst_target_func)
        # 实测发现，对于opp包，整体目录cp的安装速度要快于目录中各文件mv
        # 可能的原因是，cp遍历目录的速度较快，并且目录中的文件都比较小。mv依赖shell迭代目录中的所有文件。
        return FileInfoParsedResult(
            merge_dict(file_info, {'is_dir': True}), [], dir_infos, expand_infos
        )

    if use_move:
        return FileInfoParsedResult(
            {}, [file_info], parsed_result.dir_infos, parsed_result.expand_infos
        )

    return parsed_result


def trans_file_info_to_result(file_info: FileInfo) -> FileInfoParsedResult:
    """file_info转换为FileInfoParsedResult。"""
    return FileInfoParsedResult(file_info, [], [], [])


def parse_file_element(file_ele: ET.Element,
                       file_config: Dict[str, str],
                       loaded_block: LoadedBlockElement,
                       package_attr: PackageAttr,
                       env: ParseEnv) -> Iterator[FileInfoParsedResult]:
    """解析file元素。"""
    file_info = merge_dict(file_config, file_ele.attrib)
    file_info = evaluate_info(file_info, loaded_block, env.env_dict)

    if package_attr.get('expand_asterisk', False):
        expand_asterisk_func = partial(expand_file_info_asterisk, env=env)
    else:
        expand_asterisk_func = trans_to_stream

    if 'install_path' not in file_info:
        file_info['install_path'] = ''

    trans_file_info_func = pipe(
        trans_file_info_to_result,
        expand_asterisk_func,
        partial(map, partial(config_hash, env=env)),
        partial(
            map,
            partial(
                expand_file_info,
                use_move=loaded_block.use_move,
                get_dst_target_func=partial(get_dst_target, env=env)
            )
        ),
    )

    yield from trans_file_info_func(file_info)


def parse_file_info_elements(loaded_block: LoadedBlockElement,
                             default_config: Dict[str, str],
                             package_attr: PackageAttr,
                             env: ParseEnv) -> Iterator[FileInfoParsedResult]:
    """解析file_info元素。"""
    file_info_elements: List[ET.Element] = loaded_block.root_ele.findall('file_info')
    for file_info_ele in file_info_elements:
        file_config = merge_dict(
            default_config,
            file_info_ele.attrib,
            {'module': file_info_ele.attrib.get('value')}
        )

        for sub_item in list(file_info_ele):
            yield from parse_file_element(
                sub_item, file_config, loaded_block, package_attr, env
            )


def unique_infos(infos: Iterable) -> List[Dict[str, str]]:
    """infos去重。"""
    cache: Set[str] = set()
    new_infos = []
    for info in infos:
        if info['value'] in cache:
            continue
        cache.add(info['value'])
        new_infos.append(info)

    return new_infos


def parse_block_config(loaded_block: LoadedBlockElement,
                       package_attr: PackageAttr,
                       parse_env: ParseEnv):
    """解析块配置。"""
    default_config = copy.copy(loaded_block.attrs)
    default_config.update(loaded_block.root_ele.attrib)

    dir_infos = parse_dir_info_elements(
        loaded_block,
        default_config,
        package_attr,
        parse_env,
    )
    file_info_results = list(
        chain(
            parse_file_info_elements(
                loaded_block,
                default_config,
                package_attr,
                parse_env,
            )
        )
    )

    generate_infos = parse_generate_infos_by_loaded_block(
        loaded_block, default_config, parse_env.env_dict
    )

    return BlockConfig(
        unique_infos(
            itertools.chain(dir_infos,
                            flatten(result.dir_infos for result in file_info_results)
                            )
        ),
        list(flatten(map(attrgetter('move_infos'), file_info_results))),
        list(flatten(map(attrgetter('expand_infos'), file_info_results))),
        [result.file_info for result in file_info_results if result.file_info],
        generate_infos,
    )


def make_loaded_block_element(root_ele: ET.Element,
                              dst_path: str = '') -> LoadedBlockElement:
    """创建加载后的块配置。"""
    return LoadedBlockElement(root_ele, False, dst_path, set(), set(), {})


def parse_block_element(block_ele: ET.Element,
                        block_info_attr: Dict[str, str]) -> BlockElement:
    """解析单个块配置。"""

    def filter_attrs(attrs: Dict[str, str]) -> Dict[str, str]:
        # block属性中过滤掉dst_path与block_conf_path
        # dst_path由单独的参数传递
        # block中不需要block_conf_path
        return {
            key: value
            for key, value in attrs.items()
            if key not in ('dst_path', 'block_conf_path')
        }

    def with_merged_attrs(attrs: Dict[str, str]) -> BlockElement:
        name = attrs.get('name')
        block_conf_path = attrs.get('block_conf_path')

        if not name:
            raise BlockConfigError("block's name is not set!")

        if not block_conf_path:
            raise BlockConfigError("block's conf_path is not set!")

        return BlockElement(
            name=name,
            block_conf_path=block_conf_path,
            dst_path=attrs.get('dst_path', ''),
            chips=config_feature_to_set(attrs.get('chip'), 'chip'),
            features=config_feature_to_set(attrs.get('feature'), 'feature'),
            attrs=filter_attrs(attrs)
        )

    return with_merged_attrs(merge_dict(block_info_attr, block_ele.attrib))


def parse_block_info(block_info: ET.Element) -> List[BlockElement]:
    """解析块配置。"""
    def parse_block_elements(block_elements: List[ET.Element]) -> List[BlockElement]:
        return [
            parse_block_element(block_ele, block_info.attrib) for block_ele in block_elements
        ]

    return parse_block_elements(list(block_info))


def get_block_filepath(block_element: BlockElement) -> str:
    """获取块配置路径。"""
    return os.path.join(
        pkg_utils.TOP_SOURCE_DIR, BLOCK_CONFIG_PATH, block_element.block_conf_path,
        f'{block_element.name}.xml'
    )


def load_block_element(package_attr: PackageAttr,
                       block_element: BlockElement) -> LoadedBlockElement:
    """加载块配置。"""

    def with_filepath(block_xml: str):
        if not os.path.exists(block_xml):
            raise BlockConfigError(f"block's config xml {block_xml} does not exist!")

        try:
            return LoadedBlockElement(
                root_ele=ET.parse(block_xml).getroot(),
                use_move=package_attr.get('use_move', False),
                **{
                    name: getattr(block_element, name)
                    for name in BLOCK_ELEMENT_PASS_THROUGH_ARGS
                }
            )
        except Exception:
            raise BlockConfigError(f"dependent block configuration {block_xml} parse failed!")

    return with_filepath(get_block_filepath(block_element))


def parse_blocks(root_ele: ET.Element,
                 package_attr: PackageAttr,
                 parse_env: ParseEnv) -> List[BlockConfig]:
    """解析块列表。"""
    return [
        parse_block_config(
            loaded_block, package_attr, parse_env
        )
        for loaded_block in itertools.chain(
            [make_loaded_block_element(root_ele)],
            map(
                partial(load_block_element, package_attr),
                chain.from_iterable(
                    map(parse_block_info, root_ele.findall("block_info"))
                )
            )
        )
    ]


def read_version_info() -> Tuple[str, str]:
    version_path = os.path.join(pkg_utils.TOP_DIR, "version.info")
    with open(version_path, 'r') as file:
        line1 = file.readline().strip()
        line2 = file.readline().strip()
    version = line1.split("=")[1]
    version_dir = line2.split("=")[1]
    m = re.match(r'[.a-zA-Z0-9]+$', version) or re.match(r'[-a-zA-Z.0-9]+$', version)
    if not m:
        raise VersionFormatNotMatch()
    
    return version, version_dir


def parse_xml_config(filepath: str,
                     delivery_dir: str,
                     parse_option: ParseOption,
                     args: Namespace) -> XmlConfig:
    """解析打包xml配置。"""
    try:
        tree = ET.parse(filepath)
        xml_root = tree.getroot()
    except ET.ParseError as ex:
        CommLog.cilog_error("xml parse %s failed: %s!", filepath, ex)
        sys.exit(FAIL)

    default_config = xml_root.attrib.copy()

    package_attr = parse_package_attr(xml_root, args)
    if args.version_dir:
        version = args.version_dir
        version_dir = args.version_dir
    else:
        version, version_dir = read_version_info()
    if args.disable_multi_version:
        version_dir = None
    timestamp = get_timestamp(args)
    try:
        env_dict = parse_env_dict(
            parse_option.os_arch, package_attr, version, version_dir, timestamp
        )
    except ParseOsArchError:
        CommLog.cilog_error(
            "os_arch %s is not correctly configured: %s!",
            parse_option.os_arch, filepath
        )
        sys.exit(FAIL)

    parse_env = ParseEnv(
        env_dict, parse_option, delivery_dir, pkg_utils.TOP_SOURCE_DIR
    )

    blocks = parse_blocks(
        xml_root, package_attr, parse_env
    )

    if is_multi_version(version_dir):
        fill_is_common_path_func = partial(
            fill_is_common_path, target_env=env_dict.get('TARGET_ENV')
        )
    else:
        fill_is_common_path_func = iter

    return XmlConfig(
        default_config, package_attr, None, blocks, version, None,
        PackerConfig(fill_is_common_path_func)
    )
