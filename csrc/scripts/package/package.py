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

import os
import sys
import re
import argparse
import traceback
import csv
from argparse import Namespace
from collections import namedtuple
from datetime import datetime, timezone
from functools import partial
from itertools import chain
from typing import Dict, Iterator, List, Set, Tuple, TextIO
import shutil

from common.py.utils import pkg_utils
from common.py.filelist import (
    FileItem, FileList, check_filelist, create_file_item, generate_filelist,
    get_transform_nested_path_func,
)
from common.py.packer import (
    PackageName, create_makeself_pkg_params_factory, create_run_package_command, exec_pack_cmd
)
from common.py.pkg_parser import (
    ParseOption, XmlConfig, parse_xml_config, get_cann_version_info
)
from common.py.utils.pkg_utils import (
    CONFIG_SCRIPT_PATH, CompressError, ContainAsteriskError, DELIVERY_PATH, FAIL,
    FilelistError, GenerateFilelistError, PackageNameEmptyError, SUCCESS, TOP_DIR,
    UnknownOperateTypeError, path_join
)
from common.py.utils.funcbase import invoke, pipe
from common.py.utils.comm_log import CommLog


def get_comments(package_name: PackageName) -> str:
    """获取run包注释。"""
    comments = '_'.join(
        [package_name.chip_name.upper(), package_name.func_name.upper(), 'RUN_PACKAGE']
    )
    return f'"{comments}"'


def get_compress_cmd(delivery_dir: str,
                     pkg_args: Namespace,
                     xml_config: XmlConfig) -> str:
    """获取makeself压缩命令"""
    suffix = xml_config.package_attr.get('suffix')
    if suffix == "run":
        package_name = PackageName(xml_config.package_attr, pkg_args, xml_config.version)
        factory = create_makeself_pkg_params_factory(
            pkg_args.pkg_output_dir, package_name.getvalue(), get_comments(package_name)
        )
        params = factory(pkg_args.makeself_dir, xml_config.package_attr, pkg_args.independent_pkg)
        pack_cmd, err_msg = create_run_package_command(params)
        if err_msg:
            CommLog.cilog_error(err_msg)
            CommLog.cilog_error("create_run_command failed!")
            raise CompressError(package_name.getvalue())
        if pkg_args.independent_pkg:
            exec_pack_cmd(delivery_dir, pack_cmd, package_name.getvalue())
    else:
        CommLog.cilog_error("the repack type '%s' is not support!", suffix)
        sys.exit(FAIL)
    try:
        makeself_dir = os.path.join(TOP_DIR, "build/makeself.txt")
        with open(makeself_dir, 'w') as f:
            f.write(pack_cmd)
    except Exception as exception:
        CommLog.cilog_error(f"save makeself.txt failed!{str(exception)}")
        sys.exit(FAIL)
    return package_name.getvalue()


def make_parse_option(args_: argparse.Namespace) -> ParseOption:
    """创建解析参数。"""

    return ParseOption(
        args_.os_arch, args_.pkg_version,
        args_.build_type,
        args_.package_check,
        args_.ext_name
    )


PrivatePackageOption = namedtuple(
    'PrivatePackageOption',
    [
        'os_arch', 'package_suffix', 'not_in_name', 'pkg_version', 'ext_name',
        'chip_name', 'func_name', 'version_dir', 'disable_multi_version', 'suffix'
    ]
)


class PackageOption(PrivatePackageOption):
    """打包配置参数。"""
    __slots__ = ()  # 优化内存，避免创建 __dict__

    def __new__(cls, *package_option_args, **kwargs):
        return super().__new__(cls, *package_option_args, **kwargs)


def generate_info_content(target_conf, ext_name) -> List[str]:
    """生成info内容。"""

    def toolchain_llvm_config() -> Iterator[Tuple[str, str]]:
        if 'llvm' in ext_name:
            yield 'toolchain', 'llvm'

    content_list = [
        f'{key}={value}'
        for key, value in chain(
            target_conf['content'].items(), toolchain_llvm_config()
        )
    ]
    return content_list


def generate_version_header_content(target_conf) -> Iterator[str]:
    """生成version_header内容。"""
    guard_name = target_conf['value'].replace('.', '_').upper()
    yield f'#ifndef {guard_name}'
    yield f'#define {guard_name}'
    yield ''
    for name, value in target_conf['content'].items():
        if name.endswith('_VERSION'):
            version_infos = get_cann_version_info(name, value)
            for version_name, version_value in version_infos:
                yield f'#define {version_name} {version_value}'
        else:
            yield f'#define {name} {value}'
    yield ''
    yield f'#endif /* {guard_name} */'
    yield ''


def generate_customized_file(target_conf, ext_name):
    filepath = os.path.join(TOP_DIR, "build", target_conf.get('value'))

    generator = target_conf.get('generator', 'info')
    if generator == 'version_header':
        content_list = generate_version_header_content(target_conf)
    else:
        content_list = generate_info_content(target_conf, ext_name)

    file_content = '\n'.join(content_list)
    try:
        with open(filepath, 'w') as file:
            file.write(file_content)
    except Exception as ex:
        CommLog.cilog_error(f"generate customized file {filepath} failed: {ex}!")
        return FAIL

    return SUCCESS


def get_module(target_config) -> str:
    """获取配置模块。"""
    module = target_config.get('module', 'NA')
    return module if module else 'NA'


def get_operation(operation, target_config) -> str:
    """获取操作类型。"""
    if operation in ('copy', 'move') and target_config.get('entity') == 'true':
        return 'copy_entity'
    return operation


def get_permission(target_config) -> str:
    """获取配置权限。"""
    return target_config.get('install_mod', 'NA')


def get_owner_group(target_config) -> str:
    """获取配置属主。"""
    # install_own的可能值为$username:$usergroup
    # 防止变量在install_common_parser.sh中，被eval展开，添加\转义$
    # 由于awk会消耗1个\，所以需要2个转义符
    return target_config.get('install_own', 'NA').replace('$', '\\\\$')


def get_install_type(target_config) -> str:
    """获取安装类型。"""
    return target_config.get('install_type', 'NA')


def get_softlink(target_config) -> List[str]:
    """获取配置软链。"""
    softlink_str = target_config.get('install_softlink')
    if not softlink_str:
        return []
    return softlink_str.split(';')


def get_feature(target_config) -> Set[str]:
    """获取配置特性。"""
    return target_config['feature']


def get_chip(target_config) -> Set[str]:
    """获取配置芯片。"""
    return target_config['chip']


def get_configurable(target_config) -> str:
    """获取配置是否为配置文件。"""
    return target_config.get('configurable', 'FALSE')


def get_hash_value(target_config) -> str:
    """获取配置哈希值。"""
    return target_config.get('hash', 'NA')


def get_block(target_config) -> str:
    """获取配置块信息。"""
    return target_config.get('name', 'NA')


def get_pkg_inner_softlink(target_config) -> List[str]:
    """获取配置包内软链。"""
    softlink_str = target_config.get('pkg_inner_softlink')
    if not softlink_str:
        return []
    return softlink_str.split(';')


def parse_install_info(infos: List,
                       operate_type,
                       filter_key) -> Iterator[FileItem]:
    """根据配置解析生成安装信息。"""
    for target_config in infos:
        target_name = get_target_name(target_config)
        if target_config.get("optional") == 'true' and operate_type in ('copy', 'move'):
            path = os.path.join(TOP_DIR, DELIVERY_PATH, target_config.get('dst_path'))
            vaule = os.path.join(TOP_DIR, DELIVERY_PATH, target_config.get('dst_path'), target_name)
            if not os.path.exists(path):
                continue
            if not os.path.exists(vaule):
                continue
        if operate_type in ('copy', 'move'):
            relative_path_in_pkg = os.path.join(target_config.get('dst_path'), target_name)
            relative_install_path = path_join(target_config.get('install_path'), target_name)
            is_dir = target_config.get('is_dir', False)
        elif operate_type == 'mkdir':
            relative_path_in_pkg = 'NA'
            relative_install_path = target_config.get('value')
            is_dir = False
        elif operate_type == 'del':
            relative_path_in_pkg = 'NA'
            relative_install_path = path_join(target_config.get('install_path'), target_name)
            is_dir = False
        else:
            raise UnknownOperateTypeError(f"unknown operate type {operate_type}")

        if relative_install_path is None:
            continue

        install_type = get_install_type(target_config)
        if any(key in install_type for key in filter_key):
            is_in_docker = 'TRUE'
        else:
            is_in_docker = 'FALSE'

        file_item = create_file_item(
            get_module(target_config),
            get_operation(operate_type, target_config),
            relative_path_in_pkg,
            relative_install_path,
            is_in_docker,
            get_permission(target_config),
            get_owner_group(target_config),
            install_type,
            get_softlink(target_config),
            get_feature(target_config),
            'N',
            get_configurable(target_config),
            get_hash_value(target_config),
            get_block(target_config),
            get_pkg_inner_softlink(target_config),
            get_chip(target_config),
            is_dir,
        )

        yield file_item


def execute_repack_process(xml_config: XmlConfig,
                           delivery_dir: str,
                           pkg_args: Namespace,
                           package_name: PackageName = None,
                           package_option: PackageOption = None):
    """
    功能描述: 执行打包流程(拷贝--->签名--->打包)
    返回值: SUCCESS/FAIL
    """
    release_dir = os.path.join(
        delivery_dir, xml_config.default_config.get('name', 'default'))
    # 生成自定义文件
    for item in xml_config.generate_infos:
        if generate_customized_file(item, package_option.ext_name):
            return FAIL

    # 校验包中文件或目录大小
    if pkg_args.check_size == "True":
        limit_list, tag = processing_csv_file(
            release_dir, package_name.func_name, package_name.chip_name, pkg_args.build_type
        )
        if not tag:
            return FAIL
        if limit_list:
            abspath = os.path.abspath(release_dir)
            replace_path = abspath + "/"
            result = check_add_dir(replace_path, abspath, limit_list)
            if not result:
                return FAIL
    try:
        package_name = get_compress_cmd(pkg_args.pkg_output_dir, pkg_args, xml_config)
    except CompressError:
        return FAIL

    CommLog.cilog_info("package %s generate filelist.csv and makeself cmd successfully!",
                       package_name)
    return SUCCESS


def check_path_is_conflict(xml_config):
    """
    功能描述: 检查打包时安装路径与软连接路径是否冲突
    参数: xml_config
    返回值: SUCCESS/FAIL
    """
    install_path_list = set()
    pkg_softlink_list = set()
    for item in xml_config.package_content_list:
        value_list = item.get('value').split('/')
        target_name = value_list[-1] if value_list[-1] else value_list[-2]
        if item.get('install_path'):
            install_path_list.add(
                os.path.join(item['install_path'], target_name)
            )
        if item.get('pkg_inner_softlink'):
            pkg_softlink = item.get('pkg_inner_softlink')
            pkg_softlink_list.add(pkg_softlink)
    if install_path_list & pkg_softlink_list:
        CommLog.cilog_info('intersection:{}'.format(install_path_list & pkg_softlink_list))
        CommLog.cilog_info('path conflicting: pkg_inner_softlink dir equals install_path!!')
        return FAIL
    return SUCCESS


def checksum_value(limit_value, release_dir):
    """
    功能描叙: 校验传入的文件或目录大小是否合格
    参数:
    limit_value: limit.csv中的一行数据如[compiler/bin, 3976, 110%]
    返回值: True/False
    """
    path = os.path.join(release_dir, limit_value[1])
    if len(limit_value) >= 7:
        try:
            max_value = int(limit_value[4])
        except ValueError:
            CommLog.cilog_error("{0} configuration is not standard., Please check limit.csv.".format(path))
            return True
    else:
        CommLog.cilog_error("{0} configuration is less than four, Please check limit.csv.".format(path))
        return True
    if not os.path.exists(path):
        CommLog.cilog_warning("{0} doesn't exist, Please check limit.csv.".format(path))
        return True
    size = 0
    for root, dirs, files in os.walk(path):
        size += os.path.getsize(root)
        for f in files:
            filepath = os.path.join(root, f)
            if os.path.islink(filepath):
                continue
            if not os.path.exists(filepath):
                continue
            size += os.path.getsize(os.path.join(root, f))
    if size == 0:
        size = os.path.getsize(path)
    if size > max_value * 1024:
        CommLog.cilog_error(f"\n{path} size {size} bytes exceeds maximum {max_value * 1024} bytes")
        return False
    return True


def processing_csv_file(release_dir, package_name, chip_name, build_type):
    """
    功能描叙: 处理limit.csv文件数据
    返回值: [],True/[],False
    """
    ret = True
    limit_list = []
    product = os.path.basename(os.path.dirname(release_dir))
    limit_path = os.path.join(pkg_utils.TOP_SOURCE_DIR, CONFIG_SCRIPT_PATH, "common/limit.csv")
    if not os.path.exists(limit_path):
        CommLog.cilog_warning("{0} doesn't exist.".format(limit_path))
        return limit_list, ret
    with open(limit_path, "r") as file:
        reader = csv.reader(file)
        next(reader)
        for data in reader:
            if not data:
                CommLog.cilog_warning("The limit.csv file contains empty lines.")
                continue
            if is_match_line(package_name, chip_name, product, build_type, data):
                if data[1][-1] == "/":
                    limit_list.append(data[1][:-1])
                else:
                    limit_list.append(data[1])
                res = checksum_value(data, release_dir)
                if not res:
                    ret = False
    return limit_list, ret


def is_match_line(package_name, chip_name, product, build_type, data):
    return package_name == data[0] and chip_name == data[5] and product == data[6] and build_type == data[7].lower()


def check_add_dir(package_path, dirs, limit_list, ret=True):
    """
    功能描述: 校验新增目录
    参数: path, limit_list
    返回值: False/True
    """
    for limit_path in limit_list:
        if dirs == os.path.join(os.path.split(dirs)[0], limit_path):
            return ret
    for dir_file in os.listdir(dirs):
        path = os.path.join(dirs, dir_file)
        relative_path = path.replace(package_path, "")
        if os.path.isfile(path) and relative_path not in limit_list:
            CommLog.cilog_error("{0} is not in limit.csv file and is newly added.".format(path))
            ret = False
        elif os.path.isdir(path) and relative_path not in limit_list:
            ret = check_add_dir(package_path, path, limit_list, ret)
    return ret


def get_target_name(target_conf) -> str:
    """获取目标名。"""
    rename = target_conf.get('rename')
    if rename:
        return rename

    value_list = target_conf.get('value').split('/')
    target_name = value_list[-1] if value_list[-1] else value_list[-2]
    return target_name


def gen_file_install_list(xml_config: XmlConfig,
                          filter_key) -> Tuple[FileList, FileList]:
    """生成filelist列表。"""
    file_install_list = []

    dir_filelist = parse_install_info(
        xml_config.dir_install_list, 'mkdir', filter_key
    )
    move_filelist = parse_install_info(
        xml_config.move_content_list, 'move', filter_key
    )
    pkg_filelist = parse_install_info(
        xml_config.package_content_list, 'copy', filter_key
    )
    gen_filelist = parse_install_info(
        xml_config.generate_infos, 'copy', filter_key
    )
    # file_info中配置为文件夹，这里是被展开的文件,则需要单独删除
    del_filelist = parse_install_info(
        xml_config.expand_content_list, 'del', filter_key
    )
    collect_filelist = list(chain(dir_filelist, move_filelist, pkg_filelist, gen_filelist))
    collect_filelist = list(xml_config.packer_config.fill_is_common_path(collect_filelist))
    all_filelist = list(chain(collect_filelist, del_filelist))
    for file_item in all_filelist:
        file_install_list.append(file_item)

    return file_install_list, []


def generate_filelist_file_by_xml_config(xml_config: XmlConfig,
                                         filter_key: List[str],
                                         package_check: bool):
    """生成文件列表文件。"""
    check_move = xml_config.package_attr.get('use_move', False)
    transform_nested_path_func = get_transform_nested_path_func(
        xml_config.package_attr.get('parallel') or check_move
    )
    check_features = xml_config.package_attr.get('check_features', False)

    file_install_list, [] = invoke(
        pipe(
            gen_file_install_list,
            partial(map, transform_nested_path_func),
            tuple,
        ),
        xml_config, filter_key
    )
    generate_filelist(file_install_list, 'filelist.csv')
    # 先生成再检查，有利于问题定位
    check_filelist(file_install_list, check_features, check_move)


def get_pkg_xml_relative_path(pkg_args: Namespace) -> str:
    """获取包配置文件相对路径。"""

    def parts():
        yield CONFIG_SCRIPT_PATH
        yield pkg_args.pkg_name
        if pkg_args.chip_scenes:
            yield pkg_args.chip_scenes
        # 可以通過build_rule指定xml_file，而且优先级高于默认值
        if pkg_args.xml_file:
            yield pkg_args.xml_file
        else:
            yield f'{pkg_args.pkg_name}.xml'

    return os.path.join(*parts())


def write_config_inc_var(name: str, package_attr: Dict, file: TextIO):
    """向config.inc文件写入变量。"""
    if name in package_attr:
        value = str(package_attr[name]).lower()
        file.write(f"{name.upper()}={value}\n")


def generate_config_inc(package_attr: Dict):
    """生成config.inc文件。"""
    if 'parallel' not in package_attr and 'parallel_limit' not in package_attr and 'use_move' not in package_attr:
        return
    year = datetime.now(timezone.utc).year
    config_inc = os.path.join(TOP_DIR, "build", 'config.inc')
    header = [
        '#!/bin/sh\n',
        '#----------------------------------------------------------------------------\n',
        f'# Copyright Huawei Technologies Co., Ltd. 2023-{year}. All rights reserved.\n',
        '#----------------------------------------------------------------------------\n',
        '\n',
    ]
    if os.path.isfile(config_inc):
        os.chmod(config_inc, 0o700)
    with open(config_inc, 'w', encoding='utf-8') as file:
        file.writelines(header)
        write_config_inc_var('parallel', package_attr, file)
        write_config_inc_var('parallel_limit', package_attr, file)
        write_config_inc_var('use_move', package_attr, file)

    os.chmod(config_inc, 0o500)


def update_version_info(new_version: str):
    version_path = os.path.join(pkg_utils.TOP_DIR, "version.info")
    with open(version_path, 'r') as file:
        content = file.read()
        content = re.sub(r'Version=.*', f'Version={new_version}', content)
        content = re.sub(r'vension_dir=.*', f'version_dir={new_version}', content)
    with open(version_path, 'w') as file:
        file.write(content)


def main(pkg_name='', xml_file='', main_args=None):
    """
    功能描述: 执行打包流程(解析配置--->生成文件列表--->执行拷贝/打包动作)
    参数: pkg_name, os_arch, type
    返回值: SUCCESS/FAIL
    """
    delivery_dir = os.path.join(TOP_DIR, DELIVERY_PATH)
    if not os.path.exists(delivery_dir):
        return FAIL

    config_relative_path = get_pkg_xml_relative_path(main_args)
    pkg_xml_file = os.path.join(pkg_utils.TOP_SOURCE_DIR, config_relative_path)
    parse_option = make_parse_option(main_args)
    if main_args.version_dir:
        update_version_info(main_args.version_dir)

    try:
        xml_config = parse_xml_config(
            pkg_xml_file, delivery_dir, parse_option, main_args
        )
    except ContainAsteriskError as ex:
        CommLog.cilog_error(f"Value contain '*' in {config_relative_path}. value is '{ex.value}'.")
        return FAIL

    if pkg_name in ['driver', 'firmware']:
        filter_key = ['all', 'docker']
    elif pkg_name in ['aicpu_kernels_device', 'aicpu_kernels_host']:
        filter_key = []
    else:
        filter_key = ['all', 'run']

    # 生成filelist.csv安装列表文件
    try:
        generate_filelist_file_by_xml_config(
            xml_config, filter_key,
            main_args.package_check or xml_config.package_attr.get('package_check')
        )
    except PackageNameEmptyError:
        CommLog.cilog_error(f'package name is empty in {xml_file}, please check it')
        return FAIL
    except GenerateFilelistError as ex:
        CommLog.cilog_error(f'generate filelist {ex.filename} failed!', )
        return FAIL
    except FilelistError as ex:
        CommLog.cilog_error('check filelist error! %s', str(ex))
        return FAIL

    generate_config_inc(xml_config.package_attr)

    if main_args.independent_pkg:
        src_file_path = os.path.join(TOP_DIR, "build", "filelist.csv")
        dst_file_path = os.path.join(main_args.pkg_output_dir, "share", "info", main_args.pkg_name, "script")
        shutil.copy(src_file_path, dst_file_path)

    package_option = PackageOption(
        main_args.os_arch, main_args.package_suffix, main_args.not_in_name, main_args.pkg_version, main_args.ext_name,
        chip_name=main_args.chip_name, func_name=main_args.func_name, version_dir=main_args.version_dir,
        disable_multi_version=main_args.disable_multi_version, suffix=main_args.suffix)

    package_name = PackageName(xml_config.package_attr, main_args, xml_config.version)

    # 检查install_path与pkg_inner_softlink路径是否冲突，若冲突则报错
    if check_path_is_conflict(xml_config) == FAIL:
        return FAIL

    # 生成打包命令
    return execute_repack_process(xml_config, delivery_dir, main_args,
                                  package_name=package_name, package_option=package_option)


def args_parse():
    """
    功能描述 : 脚本入参解析
    参数 : 调用脚本的传参
    返回值 : 解析后的参数值
    """
    parser = argparse.ArgumentParser(
        description='This script is for package repack processing.')
    parser.add_argument('-c', '--chip_scenes', metavar='chip_scenes', required=False, dest='chip_scenes', nargs='?',
                        const='',
                        default='', help='This parameter define chip id for package.')
    parser.add_argument('-n', '--pkg_name', metavar='pkg_name', required=False,
                        help='This parameter define pkg_name for config_xml.')
    parser.add_argument('-o', '--os_arch', metavar='os_arch', required=False, dest='os_arch', nargs='?', const='',
                        default=None, help="This parameter define the package's os_arch")
    parser.add_argument('-t', '--type', metavar='type', required=False, dest='type', nargs='?', const='',
                        default='repack', help="This parameter define this script's function")
    parser.add_argument('-i', '--not_in_name', metavar='not_in_name', required=False, dest='not_in_name', nargs='?',
                        const='',
                        default='', help="This parameter define the package's name not contain the element")
    parser.add_argument('-v', '--pkg_version', metavar='pkg_version', required=False, dest='pkg_version', nargs='?',
                        const='',
                        default='', help="This parameter define the version for package.")
    parser.add_argument('-e', '--ext_name', metavar='ext_name', required=False, dest='ext_name', nargs='?', const='',
                        default='', help="This parameter define the package's ext_name")
    parser.add_argument('--package_suffix', nargs='?', const='none',
                        default='none', help="This parameter define the package suffix, debug or none")
    parser.add_argument('--suffix', metavar='suffix', required=False, dest='suffix', nargs='?', const='',
                        default=None, help="This parameter define the package suffix, for example such as tar.gz")
    parser.add_argument('-b', '--build_type', metavar='build_type', required=False, dest='build_type', nargs='?',
                        const='',
                        default='debug', help="This parameter define release type of package")
    parser.add_argument('-x', '--xml', metavar='xml_file', required=False, dest='xml_file', nargs='?', const='',
                        default='', help="This parameter define xml file")
    parser.add_argument('--chip_name', metavar='chip_name', required=False, dest='chip_name', nargs='?', const=None,
                        default=None,
                        help="This parameter define package chip name, has higher priority than chip name in xml")
    parser.add_argument('--func_name', metavar='func_name', required=False, dest='func_name', nargs='?', const=None,
                        default=None,
                        help="This parameter define package func name, has higher priority than func name in xml")
    parser.add_argument('--source_root', metavar='source_root', required=False, dest='source_root', nargs='?', const='',
                        help='source root dir.')
    parser.add_argument('--makeself_dir', metavar='makeself_dir', required=False, dest='makeself_dir', 
                        nargs='?', const='', help='makeself dir.')
    parser.add_argument('--independent_pkg', action='store_true', help='Independent pkg.')                    
    parser.add_argument('--pkg-output-dir', default='', help='Package dirpath.')
    parser.add_argument('--version_dir', nargs='?', const='', default='', help='Set version dir.')
    parser.add_argument('--tag', metavar='tag', nargs='?', const='', default='')
    parser.add_argument('--disable-multi-version', action='store_true', help='Disable multi version.')
    # 检查打包配置
    parser.add_argument('--package-check', action='store_true', help='check package config.')
    parser.add_argument('--check_size', nargs='?', const='', default='', help="Check the size of a file or directory.")
    parser.add_argument('--pkg-name-style', metavar='pkg_name_style', default='common', help='Package name style.')
    return parser.parse_args()


if __name__ == "__main__":
    CommLog.cilog_info("%s", " ".join(sys.argv))
    args = args_parse()
    try:
        if args.source_root:
            pkg_utils.TOP_SOURCE_DIR = args.source_root
        if args.build_type == '':
            args.build_type = 'debug'
        else:
            args.build_type = args.build_type.lower()
        status = main(args.pkg_name, args.xml_file, main_args=args)
    except Exception as e:
        CommLog.cilog_error("exception is occurred (%s)!", e)
        CommLog.cilog_info("%s", traceback.format_exc())
        status = FAIL
    sys.exit(status)
open