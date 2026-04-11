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
import shutil
import subprocess
from argparse import Namespace
from itertools import chain
from subprocess import PIPE, STDOUT
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union

from .utils.comm_log import CommLog
from .utils.pkg_utils import CompressError


class PackageName:
    """包名。"""

    def __init__(self,
                 package_attr,
                 args: Namespace,
                 version: str):
        self.product_name = package_attr.get('product_name')
        self.chip_name = args.chip_name or package_attr.get('chip_name')
        self.suffix = args.suffix or package_attr.get('suffix')
        self.func_name = get_func_name(args.func_name, package_attr)
        self.chip_plat = package_attr.get('chip_plat')
        self.deploy_type = package_attr.get('deploy_type')
        self.version = version.lower()
        self.not_in_name_list = args.not_in_name.split(",")
        self.os_arch = args.os_arch
        self.package_suffix = args.package_suffix
        self.ext_name = args.ext_name
        if args.pkg_name_style == 'underline':
            self.name_sep = '_'
        else:
            self.name_sep = '-'

    def get_attribute(self, name: str) -> Optional[str]:
        """获取属性。"""
        if name in self.not_in_name_list:
            return None
        return getattr(self, name)

    def getvalue(self) -> str:
        product_name = self.get_attribute('product_name')
        chip_name = self.get_attribute('chip_name')
        func_name = self.get_attribute('func_name')
        version = self.get_attribute('version')
        os_arch = self.get_attribute('os_arch')
        chip_plat = self.get_attribute('chip_plat')
        deploy_type = self.get_attribute('deploy_type')
        ext_name = self.get_attribute('ext_name')
        package_suffix = "debug" if self.package_suffix == "debug" else None

        region1 = "-".join(filter(None, [product_name, remove_ascend(chip_name), func_name]))
        region2 = ".".join(filter(None, [version]))
        region3 = "-".join(filter(None, [os_arch, chip_plat, deploy_type, package_suffix, ext_name]))
        package_name = "_".join(filter(None, [region1, region2, region3]))

        return f"{package_name}.{self.suffix}"


class MakeselfPkgParams(NamedTuple):
    """run包打包参数。"""
    package_name: str
    comments: str
    makeself_tool: Optional[str] = None
    makeself_header: Optional[str] = None
    help_info: Optional[str] = None
    source_target: Optional[str] = None
    
    install_script: Optional[str] = None
    independent_pkg: Optional[bool] = False
    cleanup: Optional[str] = None


def remove_ascend(text):
    if text is None:
        return None
    text_lower = text.lower()
    if text_lower == "ascend910_93":
        return "A3"
    if "ascend" in text_lower:
        return text_lower.replace("ascend", "")
    return text_lower


def get_func_name(func_name: str, package_attr) -> str:
    """获取包func_name。"""
    return func_name or package_attr.get('func_name')


def get_compress_tool() -> str:
    tools = ["pigz", "gzip", "bzip2", "xz"]
    for tool in tools:
        path = shutil.which(tool)
        if path:
            return "--" + tool
    CommLog.cilog_error("The system does not come with a compression tool pre-installed."
                        "Please ensure at least one of the folllowing compression tools is available: %s", tools)
    return ""


def get_compress_format() -> str:
    tar_format = "gnu"
    path = shutil.which("bsdtar")
    if path:
        tar_format = "ustar"
    return tar_format


def compose_makeself_command(params: MakeselfPkgParams) -> str:
    """组装makeself包打包命令。"""

    def get_cleanup_commands() -> List[str]:
        if params.cleanup:
            return ['--cleanup', params.cleanup]
        return []
    independent_pkg = params.independent_pkg
    compress_tool = get_compress_tool()
    tar_format = get_compress_format()
    if independent_pkg:
        commands = chain(
        [
            'TMPDIR=$pwd', params.makeself_tool, "--header", params.makeself_header,
            "--help-header", params.help_info, compress_tool, '--complevel', '4',
            '--nomd5', '--sha256', '--nooverwrite', '--chown', '--tar-format', tar_format,
            '--tar-extra', '--numeric-owner', '--tar-quietly'
        ],
        get_cleanup_commands(),
        [params.source_target, params.package_name, params.comments, params.install_script]
        )
    else:
        commands = chain(
        [
            compress_tool, '--complevel', '4',
            '--nomd5', '--sha256', '--nooverwrite', '--chown', '--tar-format', tar_format,
            '--tar-extra', '--numeric-owner', '--tar-quietly'
        ],
        get_cleanup_commands(),
        [params.package_name, params.comments]
       )
    
    command = ' '.join(commands)
    return command


def create_makeself_pkg_params_factory(source_target: str,
                                       package_name: str,
                                       comments: str
                                       ) -> Callable[[str, dict, bool], MakeselfPkgParams]:
    """创建Makeself打包参数工厂。"""

    def create_makeself_pkg_params(makeself_dir: str,
                                  package_attr: Dict,
                                  independent_pkg=False) -> MakeselfPkgParams:
        """创建Makeself打包参数。"""
        cleanup = package_attr.get('cleanup')

        if independent_pkg:
            install_script = str(package_attr.get('install_script'))
            help_info = str(package_attr.get('help'))
            makeself_tool = os.path.join(makeself_dir, 'makeself.sh')
            makeself_header = os.path.join(makeself_dir, 'makeself-header.sh')
            params = MakeselfPkgParams(
            package_name=package_name,
            comments=comments,
            makeself_tool=makeself_tool,
            makeself_header=makeself_header,
            help_info=help_info,
            source_target=source_target,
            
            install_script=install_script,
            independent_pkg=independent_pkg,
            cleanup=cleanup,
        )
        else:
            params = MakeselfPkgParams(
            package_name=package_name,
            comments=comments,
            cleanup=cleanup,
        )
        return params
    return create_makeself_pkg_params


def create_run_package_command(params: MakeselfPkgParams
                               ) -> Tuple[Optional[str], Optional[str]]:
    """
    功能描述: 组装打run包命令
    返回值: command
    """
    return compose_makeself_command(params), None


def exec_pack_cmd(delivery_dir: str,
                 pack_cmd: str,
                 package_name: str) -> str: 
    """执行打包命令"""
    if delivery_dir:
        cmd = f'cd {delivery_dir} && {pack_cmd}'
    else:
        cmd = pack_cmd
    CommLog.cilog_info("package cmd:%s", cmd)
    result = subprocess.run(cmd, shell=True, check=False, stdout=PIPE, stderr=STDOUT)
    output = result.stdout.decode()
    if result.returncode != 0:
        CommLog.cilog_error(__file__, "compress package(%s) failed! %s.", package_name, output)
        raise CompressError(package_name)
    return package_name