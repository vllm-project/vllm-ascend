#!/bin/sh
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

## module log

if [ "$(id -u)" != "0" ]; then
    COMM_LOG_DIR="${HOME}/var/log/ascend_seclog"
else
    COMM_LOG_DIR="/var/log/ascend_seclog"
fi

COMM_OPERATION_LOGFILE="${COMM_LOG_DIR}/operation.log"
COMM_LOGFILE="${COMM_LOG_DIR}/ascend_install.log"
COMM_USERNAME="$(id -un)"
COMM_USERGROUP="$(id -gn)"

LOG_PKG_NAME="Common"
LOG_FILE="/dev/null"
LOG_STYLE="normal"

# 初始化日志系统
comm_init_log() {
    if [ ! -d "$COMM_LOG_DIR" ]; then
        mkdir -p "$COMM_LOG_DIR"
    fi
    if [ $(id -u) -ne 0 ]; then
        chmod 740 "$COMM_LOG_DIR"
    else
        chmod 750 "$COMM_LOG_DIR"
    fi
    if [ ! -f "$COMM_LOGFILE" ]; then
        touch "$COMM_LOGFILE"
    fi
    chmod 640 "$COMM_LOGFILE"
}

# 组合格式化后的日志信息
# _outvar : [输出变量] 格式化后的日志信息
# _log_type : 日志级别
# _msg : 日志信息
_comm_compose_log_msg() {
    local _outvar="$1"
    local _log_type="$2"
    local _msg="$3"
    local _cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
    local _result

    if [ "${LOG_STYLE}" = "no-colon" ]; then
        _result="[${LOG_PKG_NAME}] [${_cur_date}] [${_log_type}]${_msg}"
    else
        _result="[${LOG_PKG_NAME}] [${_cur_date}] [${_log_type}]: ${_msg}"
    fi
    eval "${_outvar}=\"${_result}\""
}

# 输出格式化后的消息
_comm_echo_log_msg() {
    local log_type_="$1"
    local log_format_="$2"

    if [ $log_type_ = "INFO" ]; then
        echo "${log_format_}"
    elif [ $log_type_ = "WARNING" ]; then
        echo "${log_format_}"
    elif [ $log_type_ = "ERROR" ]; then
        echo "${log_format_}"
    elif [ $log_type_ = "DEBUG" ]; then
        :
    fi
}

# 输出
comm_echo() {
    local log_type="$1"
    local msg="$2"
    local log_msg

    _comm_compose_log_msg "log_msg" "$log_type" "$msg"
    _comm_echo_log_msg "$log_type" "$log_msg"
}

# 写日志
comm_log() {
    local log_type="$1"
    local msg="$2"
    local log_msg

    _comm_compose_log_msg "log_msg" "$log_type" "$msg"
    _comm_echo_log_msg "$log_type" "$log_msg"
    echo "$log_msg" >> "${LOG_FILE}"
}

# 设置日志参数
set_comm_log() {
    local pkg_name="$1"
    local log_file="$2"

    LOG_PKG_NAME="${pkg_name}"
    if [ "$log_file" != "" ]; then
        LOG_FILE="${log_file}"
    fi
}

# 安全日志
comm_log_operation() {
    local cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
    local operation="$1"
    local runfilename="$2"
    local result="$3"
    local installmode="$4"
    local all_parma="$5"
    local level=""
    if [ "${operation}"  = "Install" ]; then
        level="SUGGESTION"
    elif [ "${operation}" = "Upgrade" ]; then
        level="MINOR"
    elif  [ "${operation}" = "Uninstall" ]; then
        level="MAJOR"
    else
        level="UNKNOWN"
    fi

    if [ ! -f "${COMM_OPERATION_LOGFILE}" ]; then
        touch "${COMM_OPERATION_LOGFILE}"
        chmod 640 "${COMM_OPERATION_LOGFILE}"
    fi

    echo "${operation} ${level} ${COMM_USERNAME} ${cur_date} 127.0.0.1 ${runfilename} ${result} installmode=${installmode}; cmdlist=${all_parma}" >> "${COMM_OPERATION_LOGFILE}"
}

## end module

# 转换--install-for-all参数下文件权限
comm_set_install_for_all_mod() {
    local _outvar="$1"
    local _mod="$2"
    local _new_mod

    # ${parameter%word} Remove matching suffix pattern.
    local _new_mod="${_mod%?}"
    # ${parameter#word} Remove matching prefix pattern.
    local _new_mod="${_new_mod}${_new_mod#${_new_mod%?}}"

    eval "${_outvar}=\"${_new_mod}\""
}

# 创建文件
comm_create_file() {
    local path="$1"
    local mod="$2"
    local own="$3"
    local install_for_all="$4"

    if [ -d "$path" ]; then
        comm_log "WARNING" "remove existed dir $path before create file."
        rm -rf "$path"
    fi
    touch "$path"
    if [ $? -ne 0 ]; then
        comm_log "WARNING" "create file $path failed."
        return 1
    fi

    if [ "$install_for_all" = "true" ] || [ "$install_for_all" = "y" ]; then
        comm_set_install_for_all_mod "mod" "$mod"
    fi
    chmod "$mod" "$path"
    if [ $? -ne 0 ]; then
        comm_log "WARNING" "chmod path $path $mod failed."
        return 1
    fi

    chown "$own" "$path"
    if [ $? -ne 0 ]; then
        comm_log "WARNING" "chown path $path $own failed."
        return 1
    fi
    return 0
}

# 创建目录
comm_create_dir() {
    local path="$1"
    local mod="$2"
    local own="$3"
    local install_for_all="$4"

    if [ "$path" = "" ]; then
        comm_log "WARNING" "dir path is empty"
        return 1
    fi

    if [ ! -d "$path" ]; then
        mkdir -p "$path"
        if [ $? -ne 0 ]; then
            comm_log "WARNING" "create dir $path failed."
            return 1
        fi
    fi

    if [ "$install_for_all" = "true" ] || [ "$install_for_all" = "y" ]; then
        comm_set_install_for_all_mod "mod" "$mod"
    fi
    chmod "$mod" "$path"
    if [ $? -ne 0 ]; then
        comm_log "WARNING" "chmod path $path $mod failed."
        return 1
    fi

    chown -f "$own" "$path"
    if [ $? -ne 0 ]; then
        comm_log "WARNING" "chown path $path $own failed."
        return 1
    fi
}

# 创建子包目录
comm_create_package_dir() {
    local package_dir="$1"
    local own="$2"
    local install_for_all="$3"
    if [ ! -d "$package_dir" ]; then
        comm_create_dir "$package_dir" "755" "$own" "$install_for_all"
    fi
}

# 创建install_info文件
comm_create_install_info_by_path() {
    local install_info="$1"
    local own="$2"
    if [ ! -f "$install_info" ]; then
        comm_create_file "$install_info" "640" "$own" "false"
    fi
}

# 更新安装参数
comm_update_install_param() {
    local key="$1"
    local val="$2"
    local file="$3"
    local param=""
    if [ ! -f "${file}" ]; then
        return 1
    fi
    param="$(grep -i "^${key}=" "${file}")"
    if [ "${param}" = "" ]; then
        echo "${key}=${val}" >> "${file}"
    else
        sed -i "/^${key}=/Ic ${key}=${val}" "${file}"
    fi
}

# 获取安装参数
# _outvar : [输出变量]，安装参数值
# _file : install.info文件路径
# _key : 参数名
comm_get_install_param() {
    local _outvar="$1"
    local _file="$2"
    local _key="$3"
    local _result

    if [ ! -f "${_file}" ]; then
        comm_log "WARNING" "file ${_file} doesn't exist in get install param."
        return 1
    fi
    _result="$(grep -i "^${_key}=" "${_file}" | cut -d"=" -f2-)"
    eval "${_outvar}=\"${_result}\""
}

# 解析安装参数选项
__parse_install_options() {
    local -n install_path_param_ref="$1"
    local -n install_type_ref="$2"
    local -n username_ref="$3"
    local -n usergroup_ref="$4"
    local -n feature_type_ref="$5"

    while true
    do
        case "$1" in
        --install-path-param=*)
            install_path_param_ref="$(echo "$1" | cut -d"=" -f2-)"
            shift
            ;;
        --install-type=*)
            install_type_ref="$(echo "$1" | cut -d"=" -f2-)"
            shift
            ;;
        --username=*)
            username_ref="$(echo "$1" | cut -d"=" -f2-)"
            shift
            ;;
        --usergroup=*)
            usergroup_ref="$(echo "$1" | cut -d"=" -f2-)"
            shift
            ;;
        --feature-type=*)
            feature_type_ref="$(echo "$1" | cut -d"=" -f2-)"
            shift
            ;;
        *)
            break
            ;;
        esac
    done
}

# 更新安装参数文件
comm_update_install_info() {
    local install_path_param=""
    local install_type=""
    local username=""
    local usergroup=""
    local feature_type=""
    local package install_info

    __parse_install_options "install_path_param" "install_type" "username" "usergroup" "feature_type"

    if [ $# -lt 2 ]; then
        return 1
    fi

    package="$1"
    install_info="$2"

    if [ "$install_path_param" != "" ]; then
        comm_update_install_param "${package}_Install_Path_Param" "$install_path_param" "$install_info"
    fi
    if [ "$install_type" != "" ]; then
        comm_update_install_param "${package}_Install_Type" "$install_type" "$install_info"
    fi
    if [ "$username" != "" ]; then
        comm_update_install_param "${package}_UserName" "$username" "$install_info"
    fi
    if [ "$usergroup" != "" ]; then
        comm_update_install_param "${package}_UserGroup" "$usergroup" "$install_info"
    fi
    if [ "$feature_type" != "" ]; then
        comm_update_install_param "${package}_Feature_Type" "$feature_type" "$install_info"
    fi
}

# 开始安装前打印开始信息
comm_start_log() {
    local all_parma="$@"
    local cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
    comm_log "INFO" "Start time:$cur_date"
    comm_log "INFO" "LogFile:${COMM_LOGFILE}"
    comm_log "INFO" "InputParams:$all_parma"
    comm_log "INFO" "OperationLogFile:${COMM_OPERATION_LOGFILE}"
}

# 安装结束退出前打印结束信息
comm_exit_log() {
    local cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
    comm_log "INFO" "End time:${cur_date}"
    exit "$1"
}

comm_print_usage() {
    local runfilename="$1"
    comm_log "INFO" "Please input this command for help: ./${runfilename} --help"
}

# 判断安装路径是否合法
comm_judgmentpath() {
    local install_path="$1"
    local pkg_name="$2"
    check_install_path_valid "$install_path"
    if [ $? -ne 0 ]; then
        comm_log "ERROR" "The $pkg_name install_path $install_path is invalid, only characters in [a-z,A-Z,0-9,-,_] are supported!"
        exit 1
    fi
}

# 标准化安装路径
# _outvar : [输出变量]，处理后的安装路径
# _install_path : 安装路径
comm_normalize_install_path() {
    local _outvar="$1"
    local _install_path="$2"
    local _slashes_num _result
    _slashes_num=$(echo "$_install_path" | grep -o '/' | wc -l)
    if [ "$_slashes_num" -gt 1 ]; then
        _result=$(echo "$_install_path" | sed "s/\/*$//g")
    else
        _result="$_install_path"
    fi
    eval "${_outvar}=\"${_result}\""
}

# 解析安装路径
# _outvar : [输出变量]，处理后的安装路径
# _install_path : 安装路径
# _pkg_name : 包名
comm_parse_install_path() {
    local _outvar="$1"
    local _install_path="$2"
    local _pkg_name="$3"

    comm_judgmentpath "$_install_path" "$_pkg_name"
    comm_normalize_install_path "$_outvar" "$_install_path"
}

############### 错误函数 ###############
# 文件没有找到
comm_err_file_or_directory_not_exist() {
    comm_log "ERROR" "The file or directory doesn't exist, $1"
    comm_exit_log 1
}

os_name() {
    if [ ! -f "$1" ];then
        HostOsName=unknown
        HostOsFullName=unknown
        HostOsVersion=unknown
        return
    fi

    HostOsName=$(cat "$1" | grep ^NAME= | awk -F "[\" ]" '{print $2}')
    HostOsFullName=$(cat "$1" | grep ^NAME= | awk -F "\"" '{print $2}')

    if [ x"$HostOsName" = "x" ];then
        HostOsName=$(cat "$1" | grep ^NAME= | awk -F "[=]" '{print $2}')
        HostOsFullName=$(cat "$1" | grep ^NAME= | awk -F "\"" '{print $2}')
    fi
    if [ x"$HostOsName" = "x" ];then
        HostOsName=unknown
        HostOsFullName=unknown
    fi
    HostOsVersion=$(cat "$1" | grep ^VERSION_ID= | awk -F "\"" '{print $2}')
    if [ x"$HostOsVersion" = "x" ];then
        HostOsVersion=unknown
    fi
    return
}

get_os_info() {
    if [ -f /etc/os-release ];then
        os_name /etc/os-release
    elif [ -f /etc/centos-release ];then
        HostOsName=CentOS
        HostOsFullName="CentOS Linux"
        HostOsVersion=$(cat /etc/centos-release | awk '{print $4}')
    else
        which lsb_release >/dev/null 2>&1
        if [ $? -eq 0 ];then
            HostOsName=$(lsb_release -si)
            HostOsFullName="${HostOsName}"
            HostOsVersion=$(lsb_release -sr)
        else
            os_name /etc/issue
        fi
    fi
    return
}

get_system_info() {
    get_os_info
    HostArch=$(uname -m)
    KernelVersion=$(uname -r)
}

version_gt() {
    if [ "$2"x = "x" ];then
        return 0
    else
        test "$(echo "$@" | tr " " "\n" | sort -V | head -n 1)" != "$1"
    fi
}
version_le() {
    if [ "$2"x = "x" ];then
        return 0
    else
        test "$(echo "$@" | tr " " "\n" | sort -V | head -n 1)" = "$1"
    fi
}
version_lt() {
    if [ "$2"x = "x" ];then
        return 0
    else
        test "$(echo "$@" | tr " " "\n" | sort -rV | head -n 1)" != "$1"
    fi
}
version_ge() {
    if [ "$2"x = "x" ];then
        return 0
    else
        test "$(echo "$@" | tr " " "\n" | sort -rV | head -n 1)" = "$1"
    fi
}

version_vaild() {
    local ver_range="$1"        #version range
    local ver="$2"              #version provider
    local clean_ver_range=$(echo "$ver_range" | sed 's/"//g')   # clean ver_range,去除引号啥的
    local new_ver_range=$(echo "$clean_ver_range" | sed 's/\[//' | sed 's/\]//' | sed 's/(//' | sed 's/)//')  #处理过的version range,去除[],() 得到类似 1.0,2.0

    local start=$(echo $new_ver_range | awk -F ',' '{print $1}')
    local end=$(echo $new_ver_range | awk -F ',' '{print $2}')

    if  echo $clean_ver_range | grep -Eq "^\[.*\]$" ; then
        # 闭合区间 [a,b]={x|a<=x<=b}
        if  version_ge $ver $start  && version_le $ver $end ; then
            # pass
            return 0
        fi
    elif echo $clean_ver_range | grep -Eq "^\[.*\)$"; then
        # 左闭右开 [a,b)={x|a<=x<b}
        if  version_ge $ver $start  && version_lt $ver $end ; then
            return 0
        fi
    elif echo $clean_ver_range | grep -Eq "^\(.*\]$"; then
        # 左开右闭 (a,b]={x|a<x<=b}
        if  version_gt $ver $start  && version_le $ver $end ; then
            return 0
        fi
    elif echo $clean_ver_range | grep -Eq "^\(.*\)$"; then
        # 开区间 (a,b)={x|a<x<b}
        if  version_gt $ver $start  && version_lt $ver $end ; then
            return 0
        fi
    else
        # 兼容老版本,ver_range=require_ver,ver >= require_ver 结果pass 否则 nopass。即依赖包的当前版本大于等于需求
        if version_ge $ver $clean_ver_range ;then
            return 0
        fi
    fi
    #not pass
    return 1
}

# 检查包接口版本。
# 输出VerCheckStatus和ver_check_status变量。
check_pkg_ver_deps() {
    ver_path="$1"
    req_pkg="$2"
    req_ver_path="$3"

    ver_info_list=$(awk -F '[_=]' '$1=="required" && $2=="'$req_pkg'" {print $1":"$2":"$3":"$4":"$5}' "$ver_path")

    for line in ${ver_info_list}
    do
        itf=`echo $line | cut -d":" -f 3`
        ver=`echo $line | cut -d":" -f 5`
        req_ver=`awk -F '=' '$1=="'$itf'_version" {print $2}' "$req_ver_path"`
        if ! version_vaild $ver $req_ver; then
            VerCheckStatus=FAIL
            ver_check_status=FAIL
            return
        fi
    done
    VerCheckStatus=SUCC
    ver_check_status=SUCC
}

# 检查安装路径是否合法。
# install_path : 安装路径
check_install_path_valid() {
    local install_path="$1"
    # 黑名单设置，不允许//，...这样的路径
    if echo "${install_path}" | grep -Eq '\/{2,}|\.{3,}'; then
        return 1
    fi
    # 白名单设置，只允许常见字符
    if echo "${install_path}" | grep -Eq '^\~?[a-zA-Z0-9./_-]*$'; then
        return 0
    else
        return 1
    fi
}

# 创建相对软链接
# mdc的release包场景，ln命令不支持-r参数，需要手动实现相对软链功能
# 入参需要是规范（绝对）路径，完整路径。例如：目标路径不能是一个目录，否则相对软链计算结果不正确
# src_path : 源路径
# dst_path : 目标路径
create_softlink_icp_relative() {
    local src_path="$1"
    local dst_path="$2"
    local source="top${src_path}"
    local target="top${dst_path}"

    # 若变量内容从尾向前的数据符合，则将符合的最短数据删除
    local common="${target%/*}"
    # 若变量内容从头开始的数据符合，则将符合的最短数据删除
    local forward="${source#"$common"/}"

    local result=""

    while [ "${forward}" = "${source}" ]; do
        common="$(dirname "$common")"
        forward="${source#"$common"/}"
        result="../${result}"
    done

    result="${result}${forward}"

    ln -sfn "${result}" "${dst_path}"
    if [ $? -ne 0 ]; then
        log "ERROR" "create softlink relative from ${src_path} to ${dst_path} failed!"
        return 1
    fi
    return 0
}

# 创建软链接
create_softlink() {
    local _src_dir="$1"
    local _dst_dir="$2"
    local _sub_dir_src="$3"
    local _sub_dir_dst="$4"
    local _sub_dir_dst_new _dst_path ret total_ret=0

    # 注意_sub_dir_src可能有通配符
    for _src_path in "${_src_dir}"/${_sub_dir_src}
    do
        if [ -z "$_sub_dir_dst" ]; then
            _sub_dir_dst_new="$(basename "${_src_path}")"
        else
            _sub_dir_dst_new="${_sub_dir_dst}"
        fi
        _dst_path="${_dst_dir}/${_sub_dir_dst_new}"
        if [ -L "${_dst_path}" ]; then
            rm -f "${_dst_path}"
        fi
        # 目标取规范化路径
        _dst_path="$(readlink -f "${_dst_path}")"
        if [ -d "${_dst_path}" ]; then
            # 如果目标为目录，拼接上源文件名
            _dst_path="${_dst_path}/$(basename "${_src_path}")"
        fi

        # 源取规范化路径
        _src_path="$(readlink -f "${_src_path}")"

        create_softlink_icp_relative "${_src_path}" "${_dst_path}"
        ret="$?" && [ $ret -ne 0 ] && total_ret=$ret
    done

    return $total_ret
}

# 创建软链接，调用ln -r命令
create_softlink_by_relative_ln() {
    local _src_dir="$1"
    local _dst_dir="$2"
    local _sub_dir_src="$3"
    local _sub_dir_dst="$4"
    local _src_path="$_src_dir/$_sub_dir_src"
    # 注意_src_path可能有通配符，处理流程中需要考虑通配场景。
    local _dst_path
    if [ -z "$_sub_dir_dst" ]; then
        _sub_dir_dst=$(basename $_src_path)
    fi
    _dst_path="$_dst_dir/$_sub_dir_dst"
    if [ -L "${_dst_path}" ]; then
        rm -f "${_dst_path}"
    fi
    ln -sr $_src_path $_dst_path
}

# 如果源文件存在，则创建软链接
create_softlink_if_exists() {
    local _src_dir="$1"
    local _sub_dir_src="$3"
    local _src_path="$_src_dir/$_sub_dir_src"

    if [ -e $_src_path ]; then
        create_softlink "$@"
    fi
}

# 目录是否为空
# 空目录返回0
# 目录不存在返回1
# 目录非空或无权限访问返回2
is_dir_empty() {
    # 如果目录不存在，则返回
    if [ ! -d "$1" ]; then
        return 1
    fi
    # 否则检查目录是否为空
    # 2>&1重定向无权限访问报错信息
    if [ "$(ls -A "$1" 2>&1)" != "" ]; then
        return 2
    fi
    return 0
}

# 删除空目录
remove_dir_if_empty() {
    local dirpath="$1"

    if is_dir_empty "${dirpath}"; then
        rm -rf "${dirpath}"
    fi
    return 0
}

# 删除目录
remove_dir() {
    local dirpath="$1"

    if [ -e "${dirpath}" ]; then
        chmod u+w -R "${dirpath}"
    fi
    rm -rf "${dirpath}"

    return 0
}

# 删除软链接
remove_softlink() {
    local path="$1"
    [ -L "${path}" ] && rm -f "${path}"
}

# 处理pre_check
# pkg_name : 包名
# pre_check_func : pre_check函数
# standalone : 是否为独立的pre_check命令[y/n]
process_pre_check() {
    local pkg_name="$1"
    local pre_check_func="$2"
    local standalone="$3"

    log "INFO" "${pkg_name} do pre_check started."
    ${pre_check_func}
    ret=$?
    if [ $ret -ne 0 ]; then
        log "WARNING" "${pkg_name} do pre check failed."
    fi
    log "INFO" "${pkg_name} do pre_check finished."

    if [ "${standalone}" = "y" ]; then
        if [ $ret -ne 0 ]; then
            exitInstallLog 1
        fi
        exitInstallLog 0
    fi
}

# 包是否在环境上安装
# install_path : 安装路径
# pkgname : 包名
# varname : [输出变量]，包是否在环境上安装
does_pkg_installed() {
    local install_path="$1"
    local pkgname="$2"
    local varname="$3"
    local result="false"
    local pkg_path="${install_path}/share/info/${pkgname}"

    if [ -d "${pkg_path}" ]; then
        result="true"
    fi

    read -r "$varname" <<EOF
$result
EOF
}


# 获取驱动包安装路径
# 搜索顺序:
# 1. 查看同级目录下是否存在driver包
# 2. 查看/etc/ascend_install.info中配置的路径
# _outvar : [输出变量]，driver包安装路径
# _install_path : 安装路径
get_driver_install_path() {
    local _outvar="$1"
    local _install_path="$2"

    local _install_info="/etc/ascend_install.info"
    local _ascend_hal_name="libascend_hal.so"
    local _old_ifs
    local _var
    local _driver_install_path
    local _ldconfig_result
    local _ldconfig_cnt

    if [ -d "${_install_path}/driver" ]; then
        eval "${_outvar}=\"${_install_path}\""
        return 0
    fi

    # 第一种方案
    if [ -f ${_install_info} ]; then
        . ${_install_info}
        if [ ! -z ${Driver_Install_Path_Param} ]; then
            _driver_install_path="${Driver_Install_Path_Param}"
            eval "${_outvar}=\"${_driver_install_path}\""
            return 0
        fi
    fi

    eval "${_outvar}=\"\""
}

# 执行卸载脚本
# install_path : 安装路径
# pkgname : 包名
run_uninstall_script() {
    local install_path="$1"
    local pkgname="$2"
    local uninstall_path="${install_path}/${pkgname}/script/uninstall.sh"

    if [ -f "${uninstall_path}" ]; then
        "${uninstall_path}"
        return $?
    else
        return 1
    fi
}

# 获取包版本。
get_package_version() {
    local _outvar="$1"
    local _version_info_path="$2"
    local _result

    if [ ! -f "${_version_info_path}" ]; then
        eval "${_outvar}=\"\""
        return 1
    fi

    _result="$(grep "^Version=" "${_version_info_path}" | cut -d= -f2-)"
    eval "${_outvar}=\"${_result}\""
}

# 获取包版本
# _outvar : [输出变量]，包版本
# _version_info_path : version.info文件路径
get_version() {
    get_package_version "$@"
}

# 获取包版本目录
# _outvar : [输出变量]，包版本目录
# _version_info_path : version.info文件路径
get_version_dir() {
    local _outvar="$1"
    local _version_info_path="$2"
    local _result

    if [ ! -f "${_version_info_path}" ]; then
        eval "${_outvar}=\"\""
        return 1
    fi

    _result="$(grep "^version_dir=" "${_version_info_path}" | cut -d= -f2-)"
    eval "${_outvar}=\"${_result}\""
}

# 获取包架构
# _outvar : [输出变量]，包架构
# _scene_info_path : scene.info文件路径
get_arch() {
    local _outvar="$1"
    local _scene_info_path="$2"
    local _result

    if [ ! -f "${_scene_info_path}" ]; then
        eval "${_outvar}=\"\""
        return 1
    fi

    _result="$(grep "^arch=" "${_scene_info_path}" | cut -d= -f2-)"
    eval "${_outvar}=\"${_result}\""
}

# 获取包目录
# _outvar : [输出变量]，包目录
# _version_info_path : version.info文件路径
# _install_path : 安装路径
# _pkg_name : 包名
get_install_package_dir() {
    local _outvar="$1"
    local _version_info_path="$2"
    local _install_path="$3"
    local _pkg_name="$4"
    local _pkg_is_multi_version _pkg_version_dir _result

    is_multi_version_pkg "_pkg_is_multi_version" "$_version_info_path"
    get_version_dir "_pkg_version_dir" "$_version_info_path"

    if [ "$_pkg_is_multi_version" = "true" ]; then
        _result="$_install_path/$_pkg_version_dir/$_pkg_name"
    else
        _result="$_install_path/$_pkg_name"
    fi
    eval "${_outvar}=\"${_result}\""
}

# 包名转换为日志输出的包名（首字母大写）
_package_to_log_pkg_name() {
    local _outvar="$1"
    local _pkg_name="$2"

    local _result

    _result="$(awk -v pkg_name="${_pkg_name}" '
        BEGIN {print toupper(substr(pkg_name, 1, 1)) substr(pkg_name, 2)}
    ')"

    eval "${_outvar}=\"${_result}\""
}

# 获取首字母大字的包名
get_titled_package_name() {
    _package_to_log_pkg_name "$@"
}

# 解析预安装参数选项
__parse_preinstall_options() {
    local -n install_path_ref="$1"
    local -n docker_root_ref="$2"
    local -n script_dir_ref="$3"
    local -n package_ref="$4"
    local -n logfile_ref="$5"
    local -n logstyle_ref="$6"

    # 参数列表末尾添加--参数
    eval set -- "$@ --"

    while true
    do
        case "$1" in
        --install-path=*)
            install_path_ref="$(echo "$1" | cut -d"=" -f2-)"
            shift
            ;;
        --docker-root=*)
            docker_root_ref="$(echo "$1" | cut -d"=" -f2-)"
            shift
            ;;
        --script-dir=*)
            script_dir_ref="$(echo "$1" | cut -d"=" -f2-)"
            shift
            ;;
        --package=*)
            package_ref="$(echo "$1" | cut -d"=" -f2-)"
            shift
            ;;
        --logfile=*)
            logfile_ref="$(echo "$1" | cut -d"=" -f2-)"
            shift
            ;;
        --logstyle=*)
            logstyle_ref="$(echo "$1" | cut -d"=" -f2-)"
            shift
            ;;
        *)
            break
            ;;
        esac
    done
}

# 验证预安装必需参数
__validate_preinstall_params() {
    local install_path="$1"
    local script_dir="$2"
    local package="$3"
    local logfile="$4"

    if [ "${install_path}" = "" ]; then
        echo "error: install_path is needed!"
        return 1
    fi
    if [ "${script_dir}" = "" ]; then
        echo "error: script_dir is needed!"
        return 1
    fi

    if [ "${package}" = "" ]; then
        echo "error: package is needed!"
        return 1
    fi
    if [ "${logfile}" = "" ]; then
        echo "error: logfile is needed!"
        return 1
    fi
}

# 安装前处理。
# 返回码：
# 0: 成功
# 1: 版本兼容性检查失败
# 调用方式：
# preinstall_process --install-path=<install_path> --script-dir=<script-dir> --package=<package> --logfile=<logfile> [ --logstyle=<logstyle> ] [ --docker-root=<docker_root> ]
#
# --install-path=<install_path>  : 安装路径
# --script-dir=<script-dir>      : 当前脚本目录路径
# --package=<package>            : 当前包名
# --logfile=<logfile>            : ascend_install.log日志文件路径
# --logstyle=<logstyle>          : 日志风格
# --docker-root=<docker_root>    : docker根路径
preinstall_process() {
    local install_path=""
    local docker_root=""
    local script_dir=""
    local package=""
    local logfile=""
    local real_install_path
    local version_info_path
    local ret

    __parse_preinstall_options "install_path" "docker_root" "script_dir" "package" "logfile" "LOG_STYLE"

    __validate_preinstall_params "$install_path" "$script_dir" "$package" "$logfile" || return 1

    LOG_FILE="${logfile}"
    _package_to_log_pkg_name "LOG_PKG_NAME" "${package}"

    if [ "${docker_root}" = "" ]; then
        real_install_path="${install_path}"
    else
        real_install_path="${docker_root}/${install_path}"
    fi

    _check_version_compatiable "$package" "${real_install_path}" "${script_dir}"
    if [ $? -ne 0 ]; then
        return 1
    fi

    return 0
}

# 安装前检查。
# 返回码：
# 0: 成功
# 1: 版本兼容性检查失败
# 调用方式：
# preinstall_check --install-path=<install_path> --script-dir=<script-dir> --package=<package> --logfile=<logfile> [ --logstyle=<logstyle> ] [ --docker-root=<docker_root> ]
#
# --install-path=<install_path>  : 安装路径
# --script-dir=<script-dir>      : 当前脚本目录路径
# --package=<package>            : 当前包名
# --logfile=<logfile>            : ascend_install.log日志文件路径
# --logstyle=<logstyle>          : 日志风格
# --docker-root=<docker_root>    : docker根路径
preinstall_check() {
    preinstall_process "$@"
}

# 是否为多版本包
# 读取version.info文件，检查本包是否为多版本包
# _outvar : [输出变量]，是否为多版本包
# _filepath : version.info文件路径
is_multi_version_pkg() {
    local _outvar="$1"
    local _filepath="$2"
    local _ret="false"

    if [ -f "${_filepath}" ]; then
        grep "^version_dir=" "${_filepath}" > /dev/null
        [ $? -eq 0 ] && _ret="true"
    fi
    eval "${_outvar}=\"${_ret}\""
}

# 文件列表的某一列
_filelist_column() {
    local _filepath="$1"
    local _idx="$2"
    tail -n +2 "$_filepath" | cut -d, -f${_idx} | sort | uniq
}

# 移除comm特性
_remove_comm_feature() {
    local _outvar="$1"
    local _feature="$2"
    local _result

    _result="$(echo "$_feature" | sed 's/comm//g' | sed 's/,,/,/g' | sed 's/^,//g' | sed 's/,$//g')"
    eval "${_outvar}=\"${_result}\""
}

# 特性列表转换为正则表达式
_feature_to_regex() {
    local _outvar="$1"
    local _feature="$2"
    local _result_ftr
    _result_ftr="\b(($(echo "$_feature" | sed 's/,/)|(/g')))\b"
    eval "${_outvar}=\"${_result_ftr}\""
}

# 是否包含指定feature
# 读取filelist.csv文件，检查是否包含指定feature
# _outvar : [输出变量](true/false)，是否包含指定feature
# _feature : 指定feature，可指定多个，以“,”分隔，如：dvpp,audio
# _filepath : filelist.csv文件路径
contain_feature() {
    local _outvar="$1"
    local _feature="$2"
    local _filepath="$3"
    local _feature_regex _feature_cf

    # 输入的comm不作为特性，也就是说不支持--feature=comm
    _remove_comm_feature "_feature_cf" "$_feature"
    if [ "$_feature_cf" = "" ]; then
        eval "${_outvar}=\"false\""
        return 0
    fi

    _feature_to_regex "_feature_regex" "$_feature_cf"
    _filelist_column "$_filepath" 10 | grep -E "$_feature_regex" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        eval "${_outvar}=\"true\""
    else
        eval "${_outvar}=\"false\""
    fi
}

# 向filelist.csv中写入1个条目
# filepath : filelist.csv文件路径
# operation : 操作类型
# path_in_pkg : 包内路径
# install_path : 安装路径
# permission : 文件权限
# owner_group : 文件属主，支持默认值DEFAULT
# install_type : 安装类型
# softlink : 软链接
# block : 所属块
# feature : 所属特性，支持默认值DEFAULT
# chip : 所属芯片，支持默认值DEFAULT
# arch : 包架构
add_fileitem() {
    local filepath="$1"
    local operation="$2"
    local path_in_pkg="$3"
    local install_path="$4"
    local permission="$5"
    local owner_group="$6"
    local install_type="$7"
    local softlink="$8"
    local pkg_inner_softlink="$9"
    local block="${10}"
    local feature="${11}"
    local chip="${12}"
    local arch="${13}"
    local is_common_path="N"

    if [ "$owner_group" = "DEFAULT" ]; then
        owner_group="\\\\\$username:\\\\\$usergroup"
    fi

    if [ "$feature" = "DEFAULT" ]; then
        feature="all"
    fi

    if [ "$chip" = "DEFAULT" ]; then
        chip="all"
    fi

    if echo "$install_path" | grep "^${arch}-linux/" > /dev/null; then
        is_common_path="Y"
    fi

    echo "NA,${operation},${path_in_pkg},${install_path},FALSE,${permission},${owner_group},${install_type},${softlink},${feature},${is_common_path},FALSE,NA,${block},${pkg_inner_softlink},${chip}" >> ${filepath}
}

# 设置公共变量
set_global_vars() {
    local arch
    local scene_filepath="${curpath}/../scene.info"

    if [ -f "${scene_filepath}" ]; then
        get_scene_arch "arch" "${scene_filepath}"
        if [ "${arch}" != "" ]; then
            PKG_ARCH="${arch}"
        fi
    fi
}

# 获取并发进程数
get_thread_num() {
    local _outvar="$1"
    local _thread_num="$(cat /proc/cpuinfo | grep "^processor" | wc -l)"

    # 未取得CPU核数时，默认为1
    if [ "$_thread_num" = "" ] || [ "$_thread_num" = "0" ]; then
        _thread_num="1"
    fi

    # CPU核数2倍
    eval "${_outvar}=\"$((_thread_num*2))\""
}

# 初始化fifo，用于并发控制
init_fifo() {
    local _outvar="$1"
    local _thread_num="$2"
    local _tmppath_if tmp
    local _skip_msg="skip init fifo"

    # 并发数为空时，默认为1
    if [ "$_thread_num" = "" ]; then
        _thread_num="1"
        comm_log "WARNING" "thread number is empty, use ${_thread_num}."
    fi
    # 并发数<=0时，默认为1
    if [ $_thread_num -le 0 ]; then
        comm_log "WARNING" "thread number is ${_thread_num}, use 1."
        _thread_num="1"
    fi

    get_tmp_file _tmppath_if "fifo"
    if [ -e "$_tmppath_if" ]; then
        rm -rf "$_tmppath_if"
        check_ret_warning "$?" "remove old file $_tmppath_if failed, ${_skip_msg}."
        ret="$?" && [ $ret -ne 0 ] && return $ret
    fi

    mkfifo "$_tmppath_if"
    check_ret_warning "$?" "make fifo $_tmppath_if failed, ${_skip_msg}."
    ret="$?" && [ $ret -ne 0 ] && return $ret

    # 使用文件描述符8作并发控制
    exec 8<> "$_tmppath_if"  # 创建文件描述符8
    check_ret_warning "$?" "open fifo $_tmppath_if file descriptor failed, ${_skip_msg}."
    ret="$?" && [ $ret -ne 0 ] && return $ret

    rm -f "$_tmppath_if"
    check_ret_warning "$?" "remove fifo $_tmppath_if failed, ${_skip_msg}."
    ret="$?" && [ $ret -ne 0 ] && return $ret

    for tmp in $(seq $_thread_num); do
        echo >&8
        check_ret_warning "$?" "echo fifo $_tmppath_if failed, ${_skip_msg}."
        ret="$?" && [ $ret -ne 0 ] && return $ret
    done

    eval "${_outvar}=\"${_tmppath_if}\""
}

# 根据参数执行
exec_with_param() {
    local param="$1"
    shift 1
    local exec_mode fifo_path="$PARALLEL_FIFO" tmp
    extract_1st "exec_mode" "$param"
    if [ "$exec_mode" = "concurrency" ]; then
        if [ "$fifo_path" = "none" ] || [ "$fifo_path" = "" ]; then
            "$@" &
        else
            read tmp <&8
            {
                "$@"
                ret="$?"
                echo >&8
                exit $ret
            } &
        fi
    else
        "$@"
    fi
}
