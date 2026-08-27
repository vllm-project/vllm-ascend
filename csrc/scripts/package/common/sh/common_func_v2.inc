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

# 公共函数库
# 总setenv文件权限
SETENV_MOD="550"
# 总setenv文件可写状态权限
SETENV_WRITEABLE_MOD="600"
# 默认特性配置
DEFAULT_FEATURE_PARAM="all n all"
#export PS4='+ ${FUNCNAME[0]:+${FUNCNAME[0]}():} ${BASH_SOURCE}:${LINENO}: '
#set -x
RESET_MOD="750"

# db.info文件格式如下
# ---
# CommonLib|atc,fwkacllib
# Compiler|atc,fwkacllib
# ---
# 使用"|"分隔模块名和包名列表，包名列表使用","分隔
# 文件中块名保持以升序排序
PKG_DB_INFO_RELPATH="var/ascend_package_db.info"
# 缓存mod文件名
STASH_MOD_PATH="stash_mod.txt"
# stash_mod文件权限
STASH_FILE_MOD="600"

# 写日志
log() {
    local cur_date_="$(date +"%Y-%m-%d %H:%M:%S")"
    local log_type_="${1}"
    local msg_="${2}"
    local log_format_="[Common] [${cur_date_}] [${log_type_}]: ${msg_}"
    if [ "${log_type_}" = "INFO" ]; then
        echo "${log_format_}"
    elif [ "${log_type_}" = "WARNING" ]; then
        echo "${log_format_}"
    elif [ "${log_type_}" = "ERROR" ]; then
        echo "${log_format_}"
    elif [ "${log_type_}" = "DEBUG" ]; then
        echo "${log_format_}" 1> /dev/null
    fi
}

# 返回列表长度
__length_list() {
    local list="$1"
    local var="$2"
    local list_item
    local cnt=0

    for list_item in ${list}; do
        cnt=$((cnt+1))
    done

    eval "${var}=\"${cnt}\""
}

# 获取列表索引值
__index_list() {
    local list="$1"
    shift
    local list_item
    local cnt=0

    if [ $# -eq 0 ]; then
        return 0
    fi

    for list_item in ${list}; do
        if [ ${1} -eq ${cnt} ]; then
            eval "${2}=\"${list_item}\""
            shift 2
            if [ $# -eq 0 ]; then
                return 0
            fi
        fi
        cnt=$((cnt+1))
    done

    return 0
}

# 从列表中移除一项
__remove_item_in_list() {
    local to_removed="$1"
    shift
    local list="$*"
    local list_item
    local new_list

    for list_item in ${list}; do
        if [ "${to_removed}" != "${list_item}" ]; then
            if [ "${new_list}" = "" ]; then
                new_list="${list_item}"
            else
                new_list="${new_list} ${list_item}"
            fi
        fi
    done
    echo "${new_list}"
}

# 元素是否在列表中
__item_in_list() {
    local _outvar="$1"
    local _item="$2"
    shift 2
    local _list_item
    local _matched="false"

    for _list_item in $*; do
        if [ "${_item}" = "${_list_item}" ]; then
            _matched="true"
            break
        fi
    done
    eval "${_outvar}=\"${_matched}\""
}

# 反转列表
__reverse_list() {
    local _outvar="$1"
    local _list="$2"
    local _new_list=""
    local _list_item

    for _list_item in ${_list}; do
        if [ "${_new_list}" = "" ]; then
            _new_list="${_list_item}"
        else
            _new_list="${_list_item} ${_new_list}"
        fi
    done
    eval "${_outvar}=\"${_new_list}\""
}

# 修改各文件及目录的属性
change_own() {
    local recursive="$3"
    local option=""
    local username="${USERNAME}"
    local usergroup="${USERGROUP}"
    if [ "$2" != "NA" ]; then
        if [ "${recursive}" = "true" ]; then
            option="-R"
        fi
        eval chown ${option} -h \"$2\" \"$1\"
        if [ $? -ne 0 ]; then
            log "ERROR" "$1 chown failed!"
            return 1
        fi
    fi
}

# 获取install_for_all文件权限
get_install_for_all_mod() {
    local _outvar="$1"
    local _mod="$2"
    local _new_mod _other_mod

    _new_mod="${_mod%?}"
    _other_mod="${_new_mod#${_new_mod%?}}"
    _other_mod="$(($_other_mod & 5))"  # other权限位移除写权限，仅支持普通用户运行

    eval "${_outvar}=\"${_new_mod}${_other_mod}\""
}

# 修改各文件及目录的权限
change_mod() {
    local mod="$2"
    local install_for_all="$3"
    local recursive="$4"
    local option="" new_mod
    # 对于软连接，可能目标文件还没有拷贝进来，导致无法修改mod，这里过滤掉软连接
    if [ -L "$1" ]; then
        return 0
    fi
    if [ "$2" != "NA" ]; then
        if [ "${recursive}" = "true" ]; then
            option="-R"
        fi
        # 如果设置了install_for_all，则安装时other权限跟group权限对齐
        if [ "${install_for_all}" = "y" ]; then
            get_install_for_all_mod new_mod "$mod"
            chmod ${option} "${new_mod}" "$1"
        else
            chmod ${option} "$2" "$1"
        fi
        if [ $? -ne 0 ]; then
            log "ERROR" "$1 chmod failed!"
            return 1
        fi
    fi
    return 0
}

# 获取文件权限
get_file_mod() {
    local _outvar="$1"
    local _options="" _ret
    shift

    while true; do
        case "$1" in
        -L|--dereference)
            _options="${_options} $1"
            shift
            ;;
        *)
            break
            ;;
        esac
    done

    local _path="$1"
    local _result

    _result="$(stat ${_options} -c %a "${_path}")"
    _ret="$?" && [ $_ret -ne 0 ] && return $_ret
    eval "${_outvar}=\"${_result}\""
}

# 检查路径是否为绝对路径
__check_abs_path() {
    local path="$1"

    if [ "${path#/}" != "${path}" ]; then
        is_abs_path="true"
    else
        is_abs_path="false"
    fi
}

__set_abs_path() {
    local install_path="$1"
    local path="$2"
    local varname="$3"
    local is_abs_path

    __check_abs_path "${path}"
    if [ "${is_abs_path}" != "true" ]; then
        eval "${varname}=\"${install_path}/${path}\""
    else
        eval "${varname}=\"${path}\""
    fi
}

# 创建目录
make_dir() {
    local path="$1"
    mkdir -p "${path}"
    if [ $? -ne 0 ]; then
        log "ERROR" "${path} mkdir failed!"
        exit 1
    fi
    return 0
}

# 检查install_path在docker_root之中
check_install_path_in_docker_root() {
    local install_path="$1"
    local docker_root="$2"

    echo "${install_path}" | grep "^${docker_root}" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        log "ERROR" "check install path ${install_path} in docker root ${docker_root} failed!"
        return 1
    fi
    return 0
}

# 移除路径右侧斜线(/)
rstrip_path() {
    local _outvar="$1"
    local _path="$2"

    _path="$(echo "${_path}" | sed "s/\/\+\$//g")"
    eval "${_outvar}=\"${_path}\""
}

# 包是否在tools目录下
is_package_under_tools() {
    local _outvar="$1"
    local _package="$2"
    local _result_iput="false"

    if [ "${_package}" = "aoe" ] || [ "${_package}" = "nca" ] || [ "${_package}" = "amct_acl" ] || [ "${_package}" = "ncs" ]; then
        _result_iput="true"
    fi

    eval "${_outvar}=\"${_result_iput}\""
}

# 获取包目录名
get_package_dir() {
    local _outvar="$1"
    local _package="$2"
    local _is_under_tools _result

    if [ "${USE_SHARE_INFO}" == "y" ]; then
        _result="share/info"
    else
        is_package_under_tools "_is_under_tools" "${_package}"
        if [ "${_is_under_tools}" = "true" ]; then
            _result="tools"
        else
            _result=""
        fi
    fi
    eval "${_outvar}=\"${_result}\""
}

# 获取包目录路径
get_package_dirpath() {
    local _outvar="$1"
    local _package="$2"
    local _is_under_tools _result

     if [ "${USE_SHARE_INFO}" == "y" ]; then
        _result="share/info/${_package}"
    else
        is_package_under_tools "_is_under_tools" "${_package}"
        if [ "${_is_under_tools}" = "true" ]; then
            _result="tools/${_package}"
        else
            _result="${_package}"
        fi
    fi
    eval "${_outvar}=\"${_result}\""
}

# 获取包ascend_install.info路径
get_package_install_info() {
    local _outvar="$1"
    local _install_path="$2"
    local _version_dir="$3"
    local _package="$4"
    local _package_dirpath=""

    eval "${_outvar}=\"\""

    get_package_dirpath "_package_dirpath" "${_package}"

    eval "${_outvar}=\"${_install_path}/${_version_dir}/${_package_dirpath}/ascend_install.info\""
}

# 获取包version.info路径
get_package_version_info() {
    local _outvar="$1"
    local _install_path="$2"
    local _version_dir="$3"
    local _package="$4"
    local _package_dirpath=""

    eval "${_outvar}=\"\""

    get_package_dirpath "_package_dirpath" "${_package}"

    eval "${_outvar}=\"${_install_path}/${_version_dir}/${_package_dirpath}/version.info\""
}

# 获取包filelist.csv路径
get_package_filelist() {
    local _outvar="$1"
    local _install_path="$2"
    local _version_dir="$3"
    local _package="$4"
    local _package_dirpath=""

    eval "${_outvar}=\"\""

    get_package_dirpath "_package_dirpath" "${_package}"

    eval "${_outvar}=\"${_install_path}/${_version_dir}/${_package_dirpath}/script/filelist.csv\""
}

# 获取包install_common_parser.sh路径
get_package_install_common_parser() {
    local _outvar="$1"
    local _install_path="$2"
    local _version_dir="$3"
    local _package="$4"
    local _package_dirpath=""

    eval "${_outvar}=\"\""

    get_package_dirpath "_package_dirpath" "${_package}"

    eval "${_outvar}=\"${_install_path}/${_version_dir}/${_package_dirpath}/script/install_common_parser.sh\""
}

# 获取latest_manager的install_common_parser.sh路径
get_latest_manager_install_common_parser() {
    local _outvar="$1"
    local _install_path="$2"
    local _latest_dir="$3"

    eval "${_outvar}=\"${_install_path}/${_latest_dir}/var/manager/install_common_parser.sh\""
}

# 获取包script目录路径
get_package_script_dirpath() {
    local _outvar="$1"
    local _install_path="$2"
    local _version_dir="$3"
    local _package="$4"
    local _package_dirpath=""

    eval "${_outvar}=\"\""

    get_package_dirpath "_package_dirpath" "${_package}"

    eval "${_outvar}=\"${_install_path}/${_version_dir}/${_package_dirpath}/script\""
}

# 检查参数不为空
check_param_not_empty() {
    local name="$1"
    local error_msg="$2"
    local value

    eval "value=\"\${${name}}\""

    if [ "${value}" = "" ]; then
        comm_log "ERROR" "$2"
        return 1
    fi

    return 0
}

# 检查文件存在
check_file_exists() {
    local path="$1"
    local error_msg="$2"

    if [ ! -f "${path}" ]; then
        comm_log "ERROR" "$2"
        return 1
    fi

    return 0
}

# 检查返回值是否为0
check_ret_error() {
    local ret="$1"
    local msg="$2"

    if [ ${ret} -ne 0 ]; then
        comm_log "ERROR" "${msg}"
        return ${ret}
    fi

    return 0
}

# 检查返回值是否为0
check_ret_warning() {
    local ret="$1"
    local msg="$2"

    if [ ${ret} -ne 0 ]; then
        comm_log "WARNING" "${msg}"
        return ${ret}
    fi

    return 0
}

# 获取真实路径
get_realpath() {
    local _outvar="$1"
    local _path_gr="$2"

    _path_gr="$(readlink -f "${_path_gr}")"
    eval "${_outvar}=\"${_path_gr}\""
}

# 检查返回值是否为0
cleanup_if_error() {
    local ret="$1"
    local cleanup="$2"

    if [ ${ret} -ne 0 ]; then
        eval "${cleanup}"
        return ${ret}
    fi

    return 0
}

# 获取临时目录
get_tmp_root() {
    local _outvar="$1"
    local _result_gtr

    if [ -d "${HOME}" ]; then
        _result_gtr="${HOME}"
    elif [ $(id -u) -eq 0 ] && [ -d "/root" ]; then
        _result_gtr="/root"
    else
        _result_gtr="${PWD}"
    fi

    eval "${_outvar}=\"${_result_gtr}\""
    return 0
}

# 获取临时文件
get_tmp_file() {
    local _outvar="$1"
    local _filename="$2"
    local _tmp_file_gtf _result

    get_tmp_root "_tmp_file_gtf"

    _result=$(mktemp "$_tmp_file_gtf/${_filename}_XXXXXX")
    check_ret_warning "$?" "mktemp $_tmp_file_gtf/${_filename}_XXXXXX failed."
    ret="$?" && [ $ret -ne 0 ] && return $ret

    eval "${_outvar}=\"${_result}\""
    return 0
}

# 获取包架构
get_scene_arch() {
    local _outvar="$1"
    local _scene_filepath="$2"
    local _result

    _result="$(grep "^arch=" "${_scene_filepath}" | cut -d= -f2-)"
    eval "${_outvar}=\"${_result}\""
}

# 打包feature参数
pack_feature_param() {
    local _outvar="$1"
    local _feature_type="$2"
    local _feature_exclude_all="$3"
    local _chip="$4"

    eval "${_outvar}=\"${_feature_type} ${_feature_exclude_all} ${_chip}\""
}

# 解包feature参数
# 注意，调用解包时参数不可加引号
unpack_feature_param() {
    local _feature_type_var="$1"
    local _feature_exclude_all_var="$2"
    local _chip_var="$3"
    shift 3
    eval "${_feature_type_var}=\"${1}\""
    eval "${_feature_exclude_all_var}=\"${2}\""
    eval "${_chip_var}=\"${3}\""
}

# 展开version.info文件中参数
expand_version_file() {
    if [ "${VERSION_FILE}" = "" ]; then
        return 0
    fi
    get_version "VERSION" "${VERSION_FILE}"
    get_version_dir "VERSION_DIR" "${VERSION_FILE}"
}

# 提取第一项
extract_1st() {
    local _outvar="$1"
    eval "${_outvar}=\"$2\""
}

# 提取第二项
extract_2nd() {
    local _outvar="$1"
    eval "${_outvar}=\"$3\""
}

# 路径转为sed正则表达式
path_to_regex() {
    local _outvar="$1"
    local _path="$2"
    local _reslut_ptr="$(echo "${_path}" | sed "s#\/#\\\/#g")"

    eval "${_outvar}=\"${_reslut_ptr}\""
}

# 设置默认值
set_default() {
    local _outvar="$1"
    local _input="$2"
    local _default="$3"

    if [ "${_input}" = "" ]; then
        eval "${_outvar}=\"${_default}\""
    else
        eval "${_outvar}=\"${_input}\""
    fi
}

# 标准化feature参数
# 输入的feature参数中，如果有all字段（表示安装所有feature）
# 则将feature参数重置为all，在后续流程中安装所有feature
# 卸载流程中，feature与chip强制为all，保证卸载掉block中的所有文件
normalize_feature() {
    local _outvar="$1"
    local _feature_nf="$2"
    local _operation="$3"

    if [ "$_operation" = "uninstall" ]; then
        eval "${_outvar}=\"all\""
        return 0
    fi

    if [ "$_feature_nf" = "" ]; then
        eval "${_outvar}=\"all\""
        return 0
    fi

    case "${_feature_nf}" in
    all,*)
        _feature_nf="all"
        ;;
    *,all)
        _feature_nf="all"
        ;;
    *,all,*)
        _feature_nf="all"
        ;;
    esac

    eval "${_outvar}=\"${_feature_nf}\""
}

# 删除软连接
remove_softlink_icp() {
    local softlink="$1"

    if [ "${softlink}" = "" ]; then
        return 0
    fi

    if [ "${softlink}" != "NA" ] && [ -L "${softlink}" ]; then
        rm -f "${softlink}"
        if [ $? -ne 0 ]; then
            log "ERROR" "remove ${softlink} failed!"
            return 1
        fi
    fi
    return 0
}

#移除文件
remove_file() {
    local target="$1"
    local softlink="$2"
    local ret
    if [ -e "${target}" ] || [ -L "${target}" ]; then
        rm -f "${target}"
        if [ $? -ne 0 ]; then
            log "ERROR" "remove ${target} failed!"
            return 1
        fi
    fi
    remove_softlink_icp "${softlink}"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    return 0
}

# 创建文件夹
create_folder() {
    local install_path="$1"
    local target="$2"
    local softlinks_str="$3"
    local ret target_abs

    __set_abs_path "${install_path}" "${target}" "target_abs"

    if [ ! -d "${target_abs}" ]; then
        make_dir "${target_abs}"
        ret="$?" && [ ${ret} -ne 0 ] && return ${ret}
    fi
    change_mod "${target_abs}" "${RESET_MOD}" ""
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}
    if [ "${softlinks_str}" != "NA" ]; then
        create_softlink_by_install_path "${install_path}" "${target}" "${softlinks_str}"
        ret="$?" && [ ${ret} -ne 0 ] && return ${ret}
    fi
    return 0
}

# 创建目录
create_dirs() {
    local install_path="$1"
    local line="$2"
    local target
    local target_abs
    local softlinks_str
    local ret

    __index_list "${line}" 1 "target" 4 "softlinks_str"

    __set_abs_path "${install_path}" "${target}" "target_abs"

    if [ -L "${target_abs}" ] ; then
        rm -f "${target_abs}"
        log "WARNING" "${target_abs} is an existing soft-link, deleted."
    fi
    create_folder "${install_path}" "${target}" "${softlinks_str}"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}
    return 0
}

# 修改目录的权限和属组
reset_mod_dirs() {
    local install_path="$1"
    local line="$2"
    local mod
    local target
    local target_abs
    local is_abs_path ret

    __index_list "${line}" 1 "target"

    __set_abs_path "${install_path}" "${target}" "target_abs"

    # 目录不存在时跳过
    if [ ! -d "${target_abs}" ]; then
        return 0
    fi
    # 只处理目录，没有处理目录的软链接
    change_mod "${target_abs}" "${RESET_MOD}" "" "false"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    return 0
}

# 递归修改目录的权限和属组
reset_mod_dirs_recursive() {
    local install_path="$1"
    local line="$2"
    local mod
    local target
    local target_abs
    local is_abs_path ret

    __index_list "${line}" 1 "target"

    __set_abs_path "${install_path}" "${target}" "target_abs"

    # 目录不存在时跳过
    if [ ! -d "${target_abs}" ]; then
        return 0
    fi
    # 只处理目录，没有处理目录的软链接
    change_mod "${target_abs}" "${RESET_MOD}" "" "true"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    return 0
}

__unpack_softlinks() {
    local softlinks_str="$1"
    local varname="$2"
    local item
    local temp
    OLD_IFS="${IFS}"
    IFS=";"
    temp=""
    for item in ${softlinks_str}; do
        if [ "${temp}" = "" ]; then
            temp="${item}"
        else
            temp="${temp} ${item}"
        fi
    done
    IFS="${OLD_IFS}"

    eval "${varname}=\"${temp}\""
}

#移除文件夹
remove_dir_icp() {
    if [ -e "$1" ] || [ -L "$1" ]; then
        rm -fr "$1"
        if [ $? -ne 0 ]; then
            log "ERROR" "$1 remove failed!"
            return 1
        fi
    fi
    return 0
}

# 创建软链时，移除存在的目录
remove_exists_dir_in_create_softlink() {
    local dirpath="$1"

    if [ -d "${dirpath}" ] && [ ! -L "${dirpath}" ]; then
        log "WARNING" "${dirpath} is an existing directory in create softlink, deleted."
        change_mod "${dirpath}" "700" "n" "true"
        remove_dir_icp "${dirpath}"
    fi
}

# 创建绝对软链接
create_softlink_icp_absolute() {
    local src_path="$1"
    local dst_path="$2"

    remove_exists_dir_in_create_softlink "${dst_path}"
    ln -sfn "${src_path}" "${dst_path}"
    if [ $? -ne 0 ]; then
        log "ERROR" "create softlink absolute from ${src_path} to ${dst_path} failed!"
        return 1
    fi
    return 0
}

# 该函数与common_func.inc脚本中的相同
# install_common_parser.sh不一定能source到common_func.inc（原因见source common_func.inc的注释）
# 所以这里需要重复定义
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

    remove_exists_dir_in_create_softlink "${dst_path}"
    ln -sfn "${result}" "${dst_path}"
    if [ $? -ne 0 ]; then
        log "ERROR" "create softlink relative from ${src_path} to ${dst_path} failed!"
        return 1
    fi
    return 0
}

# 创建软连接
create_softlink_by_install_path() {
    local install_path="$1"
    local target="$2"
    local softlinks_str="$3"
    local softlinks softlink
    local target_abs
    local softlink_abs
    local is_abs_path
    local ret

    if [ "${softlinks_str}" = "NA" ]; then
        return 0
    fi

    __unpack_softlinks "${softlinks_str}" "softlinks"

    for softlink in ${softlinks}; do
        __check_abs_path "${target}"
        if [ "${is_abs_path}" = "true" ]; then
            __set_abs_path "${install_path}" "${softlink}" "softlink_abs"
            create_softlink_icp_absolute "${target}" "${softlink_abs}"
            ret="$?" && [ ${ret} -ne 0 ] && return ${ret}
            continue
        fi

        __check_abs_path "${softlink}"
        if [ "${is_abs_path}" = "true" ]; then
            __set_abs_path "${install_path}" "${target}" "target_abs"
            create_softlink_icp_absolute "${target_abs}" "${softlink}"
            ret="$?" && [ ${ret} -ne 0 ] && return ${ret}
            continue
        fi

        create_softlink_icp_relative "${install_path}/${target}" "${install_path}/${softlink}"
        ret="$?" && [ ${ret} -ne 0 ] && return ${ret}
    done

    return 0
}

# 创建软链接。icp后缀为避免重名
create_softlink_icp() {
    local options=""
    local is_relative="false"

    while true; do
        case "$1" in
        -r|--relative)
            is_relative="true"
            shift
            ;;
        -*)
            log "ERROR" "unsupported option $1 in create softlink icp!"
            return 1
            ;;
        *)
            break
            ;;
        esac
    done

    local src_path="$1"
    local dst_path="$2"

    if [ "${is_relative}" = "true" ]; then
        create_softlink_icp_relative "${src_path}" "${dst_path}"
    else
        create_softlink_icp_absolute "${src_path}" "${dst_path}"
    fi
}

# 修改权限和属组
change_mod_and_own(){
    local target="$1"
    local mod="$2"
    local own="$3"
    local install_for_all="$4"
    local recursive="$5"
    local ret

    change_mod "${target}" "${mod}" "${install_for_all}" "${recursive}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    change_own "${target}" "${own}" "${recursive}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    return 0
}

# 修改目录的权限和属组
change_mod_and_own_dirs() {
    local install_path="$1"
    local line="$2"
    local target
    local mod
    local own
    local is_abs_path

    __index_list "${line}" 1 "target" 2 "mod" 3 "own"

    __check_abs_path "${target}"
    if [ "${is_abs_path}" != "true" ]; then
        target="${install_path}/${target}"
    fi
    if [ ! -d "${target}" ]; then
        return 0
    fi
    # 只处理目录，没有处理目录的软链接
    change_mod_and_own "${target}" "${mod}" "${own}" "${INSTALL_FOR_ALL}" "false"
}

# 创建stash_mod.txt文件
create_stash_mod() {
    local install_path="$1"
    rm -f "${install_path}/${STASH_MOD_PATH}"
    touch "${install_path}/${STASH_MOD_PATH}"
    chmod ${STASH_FILE_MOD} "${install_path}/${STASH_MOD_PATH}"
}

# 删除stash_mod.txt文件
remove_stash_mod() {
    local install_path="$1"
    rm -f "${install_path}/${STASH_MOD_PATH}"
}

# 修改目录的权限和属组
restore_stash_mod() {
    local install_path="$1"
    local line="$2"
    local target
    local target_abs
    local mod
    local is_abs_path ret

    __index_list "${line}" 0 "target" 1 "mod"

    __set_abs_path "${install_path}" "${target}" "target_abs"

    # 目录不存在时跳过
    if [ ! -d "${target_abs}" ]; then
        return 0
    fi
    # 只处理目录，没有处理目录的软链接
    change_mod "${target_abs}" "${mod}" "${INSTALL_FOR_ALL}"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    return 0
}

# 修改目录的权限，暂存原文件权限
reset_mod_dirs_with_stash_mod() {
    local install_path="$1"
    local line="$2"
    local mod
    local target
    local target_abs
    local is_abs_path ret

    __index_list "${line}" 1 "target"

    __set_abs_path "${install_path}" "${target}" "target_abs"

    # 目录不存在时跳过
    if [ ! -d "${target_abs}" ]; then
        return 0
    fi

    get_file_mod "mod" "${target_abs}"
    echo "${target}:${mod}" >> "${install_path}/${STASH_MOD_PATH}"

    # 只处理目录，没有处理目录的软链接
    change_mod "${target_abs}" "${RESET_MOD}" ""
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    return 0
}

# 删除软连接列表
remove_softlinks() {
    local install_path="$1"
    local softlinks_str="$2"
    local softlinks softlink softlink_abs ret

    __unpack_softlinks "${softlinks_str}" "softlinks"

    for softlink in ${softlinks}; do
        __set_abs_path "${install_path}" "${softlink}" "softlink_abs"
        remove_softlink_icp "${softlink_abs}"
        ret="$?" && [ ${ret} -ne 0 ] && return ${ret}
    done
    return 0
}

# 删除安装文件夹
remove_install_dirs() {
    local install_path="$1"
    local line="$2"
    local target softlinks_str is_abs_path

    __index_list "${line}" 1 "target" 4 "softlinks_str"

    if [ "${target}" != "NA" ]; then
        __check_abs_path "${target}"
        if [ "${is_abs_path}" != "true" ]; then
            target="${install_path}/${target}"
        fi
        if [ -d "${target}" ]; then
            # 不同的blocks中，可能配置相同的dir_info。目录不为空时不删除。
            if [ "$(ls -A "${target}")" != "" ]; then
                return 0
            fi
            remove_dir_icp "${target}"
            ret="$?" && [ $ret -ne 0 ] && return $ret
        fi

        if [ "${softlinks_str}" != "NA" ]; then
            remove_softlinks "${install_path}" "${softlinks_str}"
            ret="$?" && [ ${ret} -ne 0 ] && return ${ret}
        fi
    fi
    return 0
}

# 获取所有的blocks
get_blocks_info() {
    local _outvar="$1"
    local _install_path="$2"
    local _result _db_filepath="${_install_path}/${PKG_DB_INFO_RELPATH}"

    if [ ! -f "${_db_filepath}" ]; then
        return 0
    fi
    _result="$(cut -d'|' -f1 "${_db_filepath}" | xargs)"
    eval "${_outvar}=\"${_result}\""
}

# 列表转为正则表达式
trans_list_to_regex() {
    local _outvar="$1"
    local _value="$2"
    # 移除右侧空白符，mawk无法编译右侧存在空白符时的正则表达式，如：^(EngineeringCommon|)$
    local _result="$(echo "$_value" | sed 's/\s\+$//g' | sed 's/ /|/g' | xargs printf "^(%s)$")"
    eval "${_outvar}=\"${_result}\""
}

parse_filelist_core() {
    awk -F, '{ print $3,$4,$6,$7,$9,$12,$13,$14,$15 }'
}

# 过滤公共路径条目
filter_common_dirs() {
    awk -F, '
        $11 == "Y" {
            print $0
        }
        $11 == "YY" {
            print $1 "," $2 "," $3 "," $4 "," $5 "," $6 "," $7 "," $8 ",NA," $10 "," $11 "," $12 "," $13 "," $14 "," $15 "," $16
        }
    '
}

# 过滤包内软链
filter_pkg_inner_softlink() {
    awk -F, '$15 != "NA" { print $0 }'
}

# 过滤块
filter_blocks() {
    local blocks="$1"
    local blocks_reg

    trans_list_to_regex "blocks_reg" "$blocks"
    awk -F, "\$14 ~ \"$blocks_reg\"  {print \$0}"
}

# 过滤不匹配块
filterfalse_blocks() {
    local blocks="$1"
    local blocks_reg

    trans_list_to_regex "blocks_reg" "$blocks"
    awk -F, "\$14 !~ \"$blocks_reg\"  {print \$0}"
}

# 过滤操作类型条目
filter_operate_type() {
    local operator_type="$1"
    local operator_type_reg

    trans_list_to_regex "operator_type_reg" "$operator_type"

    # ~为部分匹配，正则表达式需要匹配头尾
    awk -F, "\$2 ~ \"$operator_type_reg\"  {print \$0}"
}

# 过滤安装类型条目
filter_install_type() {
    local install_type="$1"
    local install_type_reg

    # 注意：特征串不能嵌套，否则会错误匹配
    if [ "$install_type" != "full" ] && [ "$install_type" != "debug" ]; then
        install_type_reg="(all)|($install_type)"
    else
        install_type_reg="(all)|(docker)|(devel)|(run)"
    fi

    awk -F, "\$8 ~ \"$install_type_reg\"  {print \$0}"
}

# 过滤特性参数
filter_feature_param() {
    local feature_param="$1"
    local feature_type
    local feature_exclude_all  # feature是否为排除公共all文件
    local chip chip_list  # 芯片类型
    local feature_list feature_type_list

    unpack_feature_param "feature_type" "feature_exclude_all" "chip" ${feature_param}

    if [ "$feature_type" != "all" ]; then
        feature_list="$(echo $feature_type | tr ',' ' ')"
        if [ "${feature_exclude_all}" = "y" ]; then
            feature_type_list="${feature_list}"
        else
            feature_type_list="comm ${feature_list}"
        fi
    else
        feature_type_list="all"
    fi

    if [ "$chip" != "all" ]; then
        chip_list="all $(echo $chip | tr ',' ' ')"
    else
        chip_list="all"
    fi

    # filelist中的feature为all，表示所有场景下都安装这个文件
    # 输入参数中的feature为all，表示安装所有的feature
    awk -v feature_type_list="${feature_type_list}" \
        -v chip_list="${chip_list}" '
        BEGIN {
            FS= ","
            split(feature_type_list, input_feature_type_arr, " ")
            split(chip_list, input_chip_arr, " ")
        }

        function match_feature_type(features_str) {
            if (feature_type_list == "all") {
                return 1
            }
            matched_feature_type_tmp=0
            split(features_str, features, ";")
            for(i in features)
            {
                for(j in input_feature_type_arr) {
                    if(input_feature_type_arr[j] == features[i]) {
                        matched_feature_type_tmp = 1
                        break;
                    }
                }
            }
            return matched_feature_type_tmp
        }

        function match_chip(chip_str) {
            if (chip_list == "all") {
                return 1
            }
            matched_chip_tmp=0
            split(chip_str, chips, ";")
            for(i in chips)
            {
                for(j in input_chip_arr) {
                    if(input_chip_arr[j] == chips[i]) {
                        matched_chip_tmp = 1
                        break;
                    }
                }
            }
            return matched_chip_tmp
        }

        {
            matched_feature_type = match_feature_type($10);
            if (matched_feature_type == 0) next;

            matched_chip = match_chip($16)
            if (matched_chip == 0) next

            print $0
        }'
}

# 读取fileist的第2行到最后行
tail_filelist() {
    local filelist_path="$1"
    tail -n +2 "$filelist_path"
}

# 解析filelist.csv脚本
parse_filelist() {
    local install_type="$1"
    local operate_type="$2"
    local filelist_path="$3"
    local feature_param="$4"
    local filter_type="$5"
    local blocks="$6"

    if [ ! -f "$filelist_path" ]; then
        log "ERROR" "filelist $filelist_path does not exist!"
        exit 1
    fi

    # 注意：这里的filelist需要是全局变量。其它函数（如add_filelist_blocks_info）会引用这个变量。
    filelist=$(
        parse_filelist_v2 "$install_type" "$operate_type" "$filelist_path" "$feature_param" \
            "$filter_type" "$blocks"
    )
}

# 解析filelist.csv脚本
parse_filelist_v2() {
    local install_type="$1"
    local operate_type="$2"
    local filelist_path="$3"
    local feature_param="$4"
    local filter_type="$5"
    local blocks="$6"
    local filter_cmds=""

    if [ "$operate_type" != "all" ]; then
        filter_cmds="$filter_cmds | filter_operate_type \"$operate_type\""
    fi
    if printf "%s" "$filter_type" | grep -Eq "\<filter_common_dirs\>"; then
        filter_cmds="$filter_cmds | filter_common_dirs"
    fi
    if printf "%s" "$filter_type" | grep -Eq "\<filter_by_pkg_inner_softlink\>"; then
        filter_cmds="$filter_cmds | filter_pkg_inner_softlink"
    fi
    if printf "%s" "$filter_type" | grep -Eq "\<filter_by_blocks\>|\<filterfalse_blocks\>" && [ "$blocks" != "" ]; then
        filter_cmds="$filter_cmds | filterfalse_blocks \"$blocks\""
    fi
    if printf "%s" "$filter_type" | grep -Eq "\<filter_blocks\>" && [ "$blocks" != "" ]; then
        filter_cmds="$filter_cmds | filter_blocks \"$blocks\""
    fi

    tail_filelist "$filelist_path" \
    | filter_install_type "$install_type" \
    | filter_feature_param "$feature_param" \
    | eval cat $filter_cmds \
    | parse_filelist_core
}


pack_exec_params() {
    local _outvar="$1"
    local _exec_mode="$2"

    eval "${_outvar}=\"${_exec_mode}\""
}


# 迭代filelist中的条目，执行操作
foreach_filelist_exec() {
    local filelist="$1"
    local sort_filelist="$2"
    local exec_mode="$3"
    local exec_func="$4"
    local install_path="$5"
    local ret=0 tmp_ret exec_params
    shift 5

    pack_exec_params "exec_params" "${exec_mode}"

    if [ "${sort_filelist}" = "reverse" ]; then
        # 对第二列文件路径做倒序排列，保证先删除子文件夹，再删除父文件夹
        # 不可以使用echo "${filelist}"
        # 在dash中会消耗\\$username:\\$usergroup中的一个反斜线（bash与busybox的sh无此问题）
        # 需要使用here document，防止shell的解析过程
        filelist=$(sort -k2,2 -b -r <<EOF
${filelist}
EOF
        )
    elif [ "${sort_filelist}" = "no" ]; then
        # 不排序
        :
    else
        log "ERROR" "sort_filelist param wrong! sort_filelist is ${sort_filelist}, exec_func is ${exec_func}, parse_func is ${parse_func}"
        exit 1
    fi

    if [ "${exec_mode}" = "concurrency" ] || [ "${exec_mode}" = "normal" ]; then
        # 并发模式或正常模式
        :
    else
        log "ERROR" "exec_mode param wrong! exec_mode is ${exec_mode}, exec_func is ${exec_func}, parse_func is ${parse_func}"
        exit 1
    fi

    while read line; do
        array=${line}
        __length_list "${array}" "len_array"
        if [ ${len_array} -eq 0 ]; then
            continue
        fi
        "${exec_func}" "${install_path}" "${line}" "${exec_params}" "$@"
        tmp_ret="$?" && [ ${tmp_ret} -ne 0 ] && ret="${tmp_ret}"
    done << EOF
${filelist}
EOF

    if [ "${exec_mode}" = "concurrency" ]; then
        wait
    fi

    return $ret
}

# 迭代遍历filelist中的条目，执行一些操作
foreach_filelist() {
    local filter_type="$1"
    local exec_func="$2"
    local install_type="$3"
    local install_path="$4"
    local operate_type="$5"
    local filelist_path="$6"
    local feature_param="$7"
    local sort_filelist="$8"  # 排序filelist，可选参数为[no,reverse]
    local exec_mode="$9"  # 执行模式，可选参数为[normal,concurrency]
    shift 9
    local ret blocks

    if [ "$filter_type" = "filter_by_blocks" ]; then
        get_blocks_info "blocks" "$install_path"
    fi

    foreach_filelist_v2 "$exec_func" "$install_type" "$install_path" "$operate_type" "$filelist_path" \
        "$feature_param" "$filter_type" "$blocks" "$sort_filelist" "$exec_mode" "$@"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    return 0
}

# 迭代遍历filelist中的条目，执行一些操作
foreach_filelist_v2() {
    local exec_func="$1"
    local install_type="$2"
    local install_path="$3"
    local operate_type="$4"
    local filelist_path="$5"
    local feature_param="$6"
    local filter_type="$7"
    local blocks="$8"
    local sort_filelist="$9"  # 排序filelist，可选参数为[no,reverse]
    local exec_mode="${10}"  # 执行模式，可选参数为[normal,concurrency]
    shift 10
    local filter_cmds=""
    local ret

    parse_filelist "${install_type}" "${operate_type}" "${filelist_path}" "${feature_param}" \
        "${filter_type}" "${blocks}"

    foreach_filelist_exec "${filelist}" "${sort_filelist}" "${exec_mode}" "${exec_func}" "${install_path}" "$@"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    return 0
}

# 解析stash_mod.txt
parse_stashmod() {
    local stashmod_path="$1"

    if [ ! -f "${stashmod_path}" ]; then
        log "ERROR" "stash mod ${stashmod_path} does not exist!"
        exit 1
    fi

    stashmod_list=$(awk '
        BEGIN{
            FS=":"
        }
        {
            print $1,$2
        }' "${stashmod_path}")
}

# 迭代遍历stash_mod中的条目，执行操作
foreach_stashmod() {
    local exec_func="$1"
    local install_path="$2"
    local sort_type="$3"  # 排序方式，可选参数为[no,reverse]
    local array
    local len_array
    local line
    local ret=0 tmp_ret

    parse_stashmod "${install_path}/${STASH_MOD_PATH}"

    if [ "${sort_type}" = "reverse" ]; then
        # 对第二列文件路径做倒序排列，保证先处理子文件夹，再处理父文件夹
        stashmod_list=$(echo "${stashmod_list}" | sort -k1,1 -r)
    elif [ "${sort_type}" = "no" ]; then
        # 不排序
        echo > /dev/null
    else
        log "ERROR" "sort_type param wrong! sort_type is ${sort_type}, exec_func is ${exec_func}"
        exit 1
    fi

    while read line; do
        array=${line}
        __length_list "${array}" "len_array"
        if [ ${len_array} -eq 0 ]; then
            continue
        fi
        "${exec_func}" "${install_path}" "${line}"
        tmp_ret="$?" && [ ${tmp_ret} -ne 0 ] && ret="${tmp_ret}"
    done << EOF
${stashmod_list}
EOF

    return $ret
}

# 创建目录并且设置权限
make_dir_with_permission() {
    local path="$1"
    local mod="$2"
    local username="$3"
    local usergroup="$4"
    local install_for_all="$5"
    local ret

    make_dir "${path}"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    change_mod "${path}" "${mod}" "${install_for_all}"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    change_own "${path}" "${username}:${usergroup}"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    return 0
}

# blocks转换为db.info配置
blocks_to_db_item() {
    local package="$1"
    local version_dir="$2"

    xargs printf "%s|[$package:$version_dir]\n"
}

# 折叠第2个参数
# 需要保证输入数据第1个参数已排序
fold_2() {
    local fs="$1"
    local concat="$2"
    local options=""

    if [ "$fs" != "" ]; then
        options="-F$fs"
    fi

    awk $options -v CONCAT="$concat" '
    NR == 1 {
        field1 = $1
        field2 = $2
    }
    NR != 1 {
        if (field1 == $1) {
            field2 = field2 CONCAT $2
        } else {
            print field1 FS field2
            field1 = $1
            field2 = $2
        }
    }
    END {
        if (field1) {
            print field1 FS field2
        }
    }'
}

# 折叠第3个参数，保持第2个参数
# 需要保证输入数据第1个参数已排序
fold_3_keep_2() {
    local fs="$1"
    local concat="$2"
    local options=""

    if [ "$fs" != "" ]; then
        options="-F$fs"
    fi

    awk $options -v CONCAT="$concat" '
    NR == 1 {
        field1 = $1
        field2 = $2
        field3 = $3
    }
    NR != 1 {
        if (field1 == $1) {
            field3 = field3 CONCAT $3
        } else {
            print field1 FS field2 FS field3
            field1 = $1
            field2 = $2
            field3 = $3
        }
    }
    END {
        if (field1) {
            print field1 FS field2 FS field3
        }
    }'
}

# 根据第1列排序
sort_1() {
    local sep="$1"
    local options=""
    if [ "$sep" != "" ]; then
        options="-t$sep"
    fi
    sort $options -k1,1 -s
}

# 显示3,4列有差异的条目
show_diff_3_4() {
    awk '$3 != $4 { print $0 }'
}

# 显示具有最少列数的条目
show_min_nf() {
    local min_nf="$1"
    awk -v MIN_NF="$min_nf" 'NF >= MIN_NF { print $0 }'
}

# 选取第3,2,1列
select_fields_3_2_1() {
    awk '{print $3, $2, $1}'
}

# 选取第1列
select_fields_1() {
    awk '{print $1}'
}

# 删除db.info配置
del_db_items() {
    local package="$1"
    local version_dir="$2"

    sed "s/\\[$package:$version_dir\\]//g; /|\$/d"
}

# 保留块最后一个配置
# 输出：第1列（块），第2列（包名），第3列（版本目录）
remain_db_last_item() {
    awk -F '[|:\\[\\]]+' '{ print $1, $(NF-2), $(NF-1) }'
}

# 移除空白行
remove_blank_line() {
    sed '/^$/d'
}

# 保证文件权限
ensure_permission() {
    local path="$1"
    local mod="$2"
    local username="$3"
    local usergroup="$4"
    local install_for_all="$5"

    change_mod "$path" "$mod" "$install_for_all"
    ret=$? && [ $ret -ne 0 ] && return $ret

    change_own "$path" "$username:$usergroup"
    ret=$? && [ $ret -ne 0 ] && return $ret

    return 0
}

# 创建文件
touch_file() {
    local path="$1"
    local mod="$2"
    local username="$3"
    local usergroup="$4"
    local install_for_all="$5"
    local ret

    touch "$path"
    ret=$? && [ $ret -ne 0 ] && return $ret

    ensure_permission "$path" "$mod" "$username" "$usergroup" "$install_for_all"
    ret=$? && [ $ret -ne 0 ] && return $ret

    return 0
}

# 保证文件存在
ensure_file() {
    local path="$1"
    local mod="$2"
    local username="$3"
    local usergroup="$4"
    local install_for_all="$5"
    local ret

    if [ ! -f "$path" ]; then
        touch_file "$path" "$mod" "$username" "$usergroup" "$install_for_all"
        ret=$? && [ $ret -ne 0 ] && return $ret
    fi

    return 0
}

# 写文本文件
write_text() {
    local content="$1"
    local filepath="$2"

    printf "%s\n" "$content" > "$filepath"
}

# 修改文件权限并操作
with_chmod() {
    local path="$1"
    local mod="$2"
    local origin_mod ret
    shift 2

    origin_mod="$(stat -L -c "%a" "$path")"
    chmod "$mod" "$path"
    "$@"
    ret="$?"
    chmod "$origin_mod" "$path"

    return $ret
}

# filelist中所有公共目录的块
all_common_dirs_blocks_in_filelist() {
    local install_type="$1"
    local filelist_path="$2"
    local feature_param="$3"

    parse_filelist_v2 "$install_type" "all" "$filelist_path" "$feature_param" "filter_common_dirs" "" \
        | cut -d' ' -f8 | sort | uniq
}

# 是否为版本目录
is_version_dirpath() {
    local version_dirpath="$1"
    if [ -d "$version_dirpath/share/info" ]; then
        return 0
    fi
    if [ -f "$version_dirpath/var/ascend_package_db.info" ]; then
        return 0
    fi
    return 1
}
