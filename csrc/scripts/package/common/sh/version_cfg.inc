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

# 版本状态配置文件处理
# 版本状态配置文件格式为
# # version: 1.0
# runtime_running_version=[1.80:CANN-1.80]
# compiler_running_version=[1.80:CANN-1.80]
# toolkit_running_version=[1.80:CANN-1.80]
# fwkplugin_running_version=[1.80:CANN-1.80]
# runtime_installed_version=[1.79:CANN-1.79][1.80:CANN-1.80]
# compiler_installed_version=[1.79:CANN-1.79][1.80:CANN-1.80][1.81:CANN-1.81]
# toolkit_installed_version=[1.79:CANN-1.79][1.80:CANN-1.80][1.81:CANN-1.81]
# fwkplugin_installed_version=[1.79:CANN-1.79][1.80:CANN-1.80][1.81:CANN-1.81]

# 子包会source本文件，本文件中调用的函数，都需要在文件内找到定义

# latest目录名
LATEST_DIR="latest"

# 配置版本号
_CFG_VERSION="1.0"

# 获取列表索引值
__index_list_by_cut() {
    local _list_ilbc="$1"
    local _cnt_ilbc _value_ilbc
    shift

    while true; do
        if [ $# -eq 0 ]; then
            return 0
        fi
        _cnt_ilbc="${1}" && _cnt_ilbc=$((_cnt_ilbc+1))
        _value_ilbc="$(echo "${_list_ilbc}" | cut -d' ' -f"${_cnt_ilbc}")"
        eval "${2}=\"${_value_ilbc}\""
        shift 2
    done

    return 0
}

# 获取版本状态配置路径
get_version_cfg_path() {
    local _outvar="$1"
    local _dirpath="$2"
    eval "${_outvar}=\"${_dirpath}/version.cfg\""
}

# 获取版本状态配置新路径
get_version_cfg_new_path() {
    local _outvar="$1"
    local _dirpath="$2"
    eval "${_outvar}=\"${_dirpath}/version.cfg.new\""
}

# 解析配置行
parse_cfg_line() {
    local _outvar="$1"
    local _cfg_line="$2"
    local _cfg_items_pcl

    _cfg_items_pcl=$(echo "${_cfg_line}" | awk '
        function join(items, len, sep,  result, i) {
            if (len >= 1) {
                result = items[1]
                for (i = 2; i <= len; i++) {
                    result = sprintf("%s%s%s", result, sep, items[i])
                }
                return result
            }
            return ""
        }

        /^[A-Za-z0-9_-]+_version=/ {
            split($0, line_tokens, "=")
            len_attr_tokens = split(line_tokens[1], attr_tokens, "_")
            len_package_names = 0
            for (i = 1; i < len_attr_tokens - 1; i++) {
                len_package_names++
                package_names[len_package_names] = attr_tokens[i]
            }
            package_name = join(package_names, len_package_names, "_")
            attr_type = attr_tokens[len_attr_tokens-1]
            value = line_tokens[2]

            printf("%s %s %s", package_name, attr_type, value)
        }
    ')
    eval "${_outvar}=\"${_cfg_items_pcl}\""
}

# 解析配置版本号列表
unpack_versions() {
    local _outvar="$1"
    local _value="$2"

    local _versions="$(echo "${_value}" | sed 's/\[//g')"
    _versions="$(echo "${_versions}" | sed 's/\]/ /g')"
    _versions="$(echo ${_versions})"  # strip
    eval "${_outvar}=\"${_versions}\""
}

# 打包配置版本号列表
pack_versions() {
    local _outvar="$1"
    local _versions="$2"

    local _value="$(echo "${_versions}" | sed 's/ /\]\[/g')"
    eval "${_outvar}=\"[${_value}]\""
}

# 解析版本序对
unpack_version_pair() {
    local _outvar="$1"
    local _value="$2"

    eval "${_outvar}=\"$(echo "${_value}" | tr ':' ' ')\""
}

# 打包运行版本配置
pack_running_version() {
    local _outvar="$1"
    local _package="$2"
    local _version="$3"
    local _version_dir="$4"

    local _version_str
    pack_versions "_version_str" "${_version}:${_version_dir}"
    eval "${_outvar}=\"${_package}_running_version=${_version_str}\""
}

# 打包升级版本配置
pack_upgrade_version() {
    local _outvar="$1"
    local _package="$2"
    local _version="$3"
    local _version_dir="$4"

    local _version_str
    pack_versions "_version_str" "${_version}:${_version_dir}"
    eval "${_outvar}=\"${_package}_upgrade_version=${_version_str}\""
}

# 检查配置版本
check_cfg_version() {
    local filepath="$1"
    local version

    if [ ! -f "${filepath}" ]; then
        return 0
    fi

    version=$(sed -n '1p' "${filepath}" | cut -d" " -f3)
    if [ "${version}" != "${_CFG_VERSION}" ]; then
        return 1
    fi
    return 0
}

# 输出配置版本错误
error_check_cfg_version() {
    local filepath="$1"
    log "ERROR" "check ${filepath} version failed!"
}

# 检查配置版本并输出报错日志
check_cfg_version_with_log() {
    local filepath="$1"
    local ret

    check_cfg_version "${filepath}"
    ret="$?"
    if [ ${ret} -ne 0 ]; then
        error_check_cfg_version "${filepath}"
        return ${ret}
    fi
    return 0
}

# 添加配置版本注释
add_cfg_version_comment() {
    local filepath="$1"
    sed -i "1i # version: ${_CFG_VERSION}" "${filepath}"
}

# 处理移除子包版本配置行
__process_unset_package_line() {
    local line="$1" 
    local package="$2" 
    local target_type="$3" 
    local filepath_new="$4"
    local cfg_items package_name attr_type

    parse_cfg_line "cfg_items" "${line}"
    [ "${cfg_items}" = "" ] && return

    __index_list_by_cut "${cfg_items}" 0 "package_name" 1 "attr_type"

    if [ "${attr_type}" = "${target_type}" ]; then
        [ "${package_name}" != "${package}" ] && echo "${line}" >> "${filepath_new}"
    else
        echo "${line}" >> "${filepath_new}"
    fi
}

# 处理文件更新和清理
__finalize_version_file() {
    local filepath="$1"
    local filepath_new="$2"
    local cleanup="$3"

    if [ -f "${filepath_new}" ]; then
        add_cfg_version_comment "${filepath_new}"
        mv -f "${filepath_new}" "${filepath}"
        if [ $? -ne 0 ]; then
            log "ERROR" "replace ${filepath} failed!"
            eval "${cleanup}"
            return 1
        fi
    else
        rm -f "${filepath}"
        if [ $? -ne 0 ]; then
            log "ERROR" "delete ${filepath} failed!"
            eval "${cleanup}"
            return 1
        fi
    fi

    eval "${cleanup}"
    local ret="$?" && [ $ret -ne 0 ] && return $ret
}

# 移除子包版本
unset_package_version() {
    local dirpath="$1"
    local package="$2"
    local target_type="$3"
    local filepath filepath_new line cfg_items package_name attr_type
    local cleanup="" ret mod

    get_version_cfg_path "filepath" "${dirpath}"
    get_version_cfg_new_path "filepath_new" "${dirpath}"

    # 版本状态配置文件不存在则正常退出
    [ ! -f "${filepath}" ] && return 0

    check_cfg_version_with_log "${filepath}" || return 1

    # 保存当前目录的权限
    get_file_mod "mod" "${dirpath}"

    # 添加目录写权限
    chmod u+w "${dirpath}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    # 恢复权限
    cleanup="chmod ${mod} \"${dirpath}\""

    rm -f "${filepath_new}"
    if [ $? -ne 0 ]; then
        log "ERROR" "delete ${filepath_new} failed!"
        eval "${cleanup}"
        return 1
    fi

    while read line; do
        __process_unset_package_line "${line}" "${package}" "${target_type}" "${filepath_new}"
    done < "${filepath}"

    __finalize_version_file  "${filepath}" "${filepath_new}" "${cleanup}"
    ret="$?" && [ $ret -ne 0 ] && return $ret
    return 0
}

# 获取子包版本
get_package_version_version_cfg() {
    local _outvar="$1"
    local _dirpath="$2"
    local _package="$3"
    local _target_type="$4"
    local _filepath _version_gpv

    get_version_cfg_path "_filepath" "${_dirpath}"

    if [ ! -f "${_filepath}" ]; then
        eval "${_outvar}=\"\""
        return 0
    fi

    check_cfg_version "${_filepath}"
    if [ $? -ne 0 ]; then
        eval "${_outvar}=\"\""
        return 1
    fi

    _version_gpv="$(grep "^${_package}_${_target_type}_version=" "${_filepath}" | cut -d= -f2-)"
    if [ "${_version_gpv}" != "" ]; then
        unpack_versions "_version_gpv" "${_version_gpv}"
    fi

    eval "${_outvar}=\"${_version_gpv}\""
    return 0
}

# 处理单个配置文件
__process_cfg_file() {
    local filepath="$1" 
    local package="$2" 
    local running_version="$3" 
    local filepath_new="$4"
    local cfg_items package_name attr_type line
    local setted="false"

    touch "${filepath}"
    while read line; do
        parse_cfg_line "cfg_items" "${line}"
        [ "${cfg_items}" = "" ] && continue

        __index_list_by_cut "${cfg_items}" 0 "package_name" 1 "attr_type"

        if [ "${attr_type}" = "running" ]; then
            if [ "${package_name}" = "${package}" ]; then
                if [ "${setted}" = "false" ]; then
                    echo "${running_version}" >> "${filepath_new}"
                    setted="true"
                fi
            else
                echo "${line}" >> "${filepath_new}"
            fi
        elif [ "${attr_type}" = "upgrade" ]; then
            if [ "${setted}" = "false" ]; then
                echo "${running_version}" >> "${filepath_new}"
                setted="true"
            fi
            echo "${line}" >> "${filepath_new}"
        elif [ "${attr_type}" = "installed" ]; then
            if [ "${setted}" = "false" ]; then
                echo "${running_version}" >> "${filepath_new}"
                setted="true"
            fi
            echo "${line}" >> "${filepath_new}"
        else
            echo "${line}" >> "${filepath_new}"
        fi
    done < "${filepath}"

    if [ "${setted}" = "false" ]; then
        echo "${running_version}" >> "${filepath_new}"
        setted="true"
    fi
}

# 设置运行子包版本
set_running_package_version() {
    local dirpath="$1"
    local package="$2"
    local version="$3"
    local version_dir="$4"
    local filepath filepath_new line cfg_items package_name attr_type running_version
    local setted="false" cleanup="" ret mod

    get_version_cfg_path "filepath" "${dirpath}"
    get_version_cfg_new_path "filepath_new" "${dirpath}"

    check_cfg_version_with_log "${filepath}"
    if [ $? -ne 0 ]; then
        return 1
    fi

    pack_running_version "running_version" "${package}" "${version}" "${version_dir}"

    # 保存当前目录的权限
    get_file_mod "mod" "${dirpath}"

    # 添加目录写权限
    chmod u+w "${dirpath}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    # 恢复权限
    cleanup="chmod ${mod} \"${dirpath}\""

    rm -f "${filepath_new}"
    if [ $? -ne 0 ]; then
        log "ERROR" "delete ${filepath_new} failed!"
        eval "${cleanup}"
        return 1
    fi

    __process_cfg_file "${filepath}" "${package}" "${running_version}" "${filepath_new}"
    
    add_cfg_version_comment "${filepath_new}"
    mv -f "${filepath_new}" "${filepath}"
    if [ $? -ne 0 ]; then
        log "ERROR" "replace ${filepath} failed!"
        eval "${cleanup}"
        return 1
    fi

    eval "${cleanup}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    return 0
}

# 移除子包运行版本
unset_running_package() {
    local dirpath="$1"
    local package="$2"

    unset_package_version "${dirpath}" "${package}" "running"
}

# 获取子包运行版本
get_running_package_version() {
    local _outvar="$1"
    local _dirpath="$2"
    local _package="$3"

    get_package_version_version_cfg "${_outvar}" "${_dirpath}" "${_package}" "running"
}

# 处理升级版本文件操作和清理
__finalize_upgrade_file() {
    local filepath="$1" filepath_new="$2" setted="$3" upgrade_version="$4" cleanup="$5"

    if [ "${setted}" = "false" ]; then
        echo "${upgrade_version}" >> "${filepath_new}"
        setted="true"
    fi

    add_cfg_version_comment "${filepath_new}"
    mv -f "${filepath_new}" "${filepath}"
    if [ $? -ne 0 ]; then
        log "ERROR" "replace ${filepath} failed!"
        eval "${cleanup}"
        return 1
    fi

    eval "${cleanup}"
    local ret="$?" && [ $ret -ne 0 ] && return $ret
    return 0
}

# 处理设置升级版本配置行
__process_set_upgrade_line() {
    local line="$1"
    local package="$2"
    local upgrade_version="$3"
    local filepath_new="$4"
    local -n setted_ref="$5"
    local cfg_items package_name attr_type

    parse_cfg_line "cfg_items" "${line}"
    [ "${cfg_items}" = "" ] && return

    __index_list_by_cut "${cfg_items}" 0 "package_name" 1 "attr_type"

    if [ "${attr_type}" = "upgrade" ]; then
        if [ "${package_name}" = "${package}" ]; then
            if [ "${setted_ref}" = "false" ]; then
                echo "${upgrade_version}" >> "${filepath_new}"
                setted_ref="true"
            fi
        else
            echo "${line}" >> "${filepath_new}"
        fi
    elif [ "${attr_type}" = "installed" ]; then
        if [ "${setted_ref}" = "false" ]; then
            echo "${upgrade_version}" >> "${filepath_new}"
            setted_ref="true"
        fi
        echo "${line}" >> "${filepath_new}"
    else
        echo "${line}" >> "${filepath_new}"
    fi
}

# 设置子包升级版本
set_upgrade_package_version() {
    local dirpath="$1"
    local package="$2"
    local version="$3"
    local version_dir="$4"
    local filepath filepath_new line cfg_items package_name attr_type upgrade_version
    local setted="false" cleanup="" ret mod

    get_version_cfg_path "filepath" "${dirpath}"
    get_version_cfg_new_path "filepath_new" "${dirpath}"

    check_cfg_version_with_log "${filepath}" || return 1

    pack_upgrade_version "upgrade_version" "${package}" "${version}" "${version_dir}"

    # 保存当前目录的权限
    get_file_mod "mod" "${dirpath}"

    # 添加目录写权限
    chmod u+w "${dirpath}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    # 恢复权限
    cleanup="chmod ${mod} \"${dirpath}\""

    rm -f "${filepath_new}"
    if [ $? -ne 0 ]; then
        log "ERROR" "delete ${filepath_new} failed!"
        eval "${cleanup}"
        return 1
    fi

    touch "${filepath}"
    while read line; do
        __process_set_upgrade_line "${line}" "${package}" "${upgrade_version}" "${filepath_new}" "setted"
    done < "${filepath}"

    __finalize_upgrade_file "${filepath}" "${filepath_new}" "${setted}" "${upgrade_version}" "${cleanup}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    return 0
}

# 移除子包升级版本
unset_upgrade_package() {
    local dirpath="$1"
    local package="$2"
    unset_package_version "${dirpath}" "${package}" "upgrade"
}

# 获取子包升级版本
get_upgrade_package_version() {
    local _outvar="$1"
    local _dirpath="$2"
    local _package="$3"

    get_package_version_version_cfg "${_outvar}" "${_dirpath}" "${_package}" "upgrade"
}

# 包版本是否为某状态
is_package_version_status() {
    local _outvar="$1"
    local _latest_path="$2"
    local _package="$3"
    local _version="$4"
    local _version_dir="$5"
    local _target_type="$6"
    local _pkg_status_version
    local _ret _result_ipvs=""

    eval "${_outvar}=\"\""

    get_package_version_version_cfg "_pkg_status_version" "${_latest_path}" "${_package}" "${_target_type}"
    _ret="$?" && [ $_ret -ne 0 ] && return $_ret

    if [ "${_pkg_status_version}" = "${_version}:${_version_dir}" ]; then
        _result_ipvs="true"
    else
        _result_ipvs="false"
    fi

    eval "${_outvar}=\"${_result_ipvs}\""
}

# 包版本是否running
is_package_version_running() {
    local _outvar="$1"
    local _latest_path="$2"
    local _package="$3"
    local _version="$4"
    local _version_dir="$5"

    is_package_version_status "${_outvar}" "${_latest_path}" "${_package}" "${_version}" "${_version_dir}" "running"
}

# 包版本是否upgrade
is_package_version_upgrade() {
    local _outvar="$1"
    local _latest_path="$2"
    local _package="$3"
    local _version="$4"
    local _version_dir="$5"

    is_package_version_status "${_outvar}" "${_latest_path}" "${_package}" "${_version}" "${_version_dir}" "upgrade"
}

# 添加安装子包版本
add_installed_package_version() {
    local dirpath="$1"
    local package="$2"
    local version="$3"
    local version_dir="$4"
    local filepath filepath_new line cfg_items package_name attr_type value versions matched version_str
    local setted="false"

    get_version_cfg_path "filepath" "${dirpath}"
    get_version_cfg_new_path "filepath_new" "${dirpath}"

    check_cfg_version_with_log "${filepath}"
    if [ $? -ne 0 ]; then
        return 1
    fi

    rm -f "${filepath_new}"
    if [ $? -ne 0 ]; then
        log "ERROR" "delete ${filepath_new} failed!"
        return 1
    fi

    pack_versions "version_str" "${version}:${version_dir}"

    touch "${filepath}"
    while read line; do
        parse_cfg_line "cfg_items" "${line}"
        [ "${cfg_items}" = "" ] && continue

        __index_list_by_cut "${cfg_items}" 0 "package_name" 1 "attr_type" 2 "value"

        if [ "${attr_type}" = "installed" ] && [ "${package_name}" = "${package}" ]; then
            unpack_versions "versions" "${value}"
            __item_in_list "matched" "${version}:${version_dir}" ${versions}
            if [ "${matched}" = "true" ]; then
                echo "${line}" >> "${filepath_new}"
            else
                echo "${line}${version_str}" >> "${filepath_new}"
            fi
            setted="true"
        else
            echo "${line}" >> "${filepath_new}"
        fi
    done < "${filepath}"

    if [ "${setted}" = "false" ]; then
        echo "${package}_installed_version=${version_str}" >> "${filepath_new}"
        setted="true"
    fi

    add_cfg_version_comment "${filepath_new}"
    mv -f "${filepath_new}" "${filepath}"
    if [ $? -ne 0 ]; then
        log "ERROR" "replace ${filepath} failed!"
        return 1
    fi
}

__process_install_cfg_line() {
    local package="$1"
    local version="$2"
    local version_dir="$3"
    local filepath_new="$4"
    local line cfg_items package_name attr_type value versions len_versions

    while read line; do
        parse_cfg_line "cfg_items" "${line}"
        [ "${cfg_items}" = "" ] && continue

        __index_list_by_cut "${cfg_items}" 0 "package_name" 1 "attr_type" 2 "value"

        if [ "${attr_type}" = "installed" ] && [ "${package_name}" = "${package}" ]; then
            unpack_versions "versions" "${value}"
            versions="$(__remove_item_in_list "${version}:${version_dir}" "${versions}")"
            __length_list "${versions}" "len_versions"
            if [ ${len_versions} -gt 0 ]; then
                pack_versions "value" "${versions}"
                echo "${package}_installed_version=${value}" >> "${filepath_new}"
            fi
        else
            echo "${line}" >> "${filepath_new}"
        fi
    done < "${filepath}"
}

# 删除安装子包版本
del_installed_package_version() {
    local dirpath="$1"
    local package="$2"
    local version="$3"
    local version_dir="$4"
    local filepath filepath_new line cfg_items package_name attr_type value versions len_versions

    get_version_cfg_path "filepath" "${dirpath}"
    get_version_cfg_new_path "filepath_new" "${dirpath}"

    # 版本状态配置文件不存在则正常退出
    [ ! -f "${filepath}" ] && return 0

    check_cfg_version_with_log "${filepath}" || return 1

    rm -f "${filepath_new}"
    if [ $? -ne 0 ]; then
        log "ERROR" "delete ${filepath_new} failed!"
        return 1
    fi

    __process_del_installed_line "${package}" "${version}" "${version_dir}" "${filepath_new}"

    if [ -f "${filepath_new}" ]; then
        add_cfg_version_comment "${filepath_new}"
        mv -f "${filepath_new}" "${filepath}"
        if [ $? -ne 0 ]; then
            log "ERROR" "replace ${filepath} failed!"
            return 1
        fi
    else
        rm -f "${filepath}"
        if [ $? -ne 0 ]; then
            log "ERROR" "delete ${filepath} failed!"
            return 1
        fi
    fi
    return 0
}

# 获取子包版本
get_installed_package_versions() {
    local _outvar="$1"
    local _dirpath="$2"
    local _package="$3"
    local _filepath _value _new_versions_gipv

    get_version_cfg_path "_filepath" "${_dirpath}"

    if [ ! -f "${_filepath}" ]; then
        eval "${_outvar}=\"\""
        return 0
    fi

    check_cfg_version "${_filepath}"
    if [ $? -ne 0 ]; then
        eval "${_outvar}=\"\""
        return 1
    fi

    _value="$(grep "^${_package}_installed_version=" "${_filepath}" | cut -d= -f2-)"
    unpack_versions "_new_versions_gipv" "${_value}"
    __reverse_list "_new_versions_gipv" "${_new_versions_gipv}"

    eval "${_outvar}=\"${_new_versions_gipv}\""
}

# 获取子包最后一次的安装版本
get_package_last_installed_version() {
    local _outvar="$1"
    local _install_path="$2"
    local _package="$3"
    local _versions_gpliv="" _version_gpliv="" _ret

    eval "${_outvar}=\"\""

    get_installed_package_versions "_versions_gpliv" "${_install_path}/${LATEST_DIR}" "${_package}"
    _ret="$?" && [ ${_ret} -ne 0 ] && return ${_ret}

    [ "${_versions_gpliv}" = "" ] && return 0

    __index_list_by_cut "${_versions_gpliv}" 0 "_version_gpliv"

    eval "${_outvar}=\"${_version_gpliv}\""
}


# 获取子包最后一次的安装版本目录
# 如果不存在上次安装版本，则返回空字符串
get_package_last_installed_version_dir() {
    local _outvar="$1"
    local _install_path="$2"
    local _package="$3"
    local _version_pair _version_pair_arr _version_dir_gplivd="" _ret

    eval "${_outvar}=\"\""

    get_package_last_installed_version "_version_pair" "${_install_path}" "${_package}"
    _ret="$?" && [ ${_ret} -ne 0 ] && return ${_ret}

    unpack_version_pair "_version_pair_arr" "${_version_pair}"
    __index_list_by_cut "${_version_pair_arr}" 1 "_version_dir_gplivd"

    eval "${_outvar}=\"${_version_dir_gplivd}\""
}

# 获取子包最后一次安装的install.info路径
# 如果不存在上次安装版本，则返回空字符串
get_package_last_install_info() {
    local _outvar="$1"
    local _install_path="$2"
    local _package="$3"
    local _version_dir="" _install_info="" _ret

    eval "${_outvar}=\"\""

    get_package_last_installed_version_dir "_version_dir" "${_install_path}" "${_package}"
    _ret="$?" && [ ${_ret} -ne 0 ] && return ${_ret}

    [ "${_version_dir}" = "" ] && return 0

    get_package_install_info "_install_info" "${_install_path}" "${_version_dir}" "${_package}"

    eval "${_outvar}=\"${_install_info}\""
}

get_package_version_dir() {
    local _outvar="$1"
    local _install_path="$2"
    local _package="$3"
    local _target_type="$4"
    local _version_pair _version_pair_arr _version_dir_gprvd="" _ret

    eval "${_outvar}=\"\""

    get_package_version_version_cfg "_version_pair" "${_install_path}/${LATEST_DIR}" "${_package}" "${_target_type}"
    _ret="$?" && [ ${_ret} -ne 0 ] && return ${_ret}

    if [ "${_version_pair}" = "" ] && [ "${_target_type}" = "upgrade" ]; then
        get_package_version_version_cfg "_version_pair" "${_install_path}/${LATEST_DIR}" "${_package}" "running"
        _ret="$?" && [ ${_ret} -ne 0 ] && return ${_ret}
    fi

    [ "${_version_pair}" = "" ] && return 0

    unpack_version_pair "_version_pair_arr" "${_version_pair}"
    __index_list_by_cut "${_version_pair_arr}" 1 "_version_dir_gprvd"

    eval "${_outvar}=\"${_version_dir_gprvd}\""
}

get_package_install_message() {
    local _outvar="$1"
    local _install_path="$2"
    local _package="$3"
    local _target_type="$4"
    local _version_dir_gpim="" _install_info_gpim="" _ret

    eval "${_outvar}=\"\""

    get_package_version_dir "_version_dir_gpim" "${_install_path}" "${_package}" "${_target_type}"
    _ret="$?" && [ ${_ret} -ne 0 ] && return ${_ret}

    [ "${_version_dir_gpim}" = "" ] && return 0

    get_package_install_info "_install_info_gpim" "${_install_path}" "${_version_dir_gpim}" "${_package}"

    eval "${_outvar}=\"${_install_info_gpim}\""
}

# 获取子包的运行版本目录
# 如果不存在运行版本，则返回空字符串
get_package_running_version_dir() {
    local _outvar="$1"
    local _install_path="$2"
    local _package="$3"

    get_package_version_dir "${_outvar}" "${_install_path}" "${_package}" "running"
}

# 获取子包运行版本的install.info路径
# 如果不存在运行版本，则返回空字符串
get_package_running_install_info() {
    local _outvar="$1"
    local _install_path="$2"
    local _package="$3"

    get_package_install_message "${_outvar}" "${_install_path}" "${_package}" "running"
}

# 获取子包的升级版本目录
# 如果不存在升级版本，则返回运行版本目录
# 如果也不存在运行版本，则返回空字符串
get_package_upgrade_version_dir() {
    local _outvar="$1"
    local _install_path="$2"
    local _package="$3"

    get_package_version_dir "${_outvar}" "${_install_path}" "${_package}" "upgrade"
}

# 获取子包升级版本的install.info路径
# 如果不存在升级版本，则返回运行版本的install.info路径
# 如果也不存在运行版本，则返回空字符串
get_package_upgrade_install_info() {
    local _outvar="$1"
    local _install_path="$2"
    local _package="$3"

    get_package_install_message "${_outvar}" "${_install_path}" "${_package}" "upgrade"
}

# 获取所有运行包
get_running_packages() {
    local _outvar="$1"
    local _dirpath="$2"
    local _filepath _line _cfg_items _package_name_grp _attr_type_grp _packages=""

    eval "${_outvar}=\"\""

    get_version_cfg_path "_filepath" "${_dirpath}"

    if [ ! -f "${_filepath}" ]; then
        return 0
    fi

    check_cfg_version "${_filepath}"
    if [ $? -ne 0 ]; then
        return 1
    fi

    while read _line; do
        parse_cfg_line "_cfg_items" "${_line}"
        [ "${_cfg_items}" = "" ] && continue

        __index_list_by_cut "${_cfg_items}" 0 "_package_name_grp" 1 "_attr_type_grp"

        if [ "${_attr_type_grp}" = "running" ]; then
            if [ "${_packages}" = "" ]; then
                _packages="${_package_name_grp}"
            else
                _packages="${_packages} ${_package_name_grp}"
            fi
        fi
    done < "${_filepath}"

    eval "${_outvar}=\"${_packages}\""
}

# 根据版本目录获取版本号
get_version_by_version_dir() {
    local _outvar="$1"
    local _dirpath="$2"
    local _version_dir="$3"
    local _filepath _line _cfg_items _attr_type_gvbvd _value_gvbvd
    local _version_pairs _version_pair _version_pair_arr _version_gvbvd _version_dir_gvbvd

    eval "${_outvar}=\"\""

    get_version_cfg_path "_filepath" "${_dirpath}"
    if [ ! -f "${_filepath}" ]; then
        return 1
    fi

    check_cfg_version "${_filepath}"
    if [ $? -ne 0 ]; then
        return 1
    fi

    while read _line; do
        parse_cfg_line "_cfg_items" "${_line}"
        [ "${_cfg_items}" = "" ] && continue

        __index_list_by_cut "${_cfg_items}" 1 "_attr_type_gvbvd" 2 "_value_gvbvd"

        if [ "${_attr_type_gvbvd}" = "installed" ]; then
            unpack_versions "_version_pairs" "${_value_gvbvd}"
            for _version_pair in ${_version_pairs}; do
                unpack_version_pair "_version_pair_arr" "${_version_pair}"
                __index_list_by_cut "${_version_pair_arr}" 0 "_version_gvbvd" 1 "_version_dir_gvbvd"
                if [ "${_version_dir_gvbvd}" = "${_version_dir}" ]; then
                    eval "${_outvar}=\"${_version_gvbvd}\""
                    return 0
                fi
            done
        fi
    done < "${_filepath}"

    return 1
}

# 根据版本号与版本目录获取所有安装包
get_installed_packages_by_version_version_dir() {
    local _outvar="$1"
    local _dirpath="$2"
    local _version="$3"
    local _version_dir="$4"
    local _filepath _line _cfg_items
    local _package_gipbvvd _attr_type_gipbvvd _value_gipbvvd _version_pairs _in_list
    local _packages_gipbvvd=""

    eval "${_outvar}=\"\""

    get_version_cfg_path "_filepath" "${_dirpath}"
    if [ ! -f "${_filepath}" ]; then
        return 1
    fi

    check_cfg_version "${_filepath}"
    if [ $? -ne 0 ]; then
        return 1
    fi

    while read _line; do
        parse_cfg_line "_cfg_items" "${_line}"
        [ "${_cfg_items}" = "" ] && continue

        __index_list_by_cut "${_cfg_items}" 0 "_package_gipbvvd" 1 "_attr_type_gipbvvd" 2 "_value_gipbvvd"

        if [ "${_attr_type_gipbvvd}" = "installed" ]; then
            unpack_versions "_version_pairs" "${_value_gipbvvd}"
            __item_in_list "_in_list" "${_version}:${_version_dir}" ${_version_pairs}
            if [ "${_in_list}" = "true" ]; then
                if [ "${_packages_gipbvvd}" = "" ]; then
                    _packages_gipbvvd="${_package_gipbvvd}"
                else
                    _packages_gipbvvd="${_packages_gipbvvd} ${_package_gipbvvd}"
                fi
            fi
        fi
    done < "${_filepath}"

    eval "${_outvar}=\"${_packages_gipbvvd}\""
}

# 获取所有安装包
get_installed_packages() {
    local _outvar="$1"
    local _dirpath="$2"
    local _filepath _line _cfg_items _package_name_gip _attr_type_gip _packages=""

    eval "${_outvar}=\"\""

    get_version_cfg_path "_filepath" "${_dirpath}"

    if [ ! -f "${_filepath}" ]; then
        return 0
    fi

    check_cfg_version "${_filepath}"
    if [ $? -ne 0 ]; then
        return 1
    fi

    while read _line; do
        parse_cfg_line "_cfg_items" "${_line}"
        [ "${_cfg_items}" = "" ] && continue

        __index_list_by_cut "${_cfg_items}" 0 "_package_name_gip" 1 "_attr_type_gip"

        if [ "${_attr_type_gip}" = "installed" ]; then
            if [ "${_packages}" = "" ]; then
                _packages="${_package_name_gip}"
            else
                _packages="${_packages} ${_package_name_gip}"
            fi
        fi
    done < "${_filepath}"

    eval "${_outvar}=\"${_packages}\""
}

# version.cfg文件是否存在
# 返回值0为真，1为假
version_cfg_exists() {
    local dirpath="$1"
    local filepath

    get_version_cfg_path "filepath" "$dirpath"
    if [ -f "$filepath" ]; then
        return 0
    fi
    return 1
}

# 显示version.cfg文件
show_version_cfg() {
    local dirpath="$1"
    local filepath

    get_version_cfg_path "filepath" "$dirpath"
    if [ -f "$filepath" ]; then
        cat "$filepath"
    fi
}
