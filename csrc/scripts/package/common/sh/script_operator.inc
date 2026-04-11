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

# 公共脚本操作库
ENV_SHELL_TYPES="bash csh fish"

# 检查参数。
__check_param() {
    local name="$1"
    local value

    value="$(eval echo \${${name}})"

    if [ "${value}" = "" ]; then
        log "ERROR" "need set ${name} parameter!"
        exit 1
    fi
}

# 设置用户家目录路径
__set_userpath() {
    local username="$1"
    local docker_root="$2"

    if [ -z "${docker_root}" ] || [ "${docker_root}" = "/" ]; then
        userpath="$(eval echo "~${username}")"
    else
        if [ "${username}" = "root" ]; then
            userpath="${docker_root}/root"
        else
            userpath="${docker_root}/home/${username}"
        fi
    fi
}

# 获取setenv.[shell]路径正则
get_setenv_path_regex() {
    local _outvar="$1"
    local _package="$2"
    local _shell_type="$3"
    # setenv传入路径正则规则
    # 必须以/开头（绝对路径）
    # 必须以/${package}/bin/setenv.${shell_type}结尾
    local _path_regex="\/\(.\+\/\)\?${_package}\/bin\/setenv.${_shell_type}"

    eval "${_outvar}=\"${_path_regex}\""
}

# 创建setenv文件
__create_setenv_file() {
    local file="$1"
    local shell_type="$2"
    local add_multi_version_param="$3"
    local install_path="$4"
    local ret

    if [ ! -f "${file}" ]; then
        echo "#!/usr/bin/env ${shell_type}" > ${file}
        ret="$?" && [ $ret -ne 0 ] && return $ret

        if [ "${add_multi_version_param}" = "true" ]; then
            if [ "${shell_type}" = "bash" ]; then
                echo "export ASCEND_HOME_PATH=\"${install_path}\"" >> "${file}"
                ret="$?" && [ $ret -ne 0 ] && return $ret
            elif [ "${shell_type}" = "fish" ]; then
                echo "set -gx ASCEND_HOME_PATH \"${install_path}\"" >> "${file}"
                ret="$?" && [ $ret -ne 0 ] && return $ret
            elif [ "${shell_type}" = "csh" ]; then
                echo "setenv ASCEND_HOME_PATH \"${install_path}\"" >> "${file}"
                ret="$?" && [ $ret -ne 0 ] && return $ret
            fi
        fi
    fi
    return 0
}

# 是否存在ASCEND_HOME_PATH环境变量
has_ascend_home_path_env() {
    local file="$1"
    [ ! -f "${file}" ] && return 0
    grep "\<ASCEND_HOME_PATH\>" "${file}" > /dev/null 2>&1
}

# 添加ASCEND_HOME_PATH环境变量
add_ascend_home_path_env() {
    local file="$1"
    local shell_type="$2"
    local install_path="$3"

    if [ "${shell_type}" = "bash" ]; then
        sed -i "1aexport ASCEND_HOME_PATH=\"${install_path}\"" "${file}"
    elif [ "${shell_type}" = "fish" ]; then
        sed -i "1aset -gx ASCEND_HOME_PATH \"${install_path}\"" "${file}"
    elif [ "${shell_type}" = "csh" ]; then
        sed -i "1asetenv ASCEND_HOME_PATH \"${install_path}\"" "${file}"
    fi
}

# 删除rcfile中的source <path>
__remove_path_regex() {
    local path_regex="$1"
    local rcfile="$2"

    if [ -f "${rcfile}" ]; then
        sed -i "/source ${path_regex}\( \"multi_version\"\)\?$/d" "${rcfile}"
        if [ $? -ne 0 ]; then
            log "ERROR" "remove ${rcfile} source command failed!"
            exit 1
        fi
    fi
}

# 获取setenv.${shell_type}文件路径
get_setenv_filepath() {
    local _outvar="$1"
    local _install_path="$2"
    local _shell_type="$3"

    eval "${_outvar}=\"${_install_path}/bin/setenv.${_shell_type}\""
}

# setenv.[shell]总脚本中添加一条语句
add_setenv_cmd() {
    local install_path="$1"
    local setenv_filepath="$2"
    local package="$3"
    local shell_type="$4"
    local username="$5"
    local usergroup="$6"
    local add_multi_version_param="$7"
    local path_regex config_path multi_version_param ret

    get_setenv_path_regex "path_regex" "${package}" "${shell_type}"

    get_setenv_filepath "config_path" "$install_path" "$shell_type"
    __create_setenv_file "${config_path}" "${shell_type}" "${add_multi_version_param}" "${install_path}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    change_own "${config_path}" "${username}:${usergroup}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    change_mod "${config_path}" "${SETENV_WRITEABLE_MOD}" ""
    ret="$?" && [ $ret -ne 0 ] && return $ret

    __remove_path_regex "${path_regex}" "${config_path}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    if [ "${add_multi_version_param}" = "true" ]; then
        multi_version_param=" \"multi_version\""
    fi

    if [ "${shell_type}" = "bash" ] || [ "${shell_type}" = "fish" ]; then
        echo "source ${setenv_filepath}${multi_version_param}" >> "${config_path}"
        ret="$?" && [ $ret -ne 0 ] && return $ret
    elif [ "${shell_type}" = "csh" ]; then
        echo "set argv=(\"${setenv_filepath}\"${multi_version_param}); source ${setenv_filepath}" >> "${config_path}"
        ret="$?" && [ $ret -ne 0 ] && return $ret
    fi
    change_mod "${config_path}" "${SETENV_MOD}" "${INSTALL_FOR_ALL}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    return 0
}

# 修改路径的own
__chown_path() {
    local path="$1"
    local username="$2"
    local usergroup="$3"

    if [ "${username}" != "" ] && [ "${usergroup}" != "" ]; then
        chown -h "${username}:${usergroup}" "${path}"
        if [ $? -ne 0 ]; then
            log "ERROR" "${path} chown failed!"
            exit 1
        fi
    fi

    return 0
}

# 创建配置所在目录
__create_config_dir() {
    local dir="$1"
    local username="$2"
    local usergroup="$3"
    local ret

    if [ ! -d "${dir}" ]; then
        mkdir -p "${dir}"
        ret="$?" && [ $ret -ne 0 ] && return $ret

        __chown_path "${dir}" "${username}" "${usergroup}"
        ret="$?" && [ $ret -ne 0 ] && return $ret
    fi

    return 0
}

# 创建配置文件
__create_config_file() {
    local file="$1"
    local username="$2"
    local usergroup="$3"
    local ret

    if [ ! -f "${file}" ]; then
        touch "${file}"
        ret="$?" && [ $ret -ne 0 ] && return $ret

        __chown_path "${file}" "${username}" "${usergroup}"
        ret="$?" && [ $ret -ne 0 ] && return $ret
    fi

    return 0
}

# 为bash添加rc配置
add_bash_env_rc() {
    local userpath="$1"
    local username="$2"
    local usergroup="$3"
    local path_regex="$4"
    local setenv_filepath="$5"
    local config_path="${userpath}/.bashrc"
    local ret

    __create_config_dir "${userpath}" "${username}" "${usergroup}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    __create_config_file "${config_path}" "${username}" "${usergroup}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    __remove_path_regex "${path_regex}" "${config_path}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    echo "source ${setenv_filepath}" >> "${config_path}"
    ret="$?" && [ $ret -ne 0 ] && return $ret
}

# 为fish添加rc配置
add_fish_env_rc() {
    local userpath="$1"
    local username="$2"
    local usergroup="$3"
    local path_regex="$4"
    local setenv_filepath="$5"
    local config_path="${userpath}/.config/fish/config.fish"
    local ret

    __create_config_dir "${userpath}" "${username}" "${usergroup}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    __create_config_dir "${userpath}/.config" "${username}" "${usergroup}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    __create_config_dir "${userpath}/.config/fish" "${username}" "${usergroup}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    __create_config_file "${config_path}" "${username}" "${usergroup}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    __remove_path_regex "${path_regex}" "${config_path}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    echo "source ${setenv_filepath}" >> "${config_path}"
    ret="$?" && [ $ret -ne 0 ] && return $ret
}

# 为csh添加rc配置
add_csh_env_rc() {
    local userpath="$1"
    local username="$2"
    local usergroup="$3"
    local path_regex="$4"
    local setenv_filepath="$5"
    local config_path="${userpath}/.cshrc"
    local ret

    __create_config_dir "${userpath}" "${username}" "${usergroup}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    __create_config_file "${config_path}" "${username}" "${usergroup}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    __remove_path_regex "${path_regex}" "${config_path}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    echo "set argv=(\"${setenv_filepath}\"); source ${setenv_filepath}" >> "${config_path}"
    ret="$?" && [ $ret -ne 0 ] && return $ret
}

# 添加source setenv.[shell]脚本到rc文件中
add_env_rc() {
    local install_path="$1"
    local setenv_filepath="$2"
    local package="$3"
    local shell_type="$4"
    local setenv="$5"
    local username="$6"
    local usergroup="$7"
    local add_multi_version_param="$8"
    local docker_root="$9"
    local path_suffix="${package}/bin/setenv.${shell_type}"
    local path_regex
    local matched
    local userpath
    local config_path
    local ret

    __check_param "package"
    __check_param "username"
    __check_param "usergroup"

    echo "${shell_type}" | grep -E "^(bash|fish|csh)$" > /dev/null
    if [ $? -ne 0 ]; then
        log "ERROR" "shell type ${shell_type} not support!"
        exit 1
    fi

    get_setenv_path_regex "path_regex" "${package}" "${shell_type}"

    matched="$(echo "${setenv_filepath}" | sed -n "/^${path_regex}$/p")"
    if [ -z "${matched}" ]; then
        log "ERROR" "setenv filepath is illegal, should endswith ${path_suffix}"
        exit 1
    fi

    if [ "${setenv}" = "y" ]; then
        __set_userpath "${username}" "${docker_root}"
        case "${shell_type}" in
            bash) add_bash_env_rc "${userpath}" "${username}" "${usergroup}" "${path_regex}" "${setenv_filepath}" ;;
            fish) add_fish_env_rc "${userpath}" "${username}" "${usergroup}" "${path_regex}" "${setenv_filepath}" ;;
            csh)  add_csh_env_rc "${userpath}" "${username}" "${usergroup}" "${path_regex}" "${setenv_filepath}" ;;
        esac
        ret="$?" && [ $ret -ne 0 ] && return $ret
    fi

    # 如果bin目录不存在，则不处理
    [ ! -d "${install_path}/bin" ] && return 0

    add_setenv_cmd "${install_path}" "${setenv_filepath}" "${package}" "${shell_type}" "${username}" "${usergroup}" "${add_multi_version_param}"
}

# 删除setenv文件，如果已经没有setenv内容
__remove_setenv_file_if_no_content() {
    local file="$1"
    local shell_type="$2"
    local num

    if [ ! -f "${file}" ]; then
        return 0
    fi

    num=$(grep "setenv.${shell_type}" ${file} | wc -l)
    if [ ${num} -eq 0 ]; then
        rm -f "${file}" > /dev/null 2>&1
        if [ $? -ne 0 ]; then
            log "WARNING" "Delete file:${file} failed, please delete it by yourself."
        fi
    fi
}

# 从rc文件中删除source setenv.[shell]
del_env_rc() {
    local install_path="$1"
    local setenv_filepath="$2"
    local shell_type="$3"
    local username="$4"
    local docker_root="$5"
    local path_regex
    local userpath
    local config_path
    local oldmod

    __check_param "username"

    echo "${shell_type}" | grep -E "^(bash|fish|csh)$" > /dev/null
    if [ $? -ne 0 ]; then
        log "ERROR" "shell type ${shell_type} not support!"
        exit 1
    fi

    __set_userpath "${username}" "${docker_root}"

    # 将路径中的/转换为\/
    path_to_regex "path_regex" "${setenv_filepath}"

    if [ "${shell_type}" = "bash" ]; then
        __remove_path_regex "${path_regex}" "${userpath}/.bashrc"
        ret="$?" && [ $ret -ne 0 ] && return $ret
    elif [ "${shell_type}" = "fish" ]; then
        __remove_path_regex "${path_regex}" "${userpath}/.config/fish/config.fish"
        ret="$?" && [ $ret -ne 0 ] && return $ret
    elif [ "${shell_type}" = "csh" ]; then
        __remove_path_regex "${path_regex}" "${userpath}/.cshrc"
        ret="$?" && [ $ret -ne 0 ] && return $ret
    fi

    get_setenv_filepath "config_path" "$install_path" "$shell_type"

    if [ ! -f "${config_path}" ]; then
        # 如果文件不存在，则不处理
        return 0
    fi

    get_file_mod "oldmod" "${config_path}"
    change_mod "${config_path}" "${SETENV_WRITEABLE_MOD}" ""
    ret="$?" && [ $ret -ne 0 ] && return $ret

    __remove_path_regex "${path_regex}" "${config_path}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    change_mod "${config_path}" "${oldmod}" ""
    ret="$?" && [ $ret -ne 0 ] && return $ret

    __remove_setenv_file_if_no_content "${config_path}" "${shell_type}"
}

# 移除路径中的docker_root
remove_path_docker_root() {
    local _outvar="$1"
    local _path="$2"
    local _docker_root="$3"
    local _path_rpdr="$(echo "${_path}" | sed "s/^.\{${#_docker_root}\}//")"

    eval "${_outvar}=\"${_path_rpdr}\""
}

# 获取脚本路径与脚本真实路径（docker_root）
get_shell_path_and_shell_path_real() {
    local _outvar="$1"
    local _install_path="$2"
    local _package="$3"
    local _shell_filename="$4"
    local _docker_root="$5"
    local _package_dirpath _shell_path _shell_path_real

    get_package_dirpath "_package_dirpath" "${_package}"
    _shell_path_real="${_install_path}/${_package_dirpath}/bin/${_shell_filename}"

    if [ "${_docker_root}" != "" ]; then
        # 移除脚本路径中的docker_root前缀
        remove_path_docker_root "_shell_path" "${_shell_path_real}" "${_docker_root}"
    else
        _shell_path="${_shell_path_real}"
    fi

    eval "${_outvar}=\"${_shell_path} ${_shell_path_real}\""
}


# 设置环境变量
add_setenv() {
    return 0
}

# 删除环境变量
del_setenv() {
    return 0
}
