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

# 多版本函数库
# 创建版本目录
create_version_dir() {
    local install_path="$1"
    local version_dir="$2"
    local username="$3"
    local usergroup="$4"
    local install_for_all="$5"
    local ret

    if [ ! -d "${install_path}/${version_dir}" ]; then
        make_dir "${install_path}/${version_dir}"
        ret="$?" && [ $ret -ne 0 ] && return $ret
    fi

    # 约束版本号目录权限
    change_mod "${install_path}/${version_dir}" "750" "${install_for_all}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    change_own "${install_path}/${version_dir}" "${username}:${usergroup}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    return 0
}

# 检查版本包安装结果
check_version_install() {
    local install_path="$1"
    local version_dir="$2"
    local package="$3"
    local version_info_path

    get_package_version_info "version_info_path" "$install_path" "$version_dir" "$package"

    if [ ! -f "$version_info_path" ]; then
        comm_log "ERROR" "$version_info_path doesn't exist in check version install!"
        return 1
    fi
    return 0
}

# 安装latest管理器
install_latest_manager() {
    local var_path="$1"
    shift 1
    sh latest_manager/install.sh --install-path="$var_path" "$@"
}

# 卸载latest管理器
uninstall_latest_manager_for_upgrade() {
    local var_path="$1"
    ${var_path}/manager/uninstall.sh --upgrade
}

# 升级latest管理器
upgrade_latest_manager() {
    local var_path="$1"
    local latest_version_info latest_manager_version local_manager_version

    latest_version_info="$var_path/manager/version.info"
    if [ -f "$latest_version_info" ]; then
        get_version latest_manager_version "$latest_version_info"
        get_version local_manager_version "latest_manager/version.info"

        set_default "latest_manager_version" "$latest_manager_version" "0"
        set_default "local_manager_version" "$local_manager_version" "0"

        if [ "$latest_manager_version" -lt "$local_manager_version" ]; then
            uninstall_latest_manager_for_upgrade "$var_path"
            install_latest_manager "$var_path" "--upgrade"
        fi
    else
        install_latest_manager "$var_path"
    fi
}

get_install_for_all_param() {
    local _outvar="$1"
    local _install_for_all="$2"
    local _result=""

    if [ "$_install_for_all" = "y" ]; then
        _result="--install-for-all"
    fi

    eval "${_outvar}=\"${_result}\""
}

get_docker_root_param() {
    local _outvar="$1"
    local _docker_root="$2"
    local _result=""

    if [ "$_docker_root" != "" ]; then
        _result="--docker-root \\\"$_docker_root\\\""
    fi

    eval "${_outvar}=\"${_result}\""
}

# 通知latest管理器
notify_latest_manager() {
    local var_path="$1"
    local package="$2"
    local version="$3"
    local version_dir="$4"
    local install_for_all="$5"
    local docker_root="$6"
    local ext_params="$7"
    local operation="$8"
    local ret package_dir install_for_all_param docker_root_param

    if [ ! -f "$var_path/manager.sh" ]; then
        return 1
    fi

    if [ "${USE_SHARE_INFO}" == "y" ]; then
        ext_params="$ext_params --use-share-info"
    fi

    get_package_dir package_dir "$package"
    get_install_for_all_param "install_for_all_param" "$install_for_all"
    get_docker_root_param "docker_root_param" "$docker_root"

    eval "\"$var_path/manager.sh\"" --version "\"$version\"" --version-dir "\"$version_dir\"" \
        --package "\"$package\"" --package-dir "\"$package_dir\"" \
        "$install_for_all_param" "$docker_root_param" "$ext_params" "$operation"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    return 0
}

# 通知latest管理器安装完成
notify_latest_manager_installed() {
    local local_manager_version
    local ext_params

    get_version local_manager_version "latest_manager/version.info"
    set_default "local_manager_version" "$local_manager_version" "0"

    if [ "$INCREMENT" = "y" ]; then
        ext_params="--serial $local_manager_version --increment"
    else
        ext_params="--serial $local_manager_version"
    fi

    notify_latest_manager "$@" "$ext_params" "package_installed"
}

# 通知latest管理器创建软链
# 老版本全部卸载后，latest回滚到新版本时，会用老版本的install_common_parser.sh，
# 调用新版本的--create-package-latest-softlink，会走到这个流程。
notify_latest_manager_create_softlink() {
    notify_latest_manager "$@" "" "package_create_softlink"
}

# 通知latest管理器删除软链
notify_latest_manager_remove_softlink() {
    notify_latest_manager "$@" "" "package_remove_softlink"
}

# 通知latest管理器准备卸载
notify_latest_manager_pre_uninstall() {
    notify_latest_manager "$@" "" "package_pre_uninstall"
}

# 通知latest管理器卸载完成
notify_latest_manager_uninstalled() {
    local is_recreate_softlink="$1"
    local recreate_softlink=""
    local ret
    shift 1

    if [ "$is_recreate_softlink" = "y" ]; then
        recreate_softlink="--recreate-softlink"
    fi

    notify_latest_manager "$@" "$recreate_softlink" "package_uninstalled"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    return 0
}

# 多版本安装流程
multi_version_install() {
    local install_type="$1"
    local install_path="$2"
    local filelist_path="$3"
    local package="$4"
    local feature_param="$5"
    local version="$6"
    local version_dir="$7"
    local username="$8"
    local usergroup="$9"
    local setenv="${10}"
    local is_upgrade="${11}"
    local docker_root="${12}"
    local custom_options="${13}"
    local install_for_all="${14}"
    local ret total_ret="0" pkg_running_version version_pair_arr last_version last_version_dir

    create_version_dir "${install_path}" "${version_dir}" "${username}" "${usergroup}" "${install_for_all}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    version_install "${install_type}" "${install_path}" "${filelist_path}" "${package}" "${feature_param}" \
        "${version_dir}" "${username}" "${usergroup}" "${setenv}" "${is_upgrade}" "${docker_root}" "${custom_options}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    check_version_install "$install_path" "$version_dir" "$package"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    return ${total_ret}
}

# 删除latest下空目录
del_empty_dirs_in_latest() {
    local install_type="$1"
    local install_path="$2"
    local latest_dir="$3"
    local filelist_path="$4"
    local feature_param="$5"
    local ret

    create_stash_mod "${install_path}/${latest_dir}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    foreach_filelist "filter_common_dirs" "reset_mod_dirs_with_stash_mod" "${install_type}" "${install_path}/${latest_dir}" "mkdir" \
        "${filelist_path}" "${feature_param}" "no" "normal"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    foreach_filelist "filter_common_dirs" "remove_install_dirs" "${install_type}" "${install_path}/${latest_dir}" "mkdir" \
        "${filelist_path}" "${feature_param}" "reverse" "normal"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    foreach_stashmod "restore_stash_mod" "${install_path}/${latest_dir}" "reverse"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    remove_stash_mod "${install_path}/${latest_dir}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    return 0
}

# 多版本卸载流程
multi_version_uninstall() {
    local install_type="$1"
    local install_path="$2"
    local filelist_path="$3"
    local package="$4"
    local feature_param="$5"
    local version="$6"
    local version_dir="$7"
    local username="$8"
    local usergroup="$9"
    local docker_root="${10}"
    local custom_options="${11}"
    local is_recreate_softlink="${12}"
    local tmp_root tmp_filelist_path
    local ret total_ret=0 install_path_full="" is_running="" is_upgrade
    local running_packages is_final_running="false"

    check_param_not_empty "usergroup" "need set usergroup parameter in multi version uninstall!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    get_tmp_root "tmp_root"
    tmp_filelist_path=$(mktemp "$tmp_root/filelist_XXXXXX" || exit 1)
    cp -f "${filelist_path}" "${tmp_filelist_path}"

    if [ $? -ne 0 ]; then
        log "ERROR" "cp -f ${filelist_path} ${tmp_filelist_path} failed!"
        exit 1
    fi

    del_tmp_filelist="rm -f \"${tmp_filelist_path}\""

    version_uninstall "${install_type}" "${install_path}" "${filelist_path}" "${package}" "${feature_param}" \
        "${version_dir}" "${username}" "${docker_root}" "${custom_options}"
    if [ $? -ne 0 ]; then
        eval "${del_tmp_filelist}"
        return 1
    fi

    # 删除临时文件tmp_filelist_path
    eval "${del_tmp_filelist}"

    # 删除版本空目录
    is_dir_empty "${install_path}/${version_dir}"
    if [ $? -eq 0 ]; then
        remove_dir_icp "${install_path}/${version_dir}"
        ret="$?" && [ $ret -ne 0 ] && total_ret="1"
    fi

    return ${total_ret}
}