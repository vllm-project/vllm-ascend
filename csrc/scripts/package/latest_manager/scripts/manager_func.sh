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

USERNAME="$(id -un)"
USERGROUP="$(id -gn)"

CREATE_VERSION_SOFTLINK="create_version_softlink.sh"
REMOVE_LATEST_SOFTLINK="remove_latest_softlink.sh"

MANAGER_INFO="$VAR_PATH/manager.info"

# 获取manager引用计数
get_manager_refs() {
    local _outvar="$1"
    local _var_path="$2"
    local _result

    if [ -f "$_var_path/manager_refs" ]; then
        _result="$(cat "$_var_path/manager_refs")"
        if [ "$_result" = "" ]; then
            _result="0"
        fi
    else
        _result="0"
    fi
    eval "${_outvar}=\"${_result}\""
}

# manager引用计数是否存在
# 返回值0为真，1为假
manager_refs_exists() {
    local var_path="$1"

    if [ -f "$var_path/manager_refs" ]; then
        return 0
    fi
    return 1
}

# 删除manager引用计数
remove_manager_refs() {
    local var_path="$1"
    rm -f "$var_path/manager_refs"
}

# 增加manager引用计数
inc_manager_refs() {
    local var_path="$1"
    if [ -f "$var_path/manager_refs" ]; then
        xargs expr 1 + < "$var_path/manager_refs" | (read -r arg; echo "$arg" > "$var_path/manager_refs")
    else
        echo "1" > "$var_path/manager_refs"
    fi
}

# 减少manager引用计数
dec_manager_refs() {
    local var_path="$1"
    local cnt

    if [ -f "$var_path/manager_refs" ]; then
        xargs expr -1 + < "$var_path/manager_refs" | (read -r arg; echo "$arg" > "$var_path/manager_refs")

        get_manager_refs cnt "$var_path"
        if [ "$cnt" -le 0 ]; then
            rm -f "$var_path/manager_refs"
        fi
    fi
}

# 获取是否需要install_for_all
get_install_for_all() {
    local _outvar="$1"
    local _result

    if [ ! -f "$MANAGER_INFO" ]; then
        eval "${_outvar}=\"n\""
        return 0
    fi

    _result="$(grep "^install_for_all=" "$MANAGER_INFO" | cut -d= -f2-)"
    if [ "$_result" = "" ]; then
        eval "${_outvar}=\"n\""
        return 0
    fi

    eval "${_outvar}=\"${_result}\""
}

# 过滤配置文件参数
filter_info_param() {
    local name="$1"
    local value="$2"

    awk -v name="$name" -v value="$value" -F= '
        BEGIN {
            MATCHED = 0
            OFS = "="
        }
        $1 == name {
            MATCHED = 1
            print $1, value
        }
        $1 != name {
            print $0
        }
        END {
            if (MATCHED == 0) {
                print name, value
            }
        }
    '
}

# 修改配置文件参数
modify_info_param() {
    local filepath="$1"
    local name="$2"
    local value="$3"
    local content

    if [ ! -f "$filepath" ]; then
        return 1
    fi

    content="$(cat "$filepath" | filter_info_param "$name" "$value")"
    with_chmod "$filepath" "700" write_text "$content" "$filepath"
}

# 包创建软链事件
package_create_softlink() {
    local var_path="$1"
    local version="$2"
    local version_dir="$3"
    local package="$4"
    local docker_root="$5"
    local ret install_path pkg_running_version install_for_all

    install_path="$(dirname "$(dirname "$var_path")")"

    get_running_package_version "pkg_running_version" "$install_path/$LATEST_DIR" "$package"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    if [ "$pkg_running_version" != "" ]; then
        unpack_version_pair "version_pair_arr" "$pkg_running_version"
        __index_list "$version_pair_arr" 0 "last_version" 1 "last_version_dir"
        compat_del_package_softlink_in_latest "$install_path" "$package" "$last_version" "$last_version_dir" \
            "$LATEST_DIR" "$USERNAME" "$docker_root"
        check_ret_error "$?" "delete $package softlink in latest failed in package create softlink!"
        ret="$?" && [ $ret -ne 0 ] && return $ret
    fi

    get_install_for_all "install_for_all"
    ret="$?" && [ $ret -ne 0 ] && return $ret
    INSTALL_FOR_ALL="$install_for_all"

    do_create_package_softlink_to_latest "$install_path" "$package" "$version" "$version_dir" "$LATEST_DIR" \
        "$USERNAME" "$USERGROUP" "$install_for_all" "$docker_root"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    return 0
}

# 包安装事件
package_installed() {
    local var_path="$1"
    local version="$2"
    local version_dir="$3"
    local package="$4"
    local install_for_all="$5"
    local docker_root="$6"
    local ret install_path pkg_running_version version_pair_arr last_version last_version_dir
    local config_install_for_all

    if [ "$INCREMENT" != "y" ]; then
        inc_manager_refs "$var_path"
    fi
    install_path="$(dirname "$(dirname "$var_path")")"

    create_total_create_softlink_script "$var_path/manager" "$install_path" "$version_dir" "$USERNAME" "$USERGROUP"
    check_ret_warning "$?" "$package create total create softlink script in $version_dir in failed!"

    create_total_remove_softlink_script "$var_path/manager" "$install_path" "$LATEST_DIR" "$USERNAME" "$USERGROUP"
    check_ret_warning "$?" "Create total remove softlink script in $LATEST_DIR in package installed failed!"

    create_platform_ini "$install_path" "$LATEST_DIR" "$USERNAME" "$USERGROUP"
    check_ret_warning "$?" "Create platform.ini in $LATEST_DIR in package installed failed!"

    get_install_for_all "config_install_for_all"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    if [ "$install_for_all" != "$config_install_for_all" ]; then
        ensure_file "$MANAGER_INFO" "440" "$USERNAME" "$USERGROUP" "$install_for_all"
        ret="$?" && [ $ret -ne 0 ] && return $ret

        modify_info_param "$MANAGER_INFO" "install_for_all" "$install_for_all"
        ret="$?" && [ $ret -ne 0 ] && return $ret
    fi

    package_create_softlink "$var_path" "$version" "$version_dir" "$package" "$docker_root"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    add_installed_package_version "$install_path/$LATEST_DIR" "$package" "$version" "$version_dir"
    check_ret_error "$?" "Add $package installed version $version $version_dir in multi version install failed!"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    return 0
}

# 包删除软链事件
package_remove_softlink() {
    local var_path="$1"
    local version="$2"
    local version_dir="$3"
    local package="$4"
    local docker_root="$5"
    local ret install_path is_running install_for_all

    install_path="$(dirname "$(dirname "$var_path")")"

    # version参数可能为空
    # 1. 新版本升级老版本场景，断开兼容性软链时，会使用老版本的install_common_parser.sh，
    #    调用新版本的--remove-package-latest-softlink，但不会传version参数。
    # 2. 新版本先装，老版本后装的场景，latest目录下remove_latest_softlink.sh为老版本，
    #    调用remove_latest_softlink.sh时，不会传version参数。
    if [ "$version" = "" ]; then
        get_version_by_version_dir "version" "$install_path/$LATEST_DIR" "$version_dir"
    fi

    get_install_for_all "install_for_all"
    ret="$?" && [ $ret -ne 0 ] && return $ret
    INSTALL_FOR_ALL="$install_for_all"

    is_package_version_running "is_running" "$install_path/$LATEST_DIR" "$package" "$version" "$version_dir"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    if [ "$is_running" = "true" ]; then
        do_del_package_softlink_in_latest "$install_path" "$package" "$version" "$version_dir" "$LATEST_DIR" \
            "$USERNAME" "$docker_root"
        ret="$?" && [ $ret -ne 0 ] && return $ret
    fi

    return 0
}

# 包卸载前事件
package_pre_uninstall() {
    return 0
}

# 包卸载事件
package_uninstalled() {
    local var_path="$1"
    local version="$2"
    local version_dir="$3"
    local package="$4"
    local is_recreate_softlink="$5"
    local docker_root="$6"
    local install_path is_upgrade running_packages

    install_path="$(dirname "$(dirname "$var_path")")"

    if ! is_package_version_upgrade "is_upgrade" "$install_path/$LATEST_DIR" "$package" "$version" "$version_dir"; then
        return 1
    fi
    if [ "$is_upgrade" = "true" ]; then
        # 移除version.cfg中的upgrade配置
        if ! unset_upgrade_package "$install_path/$LATEST_DIR" "$package"; then
            comm_log "ERROR" "Unset $package upgrade version in package uninstalled failed!"
            return 1
        fi
    fi

    if ! del_installed_package_version "$install_path/$LATEST_DIR" "$package" "$version" "$version_dir"; then
        comm_log "ERROR" "Del $package installed version $version $version_dir in package uninstalled failed!"
        return 1
    fi

    # 删除版本目录下创建软链总脚本
    if ! del_total_create_softlink_script "$install_path" "$LATEST_DIR" "$version" "$version_dir"; then
        comm_log "WARNING" "Del total create softlink script in $version_dir in package uninstalled failed!"
    fi

    if ! version_cfg_exists "$install_path/$LATEST_DIR"; then
        # 删除latest目录下删除软链总脚本
        del_total_remove_softlink_script "$install_path" "$LATEST_DIR"
        check_ret_warning "$?" "Del total remove softlink script in $LATEST_DIR in package uninstalled failed!"

        # 删除latest目录下platform.ini
        del_platform_ini "$install_path" "$LATEST_DIR"
        check_ret_warning "$?" "Del platform.ini in $LATEST_DIR in package uninstalled failed!"
    fi

    if [ "$is_recreate_softlink" = "y" ]; then
        # 多版本卸载时检查版本兼容性
        if ! recreate_compatiable_softlink_in_multi_version_uninstall "$install_path" "$LATEST_DIR" "$package" \
            "$USERNAME" "$USERGROUP" "$docker_root"; then
            comm_log "ERROR" "Recreate ${package} compatiable softlink in package uninstalled failed!"
            return 1
        fi
    fi

    dec_manager_refs "$var_path"
    if ! manager_refs_exists "$var_path"; then
        rm -f "$MANAGER_INFO"
        sh "$var_path/manager/uninstall.sh"
    fi
    return 0
}

# 设置版本软链接
create_version_softlink() {
    local var_path="$1"
    local version_dir="$2"
    local install_path
    local total_ret="0" version packages package version_pair version_pair_arr
    local ret del_version del_version_dir install_for_all

    install_path="$(dirname "$(dirname "$var_path")")"

    get_version_by_version_dir "version" "$install_path/$LATEST_DIR" "$version_dir"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    get_installed_packages_by_version_version_dir "packages" "$install_path/$LATEST_DIR" "$version" "$version_dir"
    if [ "$packages" = "" ]; then
        return 1
    fi

    get_install_for_all "install_for_all"
    ret="$?" && [ $ret -ne 0 ] && return $ret
    INSTALL_FOR_ALL="$install_for_all"

    for package in ${packages}; do
        if ! get_running_package_version "version_pair" "$install_path/$LATEST_DIR" "$package"; then
            total_ret=1
            continue
        fi

        if [ "$version_pair" != "" ]; then
            unpack_version_pair "version_pair_arr" "$version_pair"
            __index_list "$version_pair_arr" 0 "del_version" 1 "del_version_dir"

            if ! do_del_package_softlink_in_latest "$install_path" "$package" "$del_version" "$del_version_dir" "$LATEST_DIR" \
                "$USERNAME" ""; then
                total_ret=1
            fi
        fi

        if ! do_create_package_softlink_to_latest "$install_path" "$package" "$version" "$version_dir" "$LATEST_DIR" \
            "$USERNAME" "$USERGROUP" "$install_for_all" ""; then
            total_ret=1
        fi
    done

    return $total_ret
}

# 删除运行软链接
remove_latest_softlink() {
    local var_path="$1"
    local install_path
    local total_ret="0" ret running_packages package version_pair version_pair_arr version version_dir
    local install_for_all

    install_path="$(dirname "$(dirname "$var_path")")"

    get_running_packages "running_packages" "$install_path/$LATEST_DIR"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    if [ "${running_packages}" = "" ]; then
        return 0
    fi

    get_install_for_all "install_for_all"
    ret="$?" && [ $ret -ne 0 ] && return $ret
    INSTALL_FOR_ALL="$install_for_all"

    for package in ${running_packages}; do
        if ! get_running_package_version "version_pair" "$install_path/$LATEST_DIR" "$package"; then
            total_ret=1
            continue
        fi

        unpack_version_pair "version_pair_arr" "$version_pair"
        __index_list "$version_pair_arr" 0 "version" 1 "version_dir"

        if ! do_del_package_softlink_in_latest "$install_path" "$package" "$version" "$version_dir" "$LATEST_DIR" \
            "$USERNAME" ""; then
            total_ret=1
        fi
    done

    return $total_ret
}

# 创建子包软链到latest目录下
do_create_package_softlink_to_latest() {
    local install_path="$1"
    local package="$2"
    local version="$3"
    local version_dir="$4"
    local latest_dir="$5"
    local username="$6"
    local usergroup="$7"
    local install_for_all="$8"
    local docker_root="$9"
    local ret total_ret="0" install_type feature_type feature_param filelist_path install_info_path
    local chip_type

    check_param_not_empty "package" "need set package parameter in create package softlink to latest!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    _package_to_log_pkg_name "LOG_PKG_NAME" "${package}"
    check_ret_error "$?" "Set log package name failed in create package softlink to latest!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    if [ ! -d "${install_path}/${latest_dir}" ]; then
        make_dir_with_permission "${install_path}/${latest_dir}" "750" "${username}" "${usergroup}" "${install_for_all}"
        ret="$?" && [ $ret -ne 0 ] && return $ret
    fi

    get_package_filelist "filelist_path" "${install_path}" "${version_dir}" "${package}"
    get_package_install_info "install_info_path" "${install_path}" "${version_dir}" "${package}"

    check_file_exists "${install_info_path}" "${install_info_path} doesn't exist in create package softlink to latest!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    get_package_install_type "install_type" "${install_info_path}" "${package}"
    check_ret_error "$?" "Get install_type from ascend_install.info failed in create package softlink to latest!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    get_package_feature_type "feature_type" "${install_info_path}" "${package}" "install"
    check_ret_error "$?" "Get feature_type from ascend_install.info failed in create package softlink to latest!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    get_package_chip_type "chip_type" "${install_info_path}" "${package}" "install"
    check_ret_error "$?" "Get chip_type from ascend_install.info failed in create package softlink to latest!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    pack_feature_param "feature_param" "${feature_type}" "n" "${chip_type}"

    create_common_dirs_softlink_to_latest "${install_type}" "${install_path}" "${package}" "${version_dir}" "${latest_dir}" \
        "${filelist_path}" "${feature_param}" "${username}" "${usergroup}" "${install_for_all}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    # 创建公共目录到latest目录下的软链接
    if [ "${USE_SHARE_INFO}" != "y" ]; then
        # 创建版本包到latest目录下的软链接
        create_package_dir_softlink_to_latest "${install_path}" "${version_dir}" "${latest_dir}" "${package}" \
            "${username}" "${usergroup}" "${install_for_all}"
        check_ret_error "$?" "Create $package $version_dir $latest_dir package dirs softlink to latest failed!"
        ret="$?" && [ $ret -ne 0 ] && return $ret
    else
        if [ ! -d "${install_path}/${latest_dir}/share" ]; then
            ln -srf "${install_path}/${version_dir}/share" "${install_path}/${latest_dir}/share"
            ret="$?" && [ $ret -ne 0 ] && return $ret
        fi
    fi

    # latest目录下公共脚本添加条目
    add_latest_common_script "${install_path}/${latest_dir}" "${package}" "${username}" "${usergroup}" "${docker_root}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    # 更新version.cfg中的running配置
    set_running_package_version "${install_path}/${latest_dir}" "${package}" "${version}" "${version_dir}"
    check_ret_error "$?" "Set ${package} running version ${version} ${version_dir} in create package softlink to latest failed!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    # 更新version.cfg中的upgrade配置
    set_upgrade_package_version "${install_path}/${latest_dir}" "${package}" "${version}" "${version_dir}"
    check_ret_error "$?" "Set ${package} upgrade version ${version} ${version_dir} in create package softlink to latest failed!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    return ${total_ret}
}


# [兼容老版本]创建子包软链到latest目录下
compat_create_package_softlink_to_latest() {
    local install_path="$1"
    local package="$2"
    local version="$3"
    local version_dir="$4"
    local latest_dir="$5"
    local username="$6"
    local usergroup="$7"
    local install_for_all="$8"
    local docker_root="$9"
    local ret installer_path

    get_package_install_common_parser "installer_path" "${install_path}" "${version_dir}" "${package}"
    check_file_exists "${installer_path}" "${installer_path} doesn't exist in create package softlink to latest!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    install_options="--create-latest-softlink"
    install_options="${install_options} --install-path=\"${install_path}\" --package=\"${package}\" --version=\"${version}\" --version-dir=\"${version_dir}\""
    install_options="${install_options} --latest-dir=\"${latest_dir}\" --username=\"${username}\" --usergroup=\"${usergroup}\""
    if [ "${install_for_all}" = "y" ]; then
        install_options="${install_options} --install_for_all"
    fi
    if [ "${docker_root}" != "" ]; then
        install_options="${install_options} --docker-root=\"${docker_root}\""
    fi

    eval sh "${installer_path}" "${install_options}"
    check_ret_error "$?" "Create ${version_dir} ${package} softlink to latest failed!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    return 0
}

# 删除latest目录下子包的软链接
do_del_package_softlink_in_latest() {
    local install_path="$1"
    local package="$2"
    local version="$3"
    local version_dir="$4"
    local latest_dir="$5"
    local username="$6"
    local docker_root="$7"
    local ret install_type feature_type feature_param package_dirpath filelist_path install_info_path
    local chip_type

    _package_to_log_pkg_name "LOG_PKG_NAME" "${package}"
    check_ret_error "$?" "Set log package name failed in del package softlink in latest!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    if [ ! -d "${install_path}/${latest_dir}" ]; then
        comm_log "ERROR" "${install_path}/${latest_dir} doesn't exist in del package softlink in latest!"
        return 1
    fi

    get_package_dirpath "package_dirpath" "$package"
    filelist_path="$install_path/$version_dir/$package_dirpath/script/filelist.csv"
    install_info_path="$install_path/$version_dir/$package_dirpath/ascend_install.info"

    check_file_exists "${install_info_path}" "${install_info_path} doesn't exist in del package softlink in latest!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    # 卸载时install_type强制为full
    install_type="full"

    get_package_feature_type "feature_type" "${install_info_path}" "${package}" "uninstall"
    check_ret_error "$?" "Get feature_type from ascend_install.info failed in del package softlink in latest!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    get_package_chip_type "chip_type" "${install_info_path}" "${package}" "uninstall"
    check_ret_error "$?" "Get chip_type from ascend_install.info failed in del package softlink to latest!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    pack_feature_param "feature_param" "${feature_type}" "n" "${chip_type}"

    del_latest_common_script "${install_path}/${latest_dir}" "${package}" "${username}" "${docker_root}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    # 删除公共目录到latest目录下的软链接
    del_common_dirs_softlink_from_latest "${install_type}" "${install_path}" "${package}" "${version_dir}" "${latest_dir}" \
        "${filelist_path}" "${feature_param}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    # latest目录下公共脚本删除条目
    if [ "${USE_SHARE_INFO}" != "y" ]; then
        # 删除版本包到latest目录下的软链接
        del_package_dir_softlink_in_latest "${install_path}" "${latest_dir}" "${package}"
        check_ret_error "$?" "Delete $package package dir softlink in latest in del package softlink in latest failed!"
        ret="$?" && [ $ret -ne 0 ] && return $ret
    else
        if [ -L "${install_path}/${latest_dir}/share" ]; then
            rm -f "${install_path}/${latest_dir}/share"
            ret="$?" && [ $ret -ne 0 ] && return $ret
        fi
    fi

    # 删除latest下tools空目录。toolkit包创建latest软链时，latest下存在软链tools/simulator -> ${arch}-linux/simulator
    # 通过filelist.csv文件，无法删除tools目录
    # 解决toolkit包安装卸载latest目录下残留tools目录问题
    remove_dir_if_empty "${install_path}/${latest_dir}/tools"

    # 移除version.cfg中的running配置
    unset_running_package "${install_path}/${latest_dir}" "${package}"
    check_ret_error "$?" "Unset ${package} running version in del package softlink in latest failed!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    return 0
}

# [兼容老版本]删除latest目录下子包的软链接
compat_del_package_softlink_in_latest() {
    local install_path="$1"
    local package="$2"
    local version="$3"
    local version_dir="$4"
    local latest_dir="$5"
    local username="$6"
    local docker_root="$7"
    local ret installer_path

    get_package_install_common_parser "installer_path" "${install_path}" "${version_dir}" "${package}"

    check_file_exists "${installer_path}" "${installer_path} doesn't exist in del package softlink in latest!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    # 更新version.cfg中的upgrade配置
    set_upgrade_package_version "${install_path}/${latest_dir}" "${package}" "${version}" "${version_dir}"
    check_ret_error "$?" "Set ${package} upgrade version ${version} ${version_dir} in create package softlink to latest failed!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    install_options="--remove-latest-softlink"
    install_options="${install_options} --install-path=\"${install_path}\" --package=\"${package}\" --version-dir=\"${version_dir}\""
    install_options="${install_options} --latest-dir=\"${latest_dir}\" --username=\"${username}\""
    if [ "${docker_root}" != "" ]; then
        install_options="${install_options} --docker-root=\"${docker_root}\""
    fi

    eval sh "${installer_path}" "${install_options}"
    check_ret_error "$?" "Remove ${version_dir} ${package} softlink in latest failed!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    return 0
}

# 获取包安装类型
get_package_install_type() {
    local _outvar="$1"
    local _install_info="$2"
    local _package="$3"
    local _install_type_gpit=""

    eval "${_outvar}=\"\""

    if [ ! -f "${_install_info}" ]; then
        return 1
    fi

    _install_type_gpit="$(grep -i "^\(${_package}_\)\?install_type=" "${_install_info}" | cut -d"=" -f2-)"
    if [ "${_install_type_gpit}" = "" ]; then
        return 1
    fi

    eval "${_outvar}=\"${_install_type_gpit}\""
}

# 获取包特性参数
get_package_feature_type() {
    local _outvar="$1"
    local _install_info="$2"
    local _package="$3"
    local _operation="$4"
    local _feature_type_gpft=""

    eval "${_outvar}=\"\""

    if [ ! -f "${_install_info}" ]; then
        return 1
    fi

    if [ "${_package}" = "opp" ]; then
        _feature_type_gpft="$(grep -i "^Opp_Install_Feature=" "${_install_info}" | cut -d"=" -f2-)"
    else
        _feature_type_gpft="$(grep -i "^\(${_package}_\)\?feature_type=" "${_install_info}" | cut -d"=" -f2-)"
    fi
    if [ "${_feature_type_gpft}" = "" ]; then
        _feature_type_gpft="all"
    fi

    normalize_feature "_feature_type_gpft" "${_feature_type_gpft}" "${_operation}"

    eval "${_outvar}=\"${_feature_type_gpft}\""
}

# 获取包芯片参数
get_package_chip_type() {
    local _outvar="$1"
    local _install_info="$2"
    local _package="$3"
    local _operation="$4"
    local _cihp_gpct=""

    eval "${_outvar}=\"\""

    if [ ! -f "${_install_info}" ]; then
        return 1
    fi

    if [ "${_package}" = "opp" ]; then
        _cihp_gpct="$(grep -i "^Opp_Install_Chip=" "${_install_info}" | cut -d"=" -f2-)"
    else
        _cihp_gpct="$(grep -i "^\(${_package}_\)\?chip_type=" "${_install_info}" | cut -d"=" -f2-)"
    fi
    if [ "${_cihp_gpct}" = "" ]; then
        _cihp_gpct="all"
    fi

    normalize_feature "_cihp_gpct" "${_cihp_gpct}" "${_operation}"

    eval "${_outvar}=\"${_cihp_gpct}\""
}

# 创建latest软链接
create_latest_softlink() {
    local latest_dirpath="$1"
    local line="$2"
    local version_dirpath="$4"
    local target latest_filepath version_filepath softlink
    local ret

    __index_list "${line}" 1 "target" 4 "softlink"

    __set_abs_path "${latest_dirpath}" "${target}" "latest_filepath"
    __set_abs_path "${version_dirpath}" "${target}" "version_filepath"

    if [ ! -e "$version_filepath" ]; then
        # 如果源路径不存在，则跳过软链创建
        return 0
    fi

    if [ -d "${latest_filepath}" ]; then
        remove_dir_icp "${latest_filepath}"
        ret="$?" && [ $ret -ne 0 ] && return $ret
    fi

    create_softlink_icp "-r" "${version_filepath}" "${latest_filepath}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    if [ "${softlink}" != "NA" ]; then
        create_softlink_by_install_path "${latest_dirpath}" "${target}" "${softlink}"
        ret="$?" && [ ${ret} -ne 0 ] && return ${ret}
    fi

    return 0
}

# 删除latest软链接
del_latest_softlink() {
    local latest_dirpath="$1"
    local line="$2"
    local target latest_filepath softlink
    local ret

    __index_list "${line}" 1 "target" 4 "softlink"

    __set_abs_path "${latest_dirpath}" "${target}" "latest_filepath"

    if [ "${softlink}" != "NA" ]; then
        remove_softlinks "${install_path}" "${softlink}"
        ret="$?" && [ ${ret} -ne 0 ] && return ${ret}
    fi

    remove_softlink_icp "${latest_filepath}" "NA"
    ret="$?" && [ $ret -ne 0 ] && return $ret
    return 0
}

# 迁移latest/conf实体目录下的文件到版本conf目录下
migrate_conf_files_from_latest_to_version() {
    local install_type="$1"
    local install_path="$2"
    local version_dir="$3"
    local latest_dir="$4"
    local filelist_path="$5"
    local feature_param="$6"
    local latest_conf_path="${install_path}/${latest_dir}/conf"
    local version_conf_path="${install_path}/${version_dir}/conf"
    local ret

    if [ -d "${latest_conf_path}" ] && [ ! -L "${latest_conf_path}" ] && [ -d "${version_conf_path}" ]; then
        foreach_filelist "filter_common_dirs" "reset_mod_dirs" "${install_type}" "${install_path}/${version_dir}" "mkdir" \
            "${filelist_path}" "${feature_param}" "no" "normal"
        ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

        is_dir_empty "${latest_conf_path}"
        # 如果目录不为空，则迁移数据
        if [ $? -ne 0 ]; then
            mv -f "${latest_conf_path}"/* "${version_conf_path}"
            ret="$?" && [ ${ret} -ne 0 ] && return ${ret}
        fi

        remove_dir_icp "${latest_conf_path}"
        ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

        foreach_filelist "filter_common_dirs" "change_mod_and_own_dirs" "${install_type}" "${install_path}/${version_dir}" "mkdir" \
            "${filelist_path}" "${feature_param}" "reverse" "normal"
        ret="$?" && [ ${ret} -ne 0 ] && return ${ret}
    fi

    return 0
}

# 按块创建公共目录软链接到latest目录下
create_common_dirs_softlink_cross_version() {
    local install_path="$1"
    local package="$2"
    local version_dir="$3"
    local latest_dir="$4"
    local blocks="$5"
    local filelist_path install_info_path install_type feature_type chip_type feature_param

    get_package_filelist "filelist_path" "$install_path" "$version_dir" "$package"
    get_package_install_info "install_info_path" "$install_path" "$version_dir" "$package"

    check_file_exists "$install_info_path" "$install_info_path doesn't exist in create common dirs softlink cross version!"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    get_package_install_type "install_type" "$install_info_path" "$package"
    check_ret_error "$?" "Get install_type from ascend_install.info failed in create common dirs softlink cross version!"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    get_package_feature_type "feature_type" "$install_info_path" "$package" "install"
    get_package_chip_type "chip_type" "$install_info_path" "$package" "install"
    pack_feature_param "feature_param" "$feature_type" "n" "$chip_type"

    create_common_dirs_softlink_by_blocks "$install_type" "$install_path" "$package" "$version_dir" "$latest_dir" \
        "$filelist_path" "$feature_param" "$blocks" ""
    check_ret_error "$?" "Create common dirs softlink by blocks failed in create common dirs softlink cross version!"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    return 0
}

# 按块创建公共目录软链接到latest目录下
create_common_dirs_softlink_by_blocks() {
    local install_type="$1"
    local install_path="$2"
    local package="$3"
    local version_dir="$4"
    local latest_dir="$5"
    local filelist_path="$6"
    local feature_param="$7"
    local blocks="$8"
    local custom_create_softlink="$9"
    local ret package_dirpath filelist_path

    foreach_filelist_v2 "reset_mod_dirs" "${install_type}" "${install_path}/${latest_dir}" "mkdir" \
        "${filelist_path}" "${feature_param}" "filter_common_dirs,filter_blocks" "$blocks" "no" "normal"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    foreach_filelist_v2 "create_dirs" "${install_type}" "${install_path}/${latest_dir}" "mkdir" \
        "${filelist_path}" "${feature_param}" "filter_common_dirs,filter_blocks" "$blocks" "no" "normal"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    foreach_filelist_v2 "create_latest_softlink" "${install_type}" "${install_path}/${latest_dir}" "copy copy_entity move" \
        "${filelist_path}" "${feature_param}" "filter_common_dirs,filter_blocks" "$blocks" "no" "normal" "${install_path}/${version_dir}"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    # 调用脚本创建自定义内容软链接到latest目录下
    if [ "$custom_create_softlink" != "" ] && [ -f "$custom_create_softlink" ]; then
        chmod u+x "${custom_create_softlink}"
        "${custom_create_softlink}" --install-path="${install_path}" --version-dir="${version_dir}" --latest-dir="${latest_dir}"
        check_ret_error "$?" "Run ${package} custom create ${version_dir} softlink failed!"
        ret="$?" && [ ${ret} -ne 0 ] && return ${ret}
    fi

    foreach_filelist_v2 "change_mod_and_own_dirs" "${install_type}" "${install_path}/${latest_dir}" "mkdir" \
        "${filelist_path}" "${feature_param}" "filter_common_dirs,filter_blocks" "$blocks" "reverse" "normal"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    return 0
}

# 创建公共目录软链接到latest目录下
create_common_dirs_softlink_to_latest() {
    local install_type="$1"
    local install_path="$2"
    local package="$3"
    local version_dir="$4"
    local latest_dir="$5"
    local filelist_path="$6"
    local feature_param="$7"
    local username="$8"
    local usergroup="$9"
    local install_for_all="${10}"
    local db_info_path="$install_path/$latest_dir/var/ascend_package_db.info"
    local ret package_dirpath custom_create_softlink db_info

    get_package_dirpath "package_dirpath" "$package"

    custom_create_softlink="${install_path}/${version_dir}/${package_dirpath}/script/${package}_custom_create_softlink.sh"

    migrate_conf_files_from_latest_to_version "${install_type}" "${install_path}" "${version_dir}" "${latest_dir}" \
        "${filelist_path}" "${feature_param}"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    ensure_file "$db_info_path" "440" "$username" "$usergroup" "$install_for_all"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    db_info="$(
        (
            cat "$db_info_path"
            printf "\n"  # 防止db.info缺失末尾换行符
            all_common_dirs_blocks_in_filelist "$install_type" "$filelist_path" "$feature_param" \
                | blocks_to_db_item "$package" "$version_dir"
        ) | remove_blank_line | sort_1 "|" | fold_2 "|" ""
    )"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    # 注意换行符
    (
        cat "$db_info_path" | remove_blank_line | remain_db_last_item
        printf "%s\n" "$db_info" | remove_blank_line | remain_db_last_item
    ) | sort_1 | fold_3_keep_2 "" " " | show_diff_3_4 | show_min_nf "4" | select_fields_3_2_1 | sort_1 | fold_3_keep_2 "" " " \
    | (
        total_ret=0
        while read -r tmp_version_dir tmp_package tmp_blocks; do
            del_common_dirs_softlink_cross_version "$install_path" "$tmp_package" "$tmp_version_dir" "$latest_dir" \
                "EngineeringCommon $tmp_blocks"
            ret="$?" && [ $ret -ne 0 ] && total_ret="1"
        done
        exit $total_ret
    )
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    # 安装时总是创建全部块的软链接
    create_common_dirs_softlink_by_blocks "$install_type" "$install_path" "$package" "$version_dir" "$latest_dir" \
        "$filelist_path" "$feature_param" "" "$custom_create_softlink"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    with_chmod "$db_info_path" "700" write_text "$db_info" "$db_info_path"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    return 0
}


# 跨版本按块删除公共目录到latest目录下的软链接
del_common_dirs_softlink_cross_version() {
    local install_path="$1"
    local package="$2"
    local version_dir="$3"
    local latest_dir="$4"
    local blocks="$5"
    local filelist_path install_info_path install_type feature_type chip_type feature_param

    get_package_filelist "filelist_path" "$install_path" "$version_dir" "$package"
    get_package_install_info "install_info_path" "$install_path" "$version_dir" "$package"

    check_file_exists "$install_info_path" "$install_info_path doesn't exist in del common dirs softlink cross version!"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    get_package_install_type "install_type" "$install_info_path" "$package"
    check_ret_error "$?" "Get install_type from ascend_install.info failed in del common dirs softlink cross version!"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    get_package_feature_type "feature_type" "$install_info_path" "$package" "uninstall"
    get_package_chip_type "chip_type" "$install_info_path" "$package" "uninstall"
    pack_feature_param "feature_param" "$feature_type" "n" "$chip_type"

    del_common_dirs_softlink_by_blocks "$install_type" "$install_path" "$package" "$version_dir" "$latest_dir" \
        "$filelist_path" "$feature_param" "$blocks" ""
    check_ret_error "$?" "Del common dirs softlink by blocks failed in del common dirs softlink cross version!"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    return 0
}

# 按块删除公共目录到latest目录下的软链接
del_common_dirs_softlink_by_blocks() {
    local install_type="$1"
    local install_path="$2"
    local package="$3"
    local version_dir="$4"
    local latest_dir="$5"
    local filelist_path="$6"
    local feature_param="$7"
    local blocks="$8"
    local custom_remove_softlink="$9"
    local ret package_dirpath filelist_path

    create_stash_mod "${install_path}/${latest_dir}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    foreach_filelist_v2 "reset_mod_dirs_with_stash_mod" "${install_type}" "${install_path}/${latest_dir}" "mkdir" \
        "${filelist_path}" "${feature_param}" "filter_common_dirs,filter_blocks" "$blocks" "no" "normal"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    # 调用脚本删除latest目录下自定义内容软链接
    if [ "$custom_remove_softlink" != "" ] && [ -f "$custom_remove_softlink" ]; then
        chmod u+x "${custom_remove_softlink}"
        "${custom_remove_softlink}" --install-path="${install_path}" --version-dir="${version_dir}" --latest-dir="${latest_dir}"
        check_ret_error "$?" "Run ${package} custom remove ${version_dir} softlink failed!"
        ret="$?" && [ ${ret} -ne 0 ] && return ${ret}
    fi

    foreach_filelist_v2 "del_latest_softlink" "${install_type}" "${install_path}/${latest_dir}" "copy copy_entity move" \
        "${filelist_path}" "${feature_param}" "filter_common_dirs,filter_blocks" "$blocks" "no" "normal"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    foreach_filelist_v2 "remove_install_dirs" "${install_type}" "${install_path}/${latest_dir}" "mkdir" \
        "${filelist_path}" "${feature_param}" "filter_common_dirs,filter_blocks" "$blocks" "reverse" "normal"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    foreach_stashmod "restore_stash_mod" "${install_path}/${latest_dir}" "reverse"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    remove_stash_mod "${install_path}/${latest_dir}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    return 0
}

# 删除公共目录到latest目录下的软链接
del_common_dirs_softlink_from_latest() {
    local install_type="$1"
    local install_path="$2"
    local package="$3"
    local version_dir="$4"
    local latest_dir="$5"
    local filelist_path="$6"
    local feature_param="$7"
    local db_info_path="$install_path/$latest_dir/var/ascend_package_db.info"
    local ret db_info_origin db_info diff_result blocks_to_remove package_dirpath custom_remove_softlink

    get_package_dirpath "package_dirpath" "$package"
    custom_remove_softlink="${install_path}/${version_dir}/${package_dirpath}/script/${package}_custom_remove_softlink.sh"

    if [ -f "$db_info_path" ]; then
        db_info_origin="$(cat "$db_info_path")""\n"
    else
        db_info_origin=""
    fi

    db_info="$(printf "$db_info_origin" | del_db_items "$package" "$version_dir")"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    # 注意换行符和次序
    diff_result="$(
        (
            printf "%s\n" "$db_info" | remove_blank_line | remain_db_last_item
            printf "$db_info_origin" | remove_blank_line | remain_db_last_item
        ) | sort_1 | fold_3_keep_2 "" " " | show_diff_3_4
    )"
    blocks_to_remove="$(printf "%s\n" "$diff_result" | select_fields_1 | xargs)"

    # blocks_to_remove可以为空
    del_common_dirs_softlink_by_blocks "$install_type" "$install_path" "$package" "$version_dir" "$latest_dir" \
        "$filelist_path" "$feature_param" "EngineeringCommon $blocks_to_remove" "$custom_remove_softlink"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    printf "%s\n" "$diff_result" | show_min_nf "4" | select_fields_3_2_1 | sort_1 | fold_3_keep_2 "" " " \
    | (
        total_ret=0
        while read -r tmp_version_dir tmp_package tmp_blocks; do
            create_common_dirs_softlink_cross_version "$install_path" "$tmp_package" "$tmp_version_dir" "$latest_dir" \
                "EngineeringCommon $tmp_blocks"
            ret="$?" && [ $ret -ne 0 ] && total_ret="1"
        done
        exit $total_ret
    )
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    if [ "$db_info" != "" ]; then
        with_chmod "$db_info_path" "700" write_text "$db_info" "$db_info_path"
    else
        rm -f "$db_info_path"
    fi
    ret="$?" && [ $ret -ne 0 ] && return $ret

    return 0
}

# latest目录下公共脚本添加条目
add_latest_common_script() {
    local latest_path="$1"
    local package="$2"
    local username="$3"
    local usergroup="$4"
    local docker_root="$5"
    local ret mod

    # 保存bin目录的权限
    get_file_mod "mod" "-L" "${latest_path}/bin"

    # 给latest/bin目录提升权限到750
    chmod "750" "${latest_path}/bin"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    # 设置setenv
    add_setenv "${latest_path}" "${package}" "NA" "${username}" "${usergroup}" "true" "${docker_root}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    # 恢复bin目录的权限
    chmod "${mod}" "${latest_path}/bin"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    return 0
}

# latest目录下公共脚本删除条目
del_latest_common_script() {
    local latest_path="$1"
    local package="$2"
    local username="$3"
    local docker_root="$4"
    local ret

    if [ ! -d "${latest_path}/bin" ]; then
        return 0
    fi

    # 保存bin目录的权限
    get_file_mod "mod" "-L" "${latest_path}/bin"

    # 给latest/bin目录提升权限到750
    chmod "750" "${latest_path}/bin"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    # unsetenv
    del_setenv "${latest_path}" "${package}" "${username}" "${docker_root}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    # 恢复bin目录的权限
    chmod "${mod}" "${latest_path}/bin"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    return 0
}

# 文件中添加一个空行
add_blank_line_to_file() {
    local filepath="$1"

    echo >> "${filepath}"
}

# 总设置软链脚本是否存在
# 返回值0为真，1为假
total_create_softlink_script_exists() {
    local install_path="$1"
    local version_dir="$2"

    if [ -f "$install_path/$version_dir/$CREATE_VERSION_SOFTLINK" ]; then
        return 0
    fi
    return 1
}

# 创建总设置软链脚本
create_total_create_softlink_script() {
    local script_dir="$1"
    local install_path="$2"
    local version_dir="$3"
    local username="$4"
    local usergroup="$5"
    local script_filepath="$install_path/$version_dir/$CREATE_VERSION_SOFTLINK"
    local ret

    rm -f "$script_filepath"
    cat "$script_dir/common_installer.inc" > "$script_filepath"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    add_blank_line_to_file "$script_filepath"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    sed -n '/^## module log/,/^## end module/ p' "$script_dir/common_func.inc" >> "$script_filepath"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    add_blank_line_to_file "$script_filepath"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    echo "LATEST_DIR=\"$LATEST_DIR\"" >> "$script_filepath"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    add_blank_line_to_file "$script_filepath"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    echo "notify_latest_manager_create_version_softlink" >> "$script_filepath"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    change_mod "$script_filepath" "550" ""
    ret="$?" && [ $ret -ne 0 ] && return $ret

    change_own "$script_filepath" "${username}:${usergroup}"
    ret="$?" && [ $ret -ne 0 ] && return $ret


    return 0
}

# 总删除软链脚本是否存在
# 返回值0为真，1为假
total_remove_softlink_script_exists() {
    local install_path="$1"
    local latest_dir="$2"

    if [ -f "$install_path/$latest_dir/$REMOVE_LATEST_SOFTLINK" ]; then
        return 0
    fi
    return 1
}

# 创建总删除软链脚本
create_total_remove_softlink_script() {
    local script_dir="$1"
    local install_path="$2"
    local latest_dir="$3"
    local username="$4"
    local usergroup="$5"
    local script_filepath="$install_path/$latest_dir/$REMOVE_LATEST_SOFTLINK"
    local ret

    rm -f "$script_filepath"
    cat "$script_dir/common_installer.inc" > "$script_filepath"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    add_blank_line_to_file "$script_filepath"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    sed -n '/^## module log/,/^## end module/ p' "$script_dir/common_func.inc" >> "$script_filepath"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    add_blank_line_to_file "$script_filepath"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    echo "LATEST_DIR=\"$LATEST_DIR\"" >> "$script_filepath"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    add_blank_line_to_file "$script_filepath"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    echo "notify_latest_manager_remove_latest_softlink" >> "$script_filepath"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    change_mod "$script_filepath" "550" ""
    ret="$?" && [ $ret -ne 0 ] && return $ret

    change_own "$script_filepath" "${username}:${usergroup}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    return 0
}

# 删除总创建软链脚本
del_total_create_softlink_script() {
    local install_path="$1"
    local latest_dir="$2"
    local version="$3"
    local version_dir="$4"
    local script_filepath="${install_path}/${version_dir}/${CREATE_VERSION_SOFTLINK}"
    local ret version_cfg_path packages

    get_version_cfg_path "version_cfg_path" "${install_path}/${latest_dir}"
    if [ ! -f "${version_cfg_path}" ]; then
        remove_file "${script_filepath}" "NA"
        ret="$?" && [ $ret -ne 0 ] && return $ret

        return 0
    fi

    get_installed_packages_by_version_version_dir "packages" "${install_path}/${latest_dir}" "${version}" "${version_dir}"
    if [ "${packages}" = "" ]; then
        remove_file "${script_filepath}" "NA"
        ret="$?" && [ $ret -ne 0 ] && return $ret
    fi

    return 0
}

# 删除总删除软链脚本
del_total_remove_softlink_script() {
    local install_path="$1"
    local latest_dir="$2"
    local script_filepath="${install_path}/${latest_dir}/${REMOVE_LATEST_SOFTLINK}"
    local ret version_cfg_path

    get_version_cfg_path "version_cfg_path" "${install_path}/${latest_dir}"

    if [ ! -f "${version_cfg_path}" ]; then
        remove_file "${script_filepath}" "NA"
        ret="$?" && [ $ret -ne 0 ] && return $ret
    fi

    return 0
}

# platform.ini配置是否存在
# 返回值0为真，1为假
platform_ini_exists() {
    local install_path="$1"
    local latest_dir="$2"

    if [ -f "$install_path/$latest_dir/platform.ini" ]; then
        return 0
    fi
    return 1
}

# 创建platform.ini配置
create_platform_ini() {
    local install_path="$1"
    local latest_dir="$2"
    local username="$3"
    local usergroup="$4"
    local config_dirpath="${install_path}/${latest_dir}"
    local config_filepath="${config_dirpath}/platform.ini"
    local ret mod

    # 配置文件存在则跳过
    if [ -f "${config_filepath}" ]; then
        return 0
    fi

    # 保存当前目录的权限
    get_file_mod "mod" -L "${config_dirpath}"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    # 恢复权限
    cleanup="chmod ${mod} \"${config_dirpath}\""

    # 添加目录写权限
    chmod u+w "${config_dirpath}"
    cleanup_if_error "$?" "${cleanup}"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    touch "${config_filepath}"
    cleanup_if_error "$?" "${cleanup}"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    change_mod "${config_filepath}" "660" ""
    cleanup_if_error "$?" "${cleanup}"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    change_own "${config_filepath}" "${username}:${usergroup}"
    cleanup_if_error "$?" "${cleanup}"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    eval "${cleanup}"

    return 0
}

# 删除platform.ini配置
del_platform_ini() {
    local install_path="$1"
    local latest_dir="$2"
    local config_dirpath="${install_path}/${latest_dir}"
    local config_filepath="${config_dirpath}/platform.ini"
    local ret mod

    if [ ! -f "${config_filepath}" ]; then
        return 0
    fi

    # 保存当前目录的权限
    get_file_mod "mod" -L "${config_dirpath}"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    # 恢复权限
    cleanup="chmod ${mod} \"${config_dirpath}\""

    # 添加目录写权限
    chmod u+w "${config_dirpath}"
    cleanup_if_error "$?" "${cleanup}"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    remove_file "${config_filepath}" "NA"
    cleanup_if_error "$?" "${cleanup}"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    eval "${cleanup}"

    return 0
}

# 获取aicpu创建软链脚本路径
get_aicpu_custom_create_softlink_path() {
    local _outvar="$1"
    local _install_path="$2"
    local _version_dir="$3"
    local _chip_name="$4"

    eval "${_outvar}=\"${_install_path}/${_version_dir}/opp/${_chip_name}/aicpu/script/aicpu_custom_create_softlink.sh\""
}

# 重建软链时处理aicpu软链
deal_with_aicpu_package() {
    local install_path="$1"
    local version_dir="$2"
    local latest_dir="$3"
    local chip_name

    for chip_name in "Ascend910" "Ascend310P" "Ascend310" "Ascend310RC" "Ascend"; do
        get_aicpu_custom_create_softlink_path "aicpu_script" "${install_path}" "${version_dir}" "${chip_name}"
        if [ -f "${aicpu_script}" ]; then
            # 创建软链时，支持强制覆盖老版本软链。
            # 因为存在这样的场景：先卸载新版本的runtime包，再卸载新版本的aicpu_kernels包，
            # 卸载新版本的runtime包时触发该流程，此时老版本的aicpu_kernels包软链还在存在于环境上。
            "${aicpu_script}" --install-path="${install_path}" --version-dir="${version_dir}" --latest-dir="${latest_dir}"
            # 忽略aicpu创建软链报错
        fi
    done

    return 0
}

# 恢复可兼容的子包软链接
recreate_compatiable_softlink_sibling_package() {
    local package="$1"

    # 判断是否有本包的running版本包
    get_running_package_version "version_pair" "$scope_install_path/$scope_latest_dir" "$package"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    if [ "$version_pair" != "" ]; then
        return 0
    fi

    if check_current_package_compatiable "$scope_install_path" "$scope_version" "$scope_version_dir" "$package"; then
        compat_create_package_softlink_to_latest "$scope_install_path" "$package" "$scope_version" "$scope_version_dir" "$scope_latest_dir" \
            "$scope_username" "$scope_usergroup" "$scope_install_for_all" "$scope_docker_root"
        ret="$?" && [ $ret -ne 0 ] && return $ret
    fi

    return 0
}

# 多版本卸载时检查版本兼容性
recreate_compatiable_softlink_in_multi_version_uninstall() {
    local scope_install_path="$1"
    local scope_latest_dir="$2"
    local package="$3"
    local scope_username="$4"
    local scope_usergroup="$5"
    local scope_docker_root="$6"
    local ret installed_versions version_pair version_pair_arr scope_version scope_version_dir
    local scope_install_for_all version_cfg_path sibling_package

    # 切换目录避免使用uninstall.sh脚本，重建子包软链接时，找到不当前路径问题
    # sh: 0: getcwd() failed: No such file or directory
    cd "${scope_install_path}"

    # 判断是否有本包的running版本包
    get_running_package_version "version_pair" "${scope_install_path}/${scope_latest_dir}" "${package}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    if [ "${version_pair}" != "" ]; then
        return 0
    fi

    # 枚举version.cfg中本包installed版本
    get_installed_package_versions "installed_versions" "${scope_install_path}/${scope_latest_dir}" "${package}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    get_install_for_all "scope_install_for_all"
    ret="$?" && [ $ret -ne 0 ] && return $ret
    INSTALL_FOR_ALL="$scope_install_for_all"

    for version_pair in ${installed_versions}; do
        unpack_version_pair "version_pair_arr" "${version_pair}"
        __index_list "${version_pair_arr}" 0 "scope_version" 1 "scope_version_dir"

        if check_current_package_compatiable "$scope_install_path" "$scope_version" "$scope_version_dir" "$package"; then
            compat_create_package_softlink_to_latest "${scope_install_path}" "${package}" "${scope_version}" "${scope_version_dir}" "${scope_latest_dir}" \
                "${scope_username}" "${scope_usergroup}" "${scope_install_for_all}" "${scope_docker_root}"
            ret="$?" && [ $ret -ne 0 ] && return $ret

            # 恢复同版本下可兼容的其他子包软链接
            get_version_cfg_path "version_cfg_path" "$scope_latest_dir"
            grep -F "installed_version=" "$version_cfg_path" | grep -F "[$scope_version:$scope_version_dir]" | cut -d= -f1 | sed 's/_installed_version$//' | while read sibling_package; do
                if [ "$sibling_package" != "$package" ]; then
                    recreate_compatiable_softlink_sibling_package "$sibling_package"
                fi
            done

            if [ "$package" = "runtime" ]; then
                deal_with_aicpu_package "${scope_install_path}" "${scope_version_dir}" "${scope_latest_dir}"
                ret="$?" && [ $ret -ne 0 ] && return $ret
            fi

            break
        fi
    done

    return 0
}

# 检查包版本兼容性
check_package_compatiable() {
    local install_path="$1"
    local version_left="$2"
    local version_dir_left="$3"
    local package_left="$4"
    local version_right="$5"
    local version_dir_right="$6"
    local package_right="$7"
    local script_dir="$8"
    local pkg_version_info_path_left pkg_version_info_path_right
    local required_left required_right

    get_package_version_info "pkg_version_info_path_left" "$install_path" "$version_dir_left" "$package_left"
    get_package_version_info "pkg_version_info_path_right" "$install_path" "$version_dir_right" "$package_right"

    if ! _get_required_package_info "required_left" "$pkg_version_info_path_right" "$package_left"; then
        return 2
    fi
    if ! _get_required_package_info "required_right" "$pkg_version_info_path_left" "$package_right"; then
        return 2
    fi

    if [ "$required_left" = "" ] && [ "$required_right" = "" ]; then
        return 0
    fi

    if _check_version_required "$version_left" "$required_left" "$script_dir" || _check_version_required "$version_right" "$required_right" "$script_dir"; then
        return 0
    fi
    return 1
}

# 检查当前包与latest版本兼容性
check_current_package_compatiable() {
    local install_path="$1"
    local current_version="$2"
    local current_version_dir="$3"
    local current_package="$4"
    local script_dir="$install_path/$LATEST_DIR/var/manager"
    local ret running_packages package version_pair version version_dir

    get_running_packages "running_packages" "$install_path/$LATEST_DIR"
    for package in ${running_packages}; do
        if [ "$current_package" = "$package" ]; then
            continue
        fi

        get_running_package_version "version_pair" "$install_path/$LATEST_DIR" "$package"
        ret="$?" && [ $ret -ne 0 ] && return $ret

        unpack_version_pair "version_pair_arr" "$version_pair"
        __index_list "$version_pair_arr" 0 "version" 1 "version_dir"

        if ! check_package_compatiable "$install_path" "$current_version" "$current_version_dir" "$current_package" \
            "$version" "$version_dir" "$package" "$script_dir"; then
            return 1
        fi
    done

    return 0
}

# 多版本安装时检查版本兼容性
check_compatiable_in_multi_version_install() {
    local install_path="$1"
    local username="$2"
    local docker_root="$3"
    local current_version="$4"
    local current_version_dir="$5"
    local current_package="$6"
    local script_dir="$install_path/$LATEST_DIR/var/manager"
    local ret running_packages package version_pair version version_dir

    get_running_packages "running_packages" "$install_path/$LATEST_DIR"
    for package in ${running_packages}; do
        if [ "$current_package" = "$package" ]; then
            continue
        fi

        get_running_package_version "version_pair" "$install_path/$LATEST_DIR" "$package"
        ret="$?" && [ $ret -ne 0 ] && return $ret

        unpack_version_pair "version_pair_arr" "$version_pair"
        __index_list "$version_pair_arr" 0 "version" 1 "version_dir"

        if ! check_package_compatiable "$install_path" "$current_version" "$current_version_dir" "$current_package" \
            "$version" "$version_dir" "$package" "$script_dir"; then
            compat_del_package_softlink_in_latest "$install_path" "$package" "$version" "$version_dir" \
                "$LATEST_DIR" "$username" "$docker_root"
            ret="$?" && [ $ret -ne 0 ] && return $ret
        fi
    done

    return 0
}

# 创建版本包目录软链接到latest目录下
create_package_dir_softlink_to_latest() {
    local install_path="$1"
    local version_dir="$2"
    local latest_dir="$3"
    local package="$4"
    local username="$5"
    local usergroup="$6"
    local install_for_all="$7"
    local ret package_dirpath package_prefix

    get_package_dirpath "package_dirpath" "${package}"
    package_prefix="$(dirname "${package_dirpath}")"

    if [ ! -d "${install_path}/${version_dir}/${package_dirpath}" ]; then
        return 0
    fi

    if [ ! -d "${install_path}/${latest_dir}/${package_prefix}" ]; then
        make_dir_with_permission "${install_path}/${latest_dir}/${package_prefix}" "750" "${username}" "${usergroup}" "${install_for_all}"
        ret="$?" && [ ${ret} -ne 0 ] && return ${ret}
    fi

    create_softlink_icp "-r" "${install_path}/${version_dir}/${package_dirpath}" "${install_path}/${latest_dir}/${package_dirpath}"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    return 0
}

# 删除latest目录下版本包目录的软链接
del_package_dir_softlink_in_latest() {
    local install_path="$1"
    local latest_dir="$2"
    local package="$3"
    local ret package_dirpath package_prefix

    get_package_dirpath "package_dirpath" "${package}"
    package_prefix="$(dirname "${package_dirpath}")"

    remove_softlink_icp "${install_path}/${latest_dir}/${package_dirpath}"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    if [ "${package_prefix}" != "." ]; then
        remove_dir_if_empty "${install_path}/${latest_dir}/${package_prefix}"
        ret="$?" && [ ${ret} -ne 0 ] && return ${ret}
    fi

    return 0
}

# 生成包db.info条目列表
packages_db_items() {
    local install_path="$1"
    local running_packages="$2"
    local total_ret=0 ret running_packages package version_pair version_pair_arr version version_dir
    local install_type feature_type feature_param filelist_path install_info_path

    for package in $running_packages; do
        if ! get_running_package_version "version_pair" "$install_path/$LATEST_DIR" "$package"; then
            total_ret=1
            continue
        fi

        unpack_version_pair "version_pair_arr" "$version_pair"
        __index_list "$version_pair_arr" 0 "version" 1 "version_dir"

        get_package_filelist "filelist_path" "$install_path" "$version_dir" "$package"
        get_package_install_info "install_info_path" "$install_path" "$version_dir" "$package"

        check_file_exists "$install_info_path" "$install_info_path doesn't exist in create package softlink to latest!"
        ret="$?" && [ $ret -ne 0 ] && total_ret=1 && continue

        get_package_install_type "install_type" "$install_info_path" "$package"
        check_ret_error "$?" "Get install_type from ascend_install.info failed in create package softlink to latest!"
        ret="$?" && [ $ret -ne 0 ] && total_ret=1 && continue

        get_package_feature_type "feature_type" "$install_info_path" "$package" "install"
        check_ret_error "$?" "Get feature_type from ascend_install.info failed in create package softlink to latest!"
        ret="$?" && [ $ret -ne 0 ] && total_ret=1 && continue

        get_package_chip_type "chip_type" "$install_info_path" "$package" "install"
        check_ret_error "$?" "Get chip_type from ascend_install.info failed in create package softlink to latest!"
        ret="$?" && [ $ret -ne 0 ] && total_ret=1 && continue

        pack_feature_param "feature_param" "${feature_type}" "n" "${chip_type}"

        all_common_dirs_blocks_in_filelist "$install_type" "$filelist_path" "$feature_param" \
            | blocks_to_db_item "$package" "$version_dir"
        ret="$?" && [ $ret -ne 0 ] && total_ret=1 && continue
    done

    return $total_ret
}

# 生成运行包的ascend_package_db.info
generate_running_packages_db_info() {
    local var_path="$1"
    local ret install_path running_packages db_info install_for_all
    local db_info_path="$var_path/ascend_package_db.info"

    install_path="$(dirname "$(dirname "$var_path")")"

    if [ -f "$db_info_path" ]; then
        return 0
    fi

    get_running_packages "running_packages" "$install_path/$LATEST_DIR"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    if [ "${running_packages}" = "" ]; then
        return 0
    fi

    db_info="$(
        packages_db_items "$install_path" "$running_packages" \
            | remove_blank_line | sort_1 "|" | fold_2 "|" ""
    )"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    get_install_for_all "install_for_all"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    ensure_file "$db_info_path" "440" "$USERNAME" "$USERGROUP" "$install_for_all"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    with_chmod "$db_info_path" "700" write_text "$db_info" "$db_info_path"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    return 0
}

# 升级setenv脚本
upgrade_setenv() {
    local latest_path="$1"
    local shell_type config_path
    for shell_type in ${ENV_SHELL_TYPES}; do
        get_setenv_filepath "config_path" "$latest_path" "$shell_type"
        add_ascend_home_path_env "$config_path" "$shell_type" "$latest_path"
    done
}

# latest下数据迁移
migrate_latest_data() {
    local var_path="$1"
    local latest_path="$(dirname "$var_path")"
    local ret config_path

    generate_running_packages_db_info "$var_path"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    get_setenv_filepath "config_path" "$latest_path" "bash"
    if ! has_ascend_home_path_env "$config_path"; then
        with_chmod "$latest_path/bin" "700" upgrade_setenv "$latest_path"
    fi

    return 0
}
