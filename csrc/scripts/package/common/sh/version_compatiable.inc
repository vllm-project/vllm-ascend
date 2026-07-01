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

# 获取版本号，用于版本兼容性对比
get_package_compat_version() {
    local _outvar="$1"
    local _version_info_path="$2"
    local _result

    if [ ! -f "${_version_info_path}" ]; then
        read -r "$_outvar" <<EOF
EOF
        return 1
    fi

    read -r "$_outvar" <<EOF
$(grep "^Version=" "$_version_info_path" | cut -d= -f2- | cut -d- -f1 | cut -d. -f-3)
EOF
}

# 获取正式包名
get_formal_package_name() {
    local _outvar="$1"
    local _package="$2"
    read -r "$_outvar" <<EOF
$(echo "$_package" | sed 's/_/-/g')
EOF
}

# 获取非正式包名
get_informal_package_name() {
    local _outvar="$1"
    local _package="$2"
    read -r "$_outvar" <<EOF
$(echo "$_package" | sed 's/-/_/g')
EOF
}

# 获取需求包列表。
_get_required_packages() {
    local _version_info_path="$1"

    if [ ! -f "$_version_info_path" ]; then
        return 0
    fi

    awk '
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

        /^required_package_.+_version=/ {
            split($0, line_tokens, "=")
            len_require_tokens = split(line_tokens[1], require_tokens, "_")
            len_package_names = 0
            for (i = 3; i < len_require_tokens; i++) {
                len_package_names++
                package_names[len_package_names] = require_tokens[i]
            }
            package_name = join(package_names, len_package_names, "_")
            print package_name
        }
    ' "${_version_info_path}"
}

# 获取需求包信息。
_get_required_package_info() {
    local _outvar="$1"
    local _version_info_path="$2"
    local _pkg_name="$3"
    local _required_pkg_name
    local _required_value

    if [ ! -f "${_version_info_path}" ]; then
        eval "${_outvar}=\"\""
        return 1
    fi

    _required_pkg_name="required_package_${_pkg_name}_version"
    _required_value="$(grep "^${_required_pkg_name}=" "${_version_info_path}" | cut -d= -f2-)"
    # _required_value取得的值带有双引号，eval时不可以在外侧再添加双引号

    eval "${_outvar}=${_required_value}"
}

# 检查版本与需求版本兼容性。
# 兼容返回0，不兼容返回1
_check_version_required() {
    local version="$1"
    local require="$2"
    local script_dir="$3"
    local src_pkg="$4"
    local dst_pkg="$5"
    local result

    result=$(awk -f "${script_dir}/check_version_required.awk" -v version="${version}" -v all_required="${require}")
    if [ "${result}" = "T" ]; then
        return 0
    fi

    echo "Version compatibility check failed, $src_pkg required $dst_pkg version $require, but $dst_pkg version is $version!"
    return 1
}

# 检查版本兼容性。
_check_version_compatiable() {
    local package="$1"
    local install_path="$2"
    local script_dir="$3"
    local version_info_path="$script_dir/../version.info"
    local err_msgs package_formal self_version src_pkg src_pkg_formal dst_pkg dst_pkg_formal
    local installed_version_info pkg_version_info_path

    get_formal_package_name "package_formal" "$package"
    get_package_compat_version "self_version" "$version_info_path"

    if [ -d "$install_path/share/info" ]; then
        err_msgs="$(
            ls "$install_path/share/info" | grep -v "^${package}$" | while read src_pkg; do
                get_formal_package_name "src_pkg_formal" "$package"
                installed_version_info="$install_path/share/info/$src_pkg/version.info"
                grep "^required_package_${package_formal}_version=" "$installed_version_info" | cut -d= -f2- | tr -d '"' | while read required; do
                    _check_version_required "$self_version" "$required" "$script_dir" "$src_pkg_formal" "$package_formal"
                done
            done
    )"
        if [ "$err_msgs" != "" ]; then
            while read err_msg; do
                comm_log "ERROR" "$err_msg"
            done <<EOF
$err_msgs
EOF
            return 1
        fi
    fi
    err_msgs="$(
        _get_required_packages "${version_info_path}" | while read dst_pkg_formal; do
            get_informal_package_name "dst_pkg" "$dst_pkg_formal"
            pkg_version_info_path="$install_path/share/info/$dst_pkg_formal/version.info"
            if [ ! -f "$pkg_version_info_path" ]; then
                pkg_version_info_path="$install_path/share/info/$dst_pkg/version.info"
            fi
            if [ -f "$pkg_version_info_path" ]; then
                get_package_compat_version "pkg_version" "$pkg_version_info_path"
                _get_required_package_info "required" "$version_info_path" "$dst_pkg_formal"
                _check_version_required "$pkg_version" "$required" "$script_dir" "$package_formal" "$dst_pkg_formal"
            fi
        done
    )"
    if [ "$err_msgs" != "" ]; then
        while read err_msg; do
            comm_log "ERROR" "$err_msg"
        done <<EOF
$err_msgs
EOF
        return 1
    fi
    return 0
}