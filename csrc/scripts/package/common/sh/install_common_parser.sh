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

# run包安装解析公共脚本
# 解析filelist.csv文件，完成目录创建，文件复制，权限设置，文件删除等操作。

# minirc场景下存在使用dash调用该脚本的情况
#export PS4='+ ${FUNCNAME[0]:+${FUNCNAME[0]}():} ${BASH_SOURCE}:${LINENO}: '
#set -x
# 总uninstall.sh文件权限
TOTAL_UNINSTALL_MOD="500"
# db.info文件权限
DB_INFO_MOD="640"
# 包架构
PKG_ARCH="UNKNOWN"
# 复制文件是否并发
PARALLEL=""
# 是否限制并发
PARALLEL_LIMIT=""
# 控制并发的fifo文件路径
PARALLEL_FIFO="none"
COPY_COMMAND="cp -rf"
MOVE_COMMAND="mv -f"

curpath="$(dirname $(readlink -f "${BASH_SOURCE:-$0}"))"

# 导入公共库
common_func_v2_path="${curpath}/common_func_v2.inc"
. "${common_func_v2_path}"

# 导入common_func.inc
# 不一定能source到common_func.inc
# 例如driver包中的install_common_parser.sh与common_func.inc，不在同一个目录下
# 需要子包整改，才将该if语句移除
common_func_path="${curpath}/common_func.inc"
if [ -f "${common_func_path}" ]; then
    . "${common_func_path}"
fi

# 导入script_operator.inc
script_operator_path="${curpath}/script_operator.inc"
if [ -f "${script_operator_path}" ]; then
    . "${script_operator_path}"
fi

# 导入version_cfg.inc
version_cfg_path="${curpath}/version_cfg.inc"
if [ -f "${version_cfg_path}" ]; then
    . "${version_cfg_path}"
fi

# 导入multi_version.inc
multi_version_path="${curpath}/multi_version.inc"
if [ -f "${multi_version_path}" ]; then
    . "${multi_version_path}"
fi

# 导入version_compatiable.inc
version_compatiable_path="${curpath}/version_compatiable.inc"
if [ -f "${version_compatiable_path}" ]; then
    . "${version_compatiable_path}"
fi

# 导入config.inc
config_path="${curpath}/config.inc"
if [ -f "${config_path}" ]; then
    . "${config_path}"
fi

# 导入cold_patch.sh
cold_patch_path="${curpath}/cold_patch.sh"
if [ -f "${cold_patch_path}" ]; then
    . "${cold_patch_path}"
fi

# bash执行时需要展开别名
if [ -n "$BASH_SOURCE" ]; then
    shopt -s expand_aliases
fi

__pkg_in_pkgs() {
    local matched
    __item_in_list "matched" "$@"
    echo "${matched}"
}

__block_in_blocks() {
    __item_in_list "matched" "$@"
    echo "${matched}"
}

# 包列表中移除包
__remove_pkg_in_pkgs() {
    __remove_item_in_list "$@"
}

# 冒泡排序，O(n**2)
__sort_blocks() {
    local blocks="$*"
    local block
    local sorted_blocks=""
    local max
    local i=0

    __length_list "${blocks}" "len_blocks"

    while [ "$i" -lt ${len_blocks} ]; do
        max=""
        for block in ${blocks}; do
            if [ "${block}" \> "${max}" ]; then
                max="${block}"
            fi
        done
        if [ "${sorted_blocks}" = "" ]; then
            sorted_blocks="${max}"
        else
            sorted_blocks="${max} ${sorted_blocks}"
        fi
        blocks="$(__remove_item_in_list "${max}" "${blocks}")"
        i=$((i+1))
    done

    echo "${sorted_blocks}"
}

__unpack_block_item() {
    local line="$1"
    echo "${line}" | tr '|' ' '
}

__pack_block_item() {
    local block_name="$1"
    local block_pkgs_str="$2"
    echo "${block_name}|${block_pkgs_str}"
}

__unpack_block_pkgs() {
    local block_pkgs_str="$1"
    echo "${block_pkgs_str}" | tr ',' ' '
}

__pack_block_pkgs() {
    echo "$*" | tr ' ' ','
}

check_install_path() {
    local install_path="$1"

    if [ "${install_path}" = "" ]; then
        log "ERROR" "install_path is empty!"
        return 1
    fi
}

# 添加pkg对应的block_info
add_pkg_blocks_info() {
    local install_path="$1"
    local pkg="$2"
    shift; shift
    # blocks为待加入db.info的块列表
    local blocks="$*"
    local block_idx_value

    local db_filepath="${install_path}/${PKG_DB_INFO_RELPATH}"
    local db_dirpath="$(dirname "${db_filepath}")"

    local db_filepath_new="${install_path}/${PKG_DB_INFO_RELPATH}~"

    local block_list
    local block_name
    local block_pkgs_str
    local block_pkgs
    local len_block_pkgs
    local block_idx=0
    local len_blocks

    __length_list "${blocks}" "len_blocks"
    if [ ${len_blocks} -eq 0 ]; then
        return 0
    fi

    blocks="$(__sort_blocks "${blocks}")"

    if [ ! -d "${db_dirpath}" ]; then
        mkdir -p "${db_dirpath}"
        if [ $? -ne 0 ]; then
            log "ERROR" "mkdir ${db_dirpath} failed!"
            exit 1
        fi
    fi

    if [ ! -f "${db_filepath}" ]; then
        touch "${db_filepath}"
        if [ $? -ne 0 ]; then
            log "ERROR" "touch ${db_filepath} failed!"
            exit 1
        fi
        change_mod_and_own "${db_filepath}" "${DB_INFO_MOD}" "\$username:\$usergroup" "${INSTALL_FOR_ALL}"
    fi

    rm -f "${db_filepath_new}"
    if [ $? -ne 0 ]; then
        log "ERROR" "delete ${db_filepath_new} failed!"
        exit 1
    fi

    while read line; do
        block_list="$(__unpack_block_item "${line}")"
        __index_list "${block_list}" 0 "block_name"
        __index_list "${block_list}" 1 "block_pkgs_str"

        # 如果pkgs为空，则忽略该条目
        if [ "${block_pkgs_str}" = "" ]; then
            continue
        fi

        block_pkgs="$(__unpack_block_pkgs "${block_pkgs_str}")"
        __length_list "${block_pkgs}" "len_block_pkgs"
        if [ ${len_block_pkgs} -eq 0 ]; then
            continue
        fi

        if [ ${block_idx} -lt ${len_blocks} ]; then
            __index_list "${blocks}" ${block_idx} "block_idx_value"
        fi

        # blocks中的块名小于db中的当前块名，直接添加到db中。
        # 注意不能continue，因为还要处理db的当前条目。
        while [ ${block_idx} -lt ${len_blocks} ] && [ "${block_idx_value}" \< "${block_name}" ]; do
            __pack_block_item "${block_idx_value}" "${pkg}" >> "${db_filepath_new}"
            if [ $? -ne 0 ]; then
                log "ERROR" "write ${db_filepath_new} failed!"
                exit 1
            fi
            block_idx=$((block_idx+1))
            __index_list "${blocks}" ${block_idx} "block_idx_value"
        done

        # 已经没有需要添加的块，保持db的当前条目。
        if [ ${block_idx} -eq ${len_blocks} ]; then
            __pack_block_item "${block_name}" "${block_pkgs_str}" >> "${db_filepath_new}"
            if [ $? -ne 0 ]; then
                log "ERROR" "write ${db_filepath_new} failed!"
                exit 1
            fi
            continue
        fi

        # blocks中的块名大于db中的当前块名，保持db的当前条目。
        if [ "${block_idx_value}" \> "${block_name}" ]; then
            __pack_block_item "${block_name}" "${block_pkgs_str}" >> "${db_filepath_new}"
            if [ $? -ne 0 ]; then
                log "ERROR" "write ${db_filepath_new} failed!"
                exit 1
            fi
            continue
        fi

        # blocks中的块名等于db中的当前块名，将pkg添加到当前条目pkg列表中。
        in_pkgs="$(__pkg_in_pkgs "${pkg}" "${block_pkgs}")"
        if [ "${in_pkgs}" = "true" ]; then
            __pack_block_item "${block_name}" "${block_pkgs_str}" >> "${db_filepath_new}"
            if [ $? -ne 0 ]; then
                log "ERROR" "write ${db_filepath_new} failed!"
                exit 1
            fi
        else
            __pack_block_item "${block_name}" "${block_pkgs_str},${pkg}" >> "${db_filepath_new}"
            if [ $? -ne 0 ]; then
                log "ERROR" "write ${db_filepath_new} failed!"
                exit 1
            fi
        fi
        block_idx=$((block_idx+1))
    done < "${db_filepath}"

    # blocks中剩余的块添加到db中。
    while [ ${block_idx} -lt ${len_blocks} ]; do
        __index_list "${blocks}" ${block_idx} "block_idx_value"
        echo "${block_idx_value}|${pkg}" >> "${db_filepath_new}"
        block_idx=$((block_idx+1))
    done

    cp "${db_filepath_new}" "${db_filepath}"
    if [ $? -ne 0 ]; then
        log "ERROR" "replace ${db_filepath} failed!"
        exit 1
    fi

    rm -f "${db_filepath_new}"
    if [ $? -ne 0 ]; then
        log "ERROR" "delete ${db_filepath_new} failed!"
        exit 1
    fi
}

# 删除block_info中的pkg信息
del_blocks_info_pkg() {
    local install_path="$1"
    local pkg="$2"

    local db_filepath="${install_path}/${PKG_DB_INFO_RELPATH}"
    local db_dirpath="$(dirname "${db_filepath}")"

    local db_filepath_new="${install_path}/${PKG_DB_INFO_RELPATH}~"

    local block_list
    local block_name
    local block_pkgs_str
    local block_pkgs
    local len_block_pkgs

    if [ ! -f "${db_filepath}" ]; then
        return 0
    fi

    rm -f "${db_filepath_new}"
    if [ $? -ne 0 ]; then
        log "ERROR" "delete ${db_filepath_new} failed!"
        exit 1
    fi

    while read line; do
        block_list="$(__unpack_block_item "${line}")"

        __index_list "${block_list}" 0 "block_name"
        __index_list "${block_list}" 1 "block_pkgs_str"

        block_pkgs="$(__unpack_block_pkgs "${block_pkgs_str}")"

        block_pkgs="$(__remove_pkg_in_pkgs "${pkg}" "${block_pkgs}")"

        __length_list "${block_pkgs}" "len_block_pkgs"

        if [ ${len_block_pkgs} -gt 0 ]; then
            block_pkgs_str="$(__pack_block_pkgs "${block_pkgs}")"
            __pack_block_item "${block_name}" "${block_pkgs_str}" >> "${db_filepath_new}"
            if [ $? -ne 0 ]; then
                log "ERROR" "write ${db_filepath_new} failed!"
                exit 1
            fi
        fi
    done < "${db_filepath}"

    # 当前条目如果没有pkg使用，则删除。
    if [ ! -f "${db_filepath_new}" ]; then
        rm -f "${db_filepath}"
        if [ $? -ne 0 ]; then
            log "ERROR" "delete ${db_filepath} failed!"
            exit 1
        fi
    else
        mv -f "${db_filepath_new}" "${db_filepath}"
        if [ $? -ne 0 ]; then
            log "ERROR" "replace ${db_filepath} failed!"
            exit 1
        fi
    fi
    return 0
}

# 准备并且检查软链接路径
prepare_and_check_softlink_path() {
    local softlink_abs="$1"
    local softlink_dir="$(dirname "${softlink_abs}")"

    if [ ! -d "${softlink_dir}" ]; then
        mkdir -p "${softlink_dir}"
    fi

    # 如果目标路径是个软链接，则移除
    if [ -L "${softlink_abs}" ]; then
        rm -f "${softlink_abs}"
        if [ $? -ne 0 ]; then
            log "ERROR" "remove softlink ${softlink_abs} failed! (create relative softlink)"
            exit 1
        fi
    fi

    # 不允许目标路径已经是一个目录，防止软链接到错误的位置。
    if [ -d "${softlink_abs}" ]; then
        log "ERROR" "softlink existed dir ${softlink_abs}!"
        exit 1
    fi
}

get_file_owner_group() {
    local _outvar="$1"
    local _path="$2"
    local _result

    _result="$(stat -c %U "${_path}"):$(stat -c %G "${_path}")"
    eval "${_outvar}=\"${_result}\""
}

# 根据pkg_inner_softlink创建软链接
create_pkg_inner_softlink() {
    local install_path="$1"
    local line="$2"
    local array
    local target
    local pkg_inner_softlink
    local pkg_inner_softlink_list
    local is_abs_path

    __index_list "${line}" 1 "target" 8 "pkg_inner_softlink"

    if [ "${pkg_inner_softlink}" != "NA" ]; then
        create_softlink_by_install_path "${install_path}" "${target}" "${pkg_inner_softlink}"
    fi
    return 0
}

# 处理软链路径为已存在目录的情况
# minirc场景，install_path目录下，存在include目录
# 并且filelist.csv中配置了(x86_64|aarch64)-linux/include目录
# 则将(x86_64|aarch64)-linux/include中的内容，软链接至include目录下
deal_with_existed_dir() {
    local install_path="$1"
    local folder="$2"
    local filelist_path="$3"
    local install_path_regex="^(x86_64|aarch64)-linux/${folder}$"
    local install_sub_path_regex="^(x86_64|aarch64)-linux/${folder}/[^/]+$"

    if [ -L "${install_path}/${folder}" ] || [ ! -d "${install_path}/${folder}" ]; then
        return 0
    fi

    target_path=$(awk -v folder="${folder}" -v install_path_regex="${install_path_regex}" '
        BEGIN{
            FS= ","
        }
        {
            if ($2 != "mkdir") next

            if ($4 !~ install_path_regex) next

            if ($9 != folder) next

            print $4
        }' "${filelist_path}")

    if [ "${target_path}" = "" ]; then
        return 0
    fi

    awk -v folder="${folder}" -v install_path_regex="${install_path_regex}" -v install_sub_path_regex="${install_sub_path_regex}" '
        BEGIN{
            FS = ","; OFS = ","
        }
        {
            if ($2 != "mkdir" && $2 != "copy" && $2 != "move") {
                print $0
                next
            }
            if ($4 ~ install_path_regex) {
                print $1, $2, $3, $4, $5, $6, $7, $8, "NA", $10, $11, $12, $13, $14, $15
                next
            }
            if ($4 !~ install_sub_path_regex) {
                print $0
                next
            }
            softlink = $9
            if (softlink != "NA") {
                print $0
                next
            }

            z = split($4, filepath_list, "/")
            softlink = folder "/" filepath_list[z]
            print $1, $2, $3, $4, $5, $6, $7, $8, softlink, $10, $11, $12, $13, $14, $15
        }' "${filelist_path}" > "${filelist_path}.tmp"
    if [ $? -ne 0 ]; then
        log "ERROR" "modify filelist for ${install_path}/${folder} failed!"
        return 1
    fi

    mv -f "${filelist_path}.tmp" "${filelist_path}"
    if [ $? -ne 0 ]; then
        log "ERROR" "replace filelist for ${install_path}/${folder} failed!"
        return 1
    fi
}

#执行创建目录动作
do_create_dirs() {
    local action="$1"
    local install_type="$2"
    local install_path="$3"
    local filelist_path="$4"
    local package="$5"
    local feature_param="$6"
    local ret

    if [ "${action}" = "resetmod" ]; then
        if [ "${package}" != "" ]; then
            del_blocks_info_pkg "${install_path}" "${package}"
            ret="$?" && [ $ret -ne 0 ] && return $ret

            create_stash_mod "${install_path}"
            ret="$?" && [ $ret -ne 0 ] && return $ret

            foreach_filelist "NA" "reset_mod_dirs_with_stash_mod" "${install_type}" "${install_path}" "mkdir" "${filelist_path}" "${feature_param}" "no" "normal"
            ret="$?" && [ $ret -ne 0 ] && return $ret
        else
            foreach_filelist "NA" "reset_mod_dirs" "${install_type}" "${install_path}" "mkdir" "${filelist_path}" "${feature_param}" "no" "normal"
            ret="$?" && [ $ret -ne 0 ] && return $ret
        fi
        # 设置copy_entity路径的权限，解决删除时权限问题
        # filter_by_blocks过滤，防止变更不删除的文件（目录）的权限
        foreach_filelist "filter_by_blocks" "reset_mod_dirs_recursive" "${install_type}" "${install_path}" "copy_entity" "${filelist_path}" "${feature_param}" "no" "normal"
        ret="$?" && [ $ret -ne 0 ] && return $ret
    elif [ "${action}" = "all" ] || [ "${action}" = "mkdir" ]; then
        deal_with_existed_dir "$install_path" "include" "$filelist_path"
        ret="$?" && [ $ret -ne 0 ] && return $ret

        deal_with_existed_dir "$install_path" "lib64" "$filelist_path"
        ret="$?" && [ $ret -ne 0 ] && return $ret

        # 先重置目录权限配置，防止软链接时缺少权限
        foreach_filelist "NA" "reset_mod_dirs" "$install_type" "$install_path" "mkdir" "$filelist_path" "${feature_param}" "no" "normal"
        ret="$?" && [ $ret -ne 0 ] && return $ret

        # 重置copy_entity权限，防止块复用场景，复制时缺少权限
        foreach_filelist "NA" "reset_mod_dirs_recursive" "$install_type" "$install_path" "copy_entity" "$filelist_path" "${feature_param}" "no" "normal"
        ret="$?" && [ $ret -ne 0 ] && return $ret

        foreach_filelist "NA" "create_dirs" "$install_type" "$install_path" "mkdir" "$filelist_path" "${feature_param}" "no" "normal"
        ret="$?" && [ $ret -ne 0 ] && return $ret

        foreach_filelist "filter_by_pkg_inner_softlink" "create_pkg_inner_softlink" "$install_type" "$install_path" "mkdir" "$filelist_path" "${feature_param}" "no" "normal"
        ret="$?" && [ $ret -ne 0 ] && return $ret
    else
        log "ERROR" "action wrong! action is ${action}"
        return 1
    fi

    return 0
}

# 拷贝文件
copy_file() {
    local install_path="$1"
    local source="$2"
    local target="$3"
    local softlink="$4"
    local pkg_inner_softlink="$5"
    local target_abs
    local target_dir

    __set_abs_path "${install_path}" "${target}" "target_abs"
    target_dir="$(dirname "${target_abs}")"

    if [ ! -e "${source}" ]; then
        log "ERROR" "copy file source file ${source} doesn't exist!"
        return 1
    fi

    ${COPY_COMMAND} "${source}" "${target_dir}"
    if [ $? -ne 0 ]; then
        log "ERROR" "${source} copy failed!"
        return 1
    fi
    if [ "${softlink}" != "NA" ]; then
        create_softlink_by_install_path "${install_path}" "${target}" "${softlink}"
        ret="$?" && [ ${ret} -ne 0 ] && return ${ret}
    fi
    if [ "${pkg_inner_softlink}" != "NA" ]; then
        create_softlink_by_install_path "${install_path}" "${target}" "${pkg_inner_softlink}"
        ret="$?" && [ ${ret} -ne 0 ] && return ${ret}
    fi
    return 0
}

# 移动文件
move_file() {
    local install_path="$1"
    local source="$2"
    local target="$3"
    local softlink="$4"
    local pkg_inner_softlink="$5"
    local target_abs
    local target_dir

    __set_abs_path "${install_path}" "${target}" "target_abs"
    target_dir="$(dirname "${target_abs}")"

    if [ ! -e "${source}" ] && [ ! -L "${source}" ]; then
        log "ERROR" "move file source file ${source} doesn't exist!"
        return 1
    fi

    ${MOVE_COMMAND} "${source}" "${target_dir}"
    if [ $? -ne 0 ]; then
        log "ERROR" "${source} move failed!"
        return 1
    fi
    if [ "${softlink}" != "NA" ]; then
        create_softlink_by_install_path "${install_path}" "${target}" "${softlink}"
        ret="$?" && [ ${ret} -ne 0 ] && return ${ret}
    fi
    if [ "${pkg_inner_softlink}" != "NA" ]; then
        create_softlink_by_install_path "${install_path}" "${target}" "${pkg_inner_softlink}"
        ret="$?" && [ ${ret} -ne 0 ] && return ${ret}
    fi
    return 0
}

# 复制文件
copy_files() {
    local install_path="$1"
    local line="$2"
    local exec_params="$3"
    local src_dir="$4"
    local src src_abs target target_abs softlink configurable pkg_inner_softlink tmpdir action
    local ret

    __index_list "${line}" 0 "src" 1 "target" 4 "softlink" 5 "configurable" 8 "pkg_inner_softlink"
    if [ "${src_dir}" = "--move" ]; then
        src_abs="${src}"
        action="move_file"
    elif [ "${src_dir}" != "" ]; then
        __set_abs_path "${src_dir}" "${src}" "src_abs"
        action="copy_file"
    else
        src_abs="${src}"
        action="copy_file"
    fi
    __set_abs_path "${install_path}" "${target}" "target_abs"

    tmpdir="$(dirname "${target_abs}")"
    if [ ! -d "${tmpdir}" ]; then
        mkdir -p "${tmpdir}"
    fi

    # 如果目标文件已经存在，而且是配置文件，则不执行覆盖操作
    if [ -e "${target_abs}" ] && [ "${configurable}" = "TRUE" ]; then
        return 0
    fi

    # 源文件不是软链，并且目标文件是软链
    if [ ! -L "${src_abs}" ] && [ -L "${target_abs}" ] ; then
        rm -f "${target_abs}"
        log "WARNING" "${target_abs} is an existing softlink in copy files, deleted."
    fi

    exec_with_param "$exec_params" "$action" "${install_path}" "${src_abs}" "${target}" "${softlink}" "${pkg_inner_softlink}"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}
    return 0
}

__set_total_uninstall_path() {
    local install_path="$1"
    local varname="$2"

    eval "${varname}=\"${install_path}/cann_uninstall.sh\""
}

# 是否为hilinux环境
__is_hilinux() {
    local varname="$1"

    which lsattr > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        eval "${varname}=\"true\""
    else
        eval "${varname}=\"false\""
    fi
}

# 移除文件上的不可修改权限
__remove_immutable() {
    local file="$1"
    local is_hilinux ret

    # 文件不存在则退出。
    if [ ! -f "${file}" ]; then
        return 0
    fi

    __is_hilinux "is_hilinux"
    if [ "${is_hilinux}" = "true" ]; then
        return 0
    fi

    attr="$(lsattr "${file}" | cut -d' ' -f1 | grep -o "i")"
    if [ "${attr}" != "" ]; then
        chattr -i "${file}"
        ret="$?" && [ $ret -ne 0 ] && return $ret
    fi
    return 0
}

# 创建总uninstall.sh脚本
__create_uninstall() {
    local file="$1"

    if [ ! -f "${file}" ]; then
        cat > ${file} <<EOF
#!/bin/sh

SHELL_DIR="\$(dirname "\${BASH_SOURCE:-\$0}")"
INSTALL_PATH="\$(cd "\${SHELL_DIR}" && pwd)"
TOTAL_RET="0"

uninstall_package() {
    local path="\$1"
    local cur_date ret

    if [ ! -d "\${INSTALL_PATH}/\${path}" ]; then
        cur_date=\$(date +"%Y-%m-%d %H:%M:%S")
        echo "[\$cur_date] [ERROR]: \${INSTALL_PATH}/\${path}: No such file or directory"
        TOTAL_RET="1"
        return 1
    fi

    cd "\${INSTALL_PATH}/\${path}"
    ./uninstall.sh
    ret="\$?" && [ \${ret} -ne 0 ] && TOTAL_RET="1"
    return \${ret}
}

if [ ! "\$*" = "" ]; then
    cur_date=\$(date +"%Y-%m-%d %H:%M:%S")
    echo "[\$cur_date] [ERROR]: \$*, parameter is not supported."
    exit 1
fi

exit \${TOTAL_RET}
EOF
    fi
}

# 添加uninstall.sh文件中的uninstall_package函数调用
__add_uninstall_package() {
    local file="$1"
    local script_dirpath="$2"

    sed -i "/^exit /i uninstall_package \"${script_dirpath}\"" "${file}"
}

# 删除uninstall.sh文件中的uninstall_package函数调用
__remove_uninstall_package() {
    local file="$1"
    local script_dirpath="$2"
    local path_regex

    path_to_regex "path_regex" "${script_dirpath}"

    if [ -f "${file}" ]; then
        sed -i "/uninstall_package \"${path_regex}\"/d" "${file}"
        if [ $? -ne 0 ]; then
            log "ERROR" "remove ${file} uninstall_package command failed!"
            return 1
        fi
    fi
    return 0
}

# 删除uninstall.sh文件，如果已经没有uninstall_package调用
__remove_uninstall_file_if_no_content() {
    local file="$1"
    local num

    if [ ! -f "${file}" ]; then
        return 0
    fi

    # 注意uninstall_package后有一个空格，表示函数调用
    num=$(grep "^uninstall_package " ${file} | wc -l)
    if [ ${num} -eq 0 ]; then
        rm -f "${file}" > /dev/null 2>&1
        if [ $? -ne 0 ]; then
            log "WARNING" "Delete file:${file} failed, please delete it by yourself."
        fi
    fi
    return 0
}

# 向cann_uninstall.sh文件中添加uninstall_package命令
add_cann_uninstall_script_dir() {
    local install_path="$1"
    local script_dir="$2"
    local username="$3"
    local usergroup="$4"
    local install_for_all="$5"
    local oldmod="" ret

    __set_total_uninstall_path "${install_path}" "total_uninstall_path"

    if [ -f "${install_path}/${script_dir}/uninstall.sh" ]; then
        if [ -f "${total_uninstall_path}" ]; then
            get_file_mod "oldmod" "${total_uninstall_path}"
        else
            __create_uninstall "${total_uninstall_path}"
        fi
        ret="$?" && [ $ret -ne 0 ] && return $ret

        __remove_immutable "${total_uninstall_path}"
        ret="$?" && [ $ret -ne 0 ] && return $ret

        change_own "${total_uninstall_path}" "${username}:${usergroup}"
        ret="$?" && [ $ret -ne 0 ] && return $ret

        change_mod "${total_uninstall_path}" "${SETENV_WRITEABLE_MOD}" ""
        ret="$?" && [ $ret -ne 0 ] && return $ret

        __add_uninstall_package "${total_uninstall_path}" "${script_dir}"
        ret="$?" && [ $ret -ne 0 ] && return $ret

        if [ "${oldmod}" = "" ]; then
            change_mod "${total_uninstall_path}" "${TOTAL_UNINSTALL_MOD}" "${install_for_all}"
        else
            change_mod "${total_uninstall_path}" "${oldmod}" ""
        fi
        ret="$?" && [ $ret -ne 0 ] && return $ret
    fi

    return 0
}

# 向cann_uninstall.sh文件中添加子包命令
add_cann_uninstall_package() {
    local install_path="$1"
    local package="$2"
    local username="$3"
    local usergroup="$4"
    local install_for_all="$5"
    local ret package_dirpath script_dir

    get_package_dirpath "package_dirpath" "${package}"
    script_dir="${package_dirpath}/script"

    # opp_kernel包存在同时安装多种芯片包的场景
    # 确保cann_uninstall.sh脚本中只有一个opp_kernel的uninstall_package
    del_cann_uninstall_script_dir "${install_path}" "${script_dir}"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    add_cann_uninstall_script_dir "${install_path}" "${script_dir}" "${username}" "${usergroup}" "${install_for_all}"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    return 0
}

# cann_uninstall.sh文件中添加命令（命令入口）
do_add_cann_uninstall() {
    local install_path="$1"
    local script_dir="$2"
    local username="$3"
    local usergroup="$4"
    local install_for_all="$5"
    local ret

    check_param_not_empty "install_path" "need set package parameter in add cann uninstall!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    check_param_not_empty "script_dir" "need set script_dir parameter in add cann uninstall!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    check_param_not_empty "username" "need set username parameter in add cann uninstall!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    check_param_not_empty "usergroup" "need set usergroup parameter in add cann uninstall!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    # 删除cann_uninstall.sh文件中已存在的uninstall_package命令
    del_cann_uninstall_script_dir "${install_path}" "${script_dir}"
    add_cann_uninstall_script_dir "${install_path}" "${script_dir}" "${username}" "${usergroup}" "${install_for_all}"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    return 0
}

# 删除cann_uninstall.sh文件中uninstall_package命令
del_cann_uninstall_script_dir() {
    local install_path="$1"
    local script_dir="$2"
    local oldmod="" ret

    __set_total_uninstall_path "${install_path}" "total_uninstall_path"

    if [ -f "${total_uninstall_path}" ]; then
        __remove_immutable "${total_uninstall_path}"
        ret="$?" && [ $ret -ne 0 ] && return $ret

        get_file_mod "oldmod" "${total_uninstall_path}"
        ret="$?" && [ $ret -ne 0 ] && return $ret

        change_mod "${total_uninstall_path}" "${SETENV_WRITEABLE_MOD}" ""
        ret="$?" && [ $ret -ne 0 ] && return $ret

        __remove_uninstall_package "${total_uninstall_path}" "${script_dir}"
        ret="$?" && [ $ret -ne 0 ] && return $ret

        change_mod "${total_uninstall_path}" "${oldmod}" ""
        ret="$?" && [ $ret -ne 0 ] && return $ret

        __remove_uninstall_file_if_no_content "${total_uninstall_path}"
        ret="$?" && [ $ret -ne 0 ] && return $ret
    fi
    return 0
}

# 删除cann_uninstall.sh文件中子包命令
del_cann_uninstall_package() {
    local install_path="$1"
    local package="$2"
    local ret package_dirpath script_dir

    get_package_dirpath "package_dirpath" "${package}"
    script_dir="${package_dirpath}/script"

    del_cann_uninstall_script_dir "${install_path}" "${script_dir}"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    return 0
}

# cann_uninstall.sh文件中删除命令（命令入口）
do_del_cann_uninstall() {
    local install_path="$1"
    local script_dir="$2"
    local ret

    check_param_not_empty "install_path" "need set package parameter in del cann uninstall!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    check_param_not_empty "script_dir" "need set script_dir parameter in del cann uninstall!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    del_cann_uninstall_script_dir "${install_path}" "${script_dir}"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    return 0
}

# 删除ascend_install.info文件
del_ascend_install_info() {
    rm -f "$curpath/../ascend_install.info"
}

# 执行拷贝动作
do_copy_files() {
    local install_type="$1"
    local install_path="$2"
    local filelist_path="$3"
    local package="$4"
    local feature_param="$5"
    local total_uninstall_path exec_mode ret

    if [ "$PARALLEL" = "true" ]; then
        exec_mode="concurrency"
    else
        exec_mode="normal"
    fi
    if [ "${USE_MOVE}" = "true" ]; then
        foreach_filelist "NA" "copy_files" "$install_type" "$install_path" "move" "$filelist_path" "${feature_param}" \
                         "no" "$exec_mode" "--move"
        foreach_filelist "NA" "copy_files" "$install_type" "$install_path" "copy copy_entity" "$filelist_path" "${feature_param}" \
                         "no" "$exec_mode"
    else
        foreach_filelist "NA" "copy_files" "$install_type" "$install_path" "copy copy_entity move" "$filelist_path" "${feature_param}" \
                         "no" "$exec_mode"
    fi
    ret="$?" && [ $ret -ne 0 ] && return $ret

    if [ "${package}" != "" ] && [ "${SET_CANN_UNINSTALL}" = "y" ]; then
        add_cann_uninstall_package "${install_path}" "${package}" "${USERNAME}" "${USERGROUP}" "${INSTALL_FOR_ALL}"
    fi
}

# 修改文件的权限和属组
change_mod_and_own_files() {
    local install_path="$1"
    local line="$2"
    local exec_params="$3"
    local target mod own is_abs_path

    __index_list "${line}" 1 "target" 2 "mod" 3 "own"
    __check_abs_path "${target}"
    if [ "${is_abs_path}" != "true" ]; then
        target="${install_path}/${target}"
    fi
    if [ -d "${target}" ]; then
        return 0
    fi
    # 只处理文件，没有处理文件的软链接
    exec_with_param "$exec_params" change_mod_and_own "${target}" "${mod}" "${own}" "${INSTALL_FOR_ALL}" "false"
}

# 递归修改文件的权限和属组
change_mod_and_own_files_recursive() {
    local install_path="$1"
    local line="$2"
    local exec_params="$3"
    local target mod own is_abs_path

    __index_list "${line}" 1 "target" 2 "mod" 3 "own"
    __check_abs_path "${target}"
    if [ "${is_abs_path}" != "true" ]; then
        target="${install_path}/${target}"
    fi
    if [ ! -e "${target}" ]; then
        return 0
    fi

    exec_with_param "$exec_params" change_mod_and_own "${target}" "${mod}" "${own}" "${INSTALL_FOR_ALL}" "true"
}

# filelist中的使用到的blocks，添加到db.info中
add_filelist_blocks_info() {
    local install_type="$1"
    local install_path="$2"
    local filelist_path="$3"
    local feature_param="$4"
    local package="$5"
    local blocks=""

    parse_filelist "${install_type}" "copy copy_entity del mkdir move" "${filelist_path}" "${feature_param}" "NA" ""
    filelist="$(echo "${filelist}" | cut -d' ' -f8 | sort | uniq)"

    while read line; do
        block_name="${line}"
        if [ "${block_name}" = "" ]; then
            continue
        fi
        if [ "${blocks}" = "" ]; then
            blocks="${block_name}"
        else
            blocks="${blocks} ${block_name}"
        fi
    done << EOF
${filelist}
EOF
    add_pkg_blocks_info "${install_path}" "${package}" "${blocks}"
}

# 修改文件和目录的权限
do_chmod_file_dir() {
    local install_type="$1"
    local install_path="$2"
    local filelist_path="$3"
    local feature_param="$4"
    local package="$5"
    local ret

    foreach_filelist "NA" "change_mod_and_own_files" "$install_type" "$install_path" "copy del move" "$filelist_path" "${feature_param}" "no" "concurrency"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    foreach_filelist "NA" "change_mod_and_own_files_recursive" "$install_type" "$install_path" "copy_entity" "$filelist_path" "${feature_param}" "no" "concurrency"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    foreach_filelist "NA" "change_mod_and_own_dirs" "$install_type" "$install_path" "mkdir" "$filelist_path" "${feature_param}" "reverse" "normal"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    if [ "${package}" != "" ]; then
        add_filelist_blocks_info "$install_type" "$install_path" "$filelist_path" "${feature_param}" "$package"
        ret="$?" && [ $ret -ne 0 ] && return $ret
    fi

    return 0
}

# 删除安装生成的pkg_inner_softlinks
remove_install_pkg_inner_softlinks() {
    local install_path="$1"
    local line="$2"
    local target
    local pkg_inner_softlink
    local pkg_inner_softlink_list
    local pkg_inner_softlink_abs
    local ret

    __index_list "${line}" 1 "target" 8 "pkg_inner_softlink"
    if [ "${target}" != "NA" ]; then
        if [ "${pkg_inner_softlink}" != "NA" ]; then
            remove_softlinks "${install_path}" "${pkg_inner_softlink}"
            ret="$?" && [ ${ret} -ne 0 ] && return ${ret}
        fi
    fi
    return 0
}

# 删除安装文件，入参为$1: install_type, $2: install_path $3:filelist_path
remove_install_files() {
    local install_path="$1"
    local line="$2"
    local exec_params="$3"
    local target softlink configurable hash_value is_abs_path

    __index_list "${line}" 1 "target" 4 "softlink" 5 "configurable" 6 "hash_value"

    if [ "${target}" != "NA" ]; then
        __check_abs_path "${target}"
        if [ "${is_abs_path}" != "true" ]; then
            target="${install_path}/${target}"
        fi
        if [ -d "${target}" ] && [ ! -L "${target}" ]; then
            return 0
        fi
        if [ "${softlink}" != "NA" ]; then
            __check_abs_path "${softlink}"
            if [ "${is_abs_path}" != "true" ]; then
                softlink="${install_path}/${softlink}"
            fi
        fi
        # 配置文件不删除
        if [ "${configurable}" = "TRUE" ]; then
            echo "${hash_value} ${target}" | sha256sum --check > /dev/null 2>&1
            if [ $? -ne 0 ]; then
                log "WARNING" "${target} user configuration file has been modified, skip deleting."
                return 0
            fi
        fi
        exec_with_param "$exec_params" remove_file "${target}" "${softlink}"
    fi
    return 0
}

# 递归删除安装文件
remove_install_files_recursive() {
    local install_path="$1"
    local line="$2"
    local exec_params="$3"
    local target configurable hash_value is_abs_path

    __index_list "${line}" 1 "target" 5 "configurable" 6 "hash_value"

    if [ "${target}" != "NA" ]; then
        __check_abs_path "${target}"
        if [ "${is_abs_path}" != "true" ]; then
            target="${install_path}/${target}"
        fi
        # 配置文件不删除
        if [ "${configurable}" = "TRUE" ]; then
            echo "${hash_value} ${target}" | sha256sum --check > /dev/null 2>&1
            if [ $? -ne 0 ]; then
                log "WARNING" "${target} user configuration file has been modified, skip deleting."
                return 0
            fi
        fi
        exec_with_param "$exec_params" remove_dir_icp "${target}"
    fi
    return 0
}

# 删除安装文件夹生成的软链接
remove_install_softlink() {
    local install_path="$1"
    local line="$2"
    local target
    local softlinks_str
    local is_abs_path

    __index_list "${line}" 1 "target" 4 "softlinks_str"

    if [ "$target" != "NA" ]; then
        if [ "${softlinks_str}" != "NA" ]; then
            remove_softlinks "${install_path}" "${softlinks_str}"
            ret="$?" && [ ${ret} -ne 0 ] && return ${ret}
        fi
    fi
    return 0
}

# 删除安装文件与目录等
do_remove() {
    local install_type="$1"
    local install_path="$2"
    local filelist_path="$3"
    local package="$4"
    local feature_param="$5"
    local tmp_filelist_path
    local total_uninstall_path
    local oldmod
    local func_before_remove
    local func_after_remove_01
    local func_after_remove_02
    local ret

    get_tmp_file tmp_filelist_path "filelist"
    cp -f "$filelist_path" "$tmp_filelist_path"

    if [ $? -ne 0 ]; then
        log "ERROR" "cp -f $filelist_path $tmp_filelist_path failed!"
        return 1
    fi

    if [ "${package}" != "" ]; then
        # root帐户--uninstall时，存在不调用restoremod，直接调用remove的场景。（toolkit包）
        func_before_remove="del_blocks_info_pkg \"${install_path}\" \"${package}\""
        if [ -f "${install_path}/${STASH_MOD_PATH}" ]; then
            # 删除文件后，恢复目录权限配置
            func_after_remove_01="foreach_stashmod \"restore_stash_mod\" \"${install_path}\" \"reverse\""
            func_after_remove_02="remove_stash_mod \"${install_path}\""
        fi

        del_cann_uninstall_package "${install_path}" "${package}"
        ret="$?" && [ $ret -ne 0 ] && return $ret

        if [ "$REMOVE_INSTALL_INFO" = "y" ]; then
            del_ascend_install_info
        fi
    fi

    eval "${func_before_remove}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    # del元素不会出现pkg_inner_softlink
    foreach_filelist "filter_by_pkg_inner_softlink" "remove_install_pkg_inner_softlinks" "$install_type" "$install_path" "mkdir copy copy_entity move" "$tmp_filelist_path" "${feature_param}" "no" "normal"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    foreach_filelist "filter_by_blocks" "remove_install_softlink" "$install_type" "$install_path" "mkdir copy copy_entity move" "$tmp_filelist_path" "${feature_param}" "no" "normal"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    foreach_filelist "filter_by_blocks" "remove_install_files" "$install_type" "$install_path" "copy del move" "$tmp_filelist_path" "${feature_param}" "no" "concurrency"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    foreach_filelist "filter_by_blocks" "remove_install_files_recursive" "$install_type" "$install_path" "copy_entity" "$tmp_filelist_path" "${feature_param}" "no" "concurrency"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    foreach_filelist "filter_by_blocks" "remove_install_dirs" "$install_type" "$install_path" "mkdir" "$tmp_filelist_path" "${feature_param}" "reverse" "normal"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    eval "${func_after_remove_01}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    eval "${func_after_remove_02}"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    rm -f "$tmp_filelist_path"
    ret="$?" && [ $ret -ne 0 ] && return $ret

    return 0
}

# 打印安装信息
print_install_content() {
    local install_type="$1"
    local install_path="$2"
    local operate_type="$3"
    local filelist_path="$4"
    local feature_param="$5"
    local blocks

    get_blocks_info "blocks" "$install_path"
    parse_filelist "$install_type" "$operate_type" "$filelist_path" "$feature_param" "filter_by_blocks" "$blocks"

    # bash与dash的echo有差异，需要使用系统的/bin/echo
    /bin/echo "$filelist"
    exit 0
}

# 包名是否存在于db.info
pkg_in_dbinfo() {
    local install_path="$1"
    local pkg="$2"

    local db_filepath="${install_path}/${PKG_DB_INFO_RELPATH}"

    if [ -z "${install_path}" ]; then
        echo "false"
        exit 1
    fi

    if [ -z "${pkg}" ]; then
        echo "false"
        exit 2
    fi

    if [ ! -f "${db_filepath}" ]; then
        echo "false"
        return 0
    fi

    awk -v input_pkg="${pkg}" '
        BEGIN {
            FS = "|"
            found = "false"
        }
        {
            split($2, pkg_list, ",")

            for (i in pkg_list) {
                if (input_pkg == pkg_list[i]) {
                    found = "true"
                    exit
                }
            }
        }
        END {
            print found
        }' "${db_filepath}"
}

# 展开自定义选项
expand_custom_options() {
    local _outvar="$1"
    local _custom_options="$2"

    eval "${_outvar}=\"$(echo "${_custom_options}" | tr "," " ")\""
}

# 调用子包自定义脚本
package_custom_script() {
    local script_name="$1"
    local install_path="$2"
    local version_dir="$3"
    local custom_options="$4"
    local ret install_options=""

    if [ ! -f "${curpath}/${script_name}" ]; then
        return 0
    fi

    install_options="--install-path=${install_path}"

    if [ "${version_dir}" != "" ]; then
        install_options="${install_options} --version-dir=${version_dir}"
    fi

    expand_custom_options "custom_options" "${custom_options}"

    chmod u+x ${curpath}/${script_name}
    ${curpath}/${script_name} ${install_options} ${custom_options}
    ret="$?" && [ $ret -ne 0 ] && return $ret

    return 0
}

# 调用子包自定义安装脚本
package_custom_install() {
    local package="$1"
    local install_path="$2"
    local version_dir="$3"
    local custom_options="$4"
    local ret

    package_custom_script "${package}_custom_install.sh" "${install_path}" "${version_dir}" "${custom_options}"
    check_ret_error "$?" "Run ${package} custom install failed!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    return 0
}

# 调用子包自定义卸载脚本
package_custom_uninstall() {
    local package="$1"
    local install_path="$2"
    local version_dir="$3"
    local custom_options="$4"
    local ret

    package_custom_script "${package}_custom_uninstall.sh" "${install_path}" "${version_dir}" "${custom_options}"
    check_ret_error "$?" "Run ${package} custom uninstall failed!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    return 0
}

# 版本安装流程函数
version_install() {
    local install_type="$1"
    local install_path="$2"
    local filelist_path="$3"
    local package="$4"
    local feature_param="$5"
    local version_dir="$6"
    local username="$7"
    local usergroup="$8"
    local setenv="$9"
    local is_upgrade="${10}"
    local docker_root="${11}"
    local custom_options="${12}"
    local is_simple="${13}"
    local install_path_full="" ret
    local package_real

    if [ "${version_dir}" != "" ]; then
        install_path_full="${install_path}/${version_dir}"
    else
        install_path_full="${install_path}"
    fi

    if [ "${is_simple}" = "y" ]; then
        package_real=""
    else
        package_real="${package}"
    fi

    # 创建目录
    do_create_dirs "mkdir" "${install_type}" "${install_path_full}" "${filelist_path}" "${package_real}" "${feature_param}"
    if [ $? -ne 0 ]; then
        log "ERROR" "failed to create folder."
        return 1
    fi

    # 拷贝目录与文件
    do_copy_files "${install_type}" "${install_path_full}" "${filelist_path}" "${package_real}" "${feature_param}"
    if [ $? -ne 0 ]; then
        log "ERROR" "failed to copy files."
        return 1
    fi

    if [ "${is_simple}" != "y" ]; then
        # set env
        add_setenv "${install_path_full}" "${package_real}" "${setenv}" "${username}" "${usergroup}" "false" "${docker_root}"
        ret="$?" && [ $ret -ne 0 ] && return $ret

        # 调用组件自定义安装流程
        package_custom_install "${package_real}" "${install_path}" "${version_dir}" "${custom_options}"
        ret="$?" && [ $ret -ne 0 ] && return $ret
    fi

    # 文件与目录赋权
    do_chmod_file_dir "${install_type}" "${install_path_full}" "${filelist_path}" "${feature_param}" "${package_real}"
    if [ $? -ne 0 ]; then
        log "ERROR" "failed to chown files."
        return 1
    fi

    return 0
}

# 版本卸载流程函数
version_uninstall() {
    local install_type="$1"
    local install_path="$2"
    local filelist_path="$3"
    local package="$4"
    local feature_param="$5"
    local version_dir="$6"
    local username="$7"
    local docker_root="$8"
    local custom_options="$9"
    local is_simple="${10}"
    local install_path_full="" ret total_ret=0
    local package_real

    if [ "${version_dir}" != "" ]; then
        install_path_full="${install_path}/${version_dir}"
    else
        install_path_full="${install_path}"
    fi

    if [ "${is_simple}" = "y" ]; then
        package_real=""
    else
        package_real="${package}"
    fi

    # 恢复权限
    do_create_dirs "resetmod" "${install_type}" "${install_path_full}" "${filelist_path}" "${package_real}" "${feature_param}"
    if [ $? -ne 0 ]; then
        log "ERROR" "failed to resetmod chmod."
        return 1
    fi

    if [ "${is_simple}" != "y" ]; then
        # unset env
        del_setenv "${install_path_full}" "${package_real}" "${username}" "${docker_root}"
        ret="$?" && [ $ret -ne 0 ] && return $ret

        # 调用组件自定义卸载流程，失败流程不中断
        package_custom_uninstall "${package_real}" "${install_path}" "${version_dir}" "${custom_options}"
        ret="$?" && [ $ret -ne 0 ] && total_ret="$ret"
    fi

    # 删除文件和目录
    do_remove "${install_type}" "${install_path_full}" "${filelist_path}" "${package_real}" "${feature_param}"
    if [ $? -ne 0 ]; then
        log "ERROR" "failed to remove files and dirs."
        return 1
    fi

    return ${total_ret}
}

# 版本安装函数
do_install() {
    local install_type="$1"
    local install_path="$2"
    local filelist_path="$3"
    local package="$4"
    local feature_param="$5"
    local docker_root="$6"
    local is_simple="$7"
    local install_path_real ret 

    check_param_not_empty "package" "need set package parameter in install!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    _package_to_log_pkg_name "LOG_PKG_NAME" "${package}"
    check_ret_error "$?" "Set log package name failed in install!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    expand_version_file

    # 获取filelist.csv文件真实路径
    get_realpath "filelist_path" "${filelist_path}"
    # 获取install_path真实路径
    get_realpath "install_path" "${install_path}"

    if [ "${docker_root}" != "" ]; then
        if [ "${WITH_DOCKER_ROOT_PREFIX}" = "y" ]; then
            install_path_real="${install_path}"
        else
            install_path_real="${docker_root}${install_path}"
        fi
    else
        install_path_real="${install_path}"
    fi

    if [ "${VERSION}" != "" ] && [ "${VERSION_DIR}" != "" ]; then
        multi_version_install "${install_type}" "${install_path_real}" "${filelist_path}" "${package}" "${feature_param}" \
            "${VERSION}" "${VERSION_DIR}" "${USERNAME}" "${USERGROUP}" "${SETENV}" "${IS_UPGRADE}" "${docker_root}" "${CUSTOM_OPTIONS}" \
            "${INSTALL_FOR_ALL}"
        ret="$?" && [ $ret -ne 0 ] && return $ret
    else
        version_install "${install_type}" "${install_path_real}" "${filelist_path}" "${package}" "${feature_param}" \
            "" "${USERNAME}" "${USERGROUP}" "${SETENV}" "${IS_UPGRADE}" "${docker_root}" "${CUSTOM_OPTIONS}" "${is_simple}"
        ret="$?" && [ $ret -ne 0 ] && return $ret
    fi
    return 0
}

# 版本卸载函数
do_uninstall() {
    local install_type="$1"
    local install_path="$2"
    local filelist_path="$3"
    local package="$4"
    local feature_param="$5"
    local docker_root="$6"
    local is_simple="$7"
    local install_path_real ret

    check_param_not_empty "package" "need set package parameter in uninstall!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    _package_to_log_pkg_name "LOG_PKG_NAME" "${package}"
    check_ret_error "$?" "Set log package name failed in uninstall!"
    ret="$?" && [ ${ret} -ne 0 ] && return ${ret}

    expand_version_file

    # 获取filelist.csv文件真实路径
    get_realpath "filelist_path" "${filelist_path}"
    # 获取install_path真实路径
    get_realpath "install_path" "${install_path}"

    if [ "${docker_root}" != "" ]; then
        if [ "${WITH_DOCKER_ROOT_PREFIX}" = "y" ]; then
            install_path_real="${install_path}"
        else
            install_path_real="${docker_root}${install_path}"
        fi
    else
        install_path_real="${install_path}"
    fi

    if [ "${VERSION}" != "" ] && [ "${VERSION_DIR}" != "" ]; then
        multi_version_uninstall "${install_type}" "${install_path_real}" "${filelist_path}" "${package}" "${feature_param}" \
            "${VERSION}" "${VERSION_DIR}" "${USERNAME}" "${USERGROUP}" "${docker_root}" "${CUSTOM_OPTIONS}" "${IS_RECREATE_SOFTLINK}"
        ret="$?" && [ $ret -ne 0 ] && return $ret
    else
        version_uninstall "${install_type}" "${install_path_real}" "${filelist_path}" "${package}" "${feature_param}" \
            "" "${USERNAME}" "${docker_root}" "${CUSTOM_OPTIONS}" "${is_simple}"
        ret="$?" && [ $ret -ne 0 ] && return $ret
    fi
    return 0
}

help_info() {
    echo "Usage:"
    echo ""
    echo "  ----------------------------------------------------------------------------------------------------"
    echo ""
    echo "  $0 {--install,--uninstall,--mkdir,--makedir,--copy,--chmoddir,--restoremod,--remove}"
    echo ""
    echo "    --install    : Install package."
    echo "    --uninstall  : Uninstall package."
    echo "    --mkdir      : (Deprecated) Use --install instead. Create the install directories."
    echo "    --makedir    : (Deprecated) Use --install instead. Create the install directories."
    echo "    --copy       : (Deprecated) Use --install instead. Copy install files."
    echo "    --chmoddir   : (Deprecated) Use --install instead. Set installed file's right."
    echo "    --restoremod : (Deprecated) Use --uninstall instead. Restore directories right."
    echo "    --remove     : (Deprecated) Use --uninstall instead. Remove installed files."
    echo ""
    echo "  Group options:"
    echo "    --install-path=<install-path> : Specify install path."
    echo "    --username=<username> : Specify install username."
    echo "    --usergroup=<usergroup> : Specify install usergroup."
    echo "    --install_for_all : Install for all users."
    echo "                        Usually specified in root user install scene."
    echo "    --docker-root=<docker-root> : Specify docker root path."
    echo "                                  Install path is not contained docker root path."
    echo "    --chip=<chip> : Specify chip."
    echo "                    Install files which chip matched or chip is all."
    echo "    --feature=<feature> : Specify feature. Default is all."
    echo "                          Install files which feature matched or feature is comm."
    echo "    --feature-exclude-all : Switch feature mode."
    echo "                            Install files which feature matched."
    echo ""
    echo "  $0 --install [options] <install_type> <install_path> <filelist_path>"
    echo ""
    echo "  Command options:"
    echo "    --package=<package> : Specify package name."
    echo "    --version=<version> : Specify version. --version-file is recommended."
    echo "    --version-dir=<version-dir> : Specify version directory. --version-file is recommended."
    echo "    --version-file=<version-file> : Specify version info file path."
    echo "                                    Common script will parse version and version_dir from version_file."
    echo "                                    If you specified this option, --version and --version-dir options"
    echo "                                    no longer need to be specified."
    echo "    --custom-options=<args> : Specify custom options for package custom script."
    echo "    --setenv : Add \"source setenv.\${shell_type}\" to rcfile."
    echo "    --set-cann-uninstall : Add uninstall command into cann_uninstall.sh."
    echo "                           Always specified this in multi-version install."
    echo ""
    echo "  $0 --uninstall [options] <install_type> <install_path> <filelist_path>"
    echo ""
    echo "  Command options:"
    echo "    --package=<package> : Specify package name."
    echo "    --version=<version> : Specify version. --version-file is recommended."
    echo "    --version-dir=<version-dir> : Specify version directory. --version-file is recommended."
    echo "    --version-file=<version-file> : Specify version info file path."
    echo "                                    Common script will parse version and version_dir from version_file."
    echo "                                    If you specified this option, --version and --version-dir options"
    echo "                                    no longer need to be specified."
    echo "    --custom-options=<args> : Specify custom options for package custom script."
    echo ""
    echo "  ----------------------------------------------------------------------------------------------------"
    echo ""
    echo "  $0 {--add-cann-uninstall,--del-cann-uninstall}"
    echo ""
    echo "    --add-cann-uninstall : Add uninstall command in cann_uninstall.sh."
    echo "    --del-cann-uninstall : Del uninstall command in cann_uninstall.sh."
    echo ""
    echo "  Group options:"
    echo "    --install-path=<install-path> : Specify install path."
    echo ""
    echo "  $0 --add-cann-uninstall [options] <script_dir>"
    echo ""
    echo "  Command options:"
    echo "    --username=<username> : Specify install username."
    echo "    --usergroup=<usergroup> : Specify install usergroup."
    echo "    --install_for_all : Install for all users."
    echo "                        Usually specified in root user install scene."
    echo ""
    echo "  $0 --del-cann-uninstall [options] <script_dir>"
    echo ""
    echo "  ----------------------------------------------------------------------------------------------------"
    echo ""
    echo "  $0 {--add-env-rc,--del-env-rc}"
    echo ""
    echo "    --add-env-rc : Add \"source setenv.\${shell_type}\" to rcfile."
    echo "    --del-env-rc : Del \"source setenv.\${shell_type}\" in rcfile."
    echo ""
    echo "  Group options:"
    echo "    --username=<username> : Specify install username."
    echo "    --usergroup=<usergroup> : Specify install usergroup."
    echo "    --docker-root=<docker-root> : Specify docker root path."
    echo "                                  Install path is not contained docker root path."
    echo ""
    echo "  $0 --add-env-rc [options] <install_path> <setenv_filepath> <shell_type>"
    echo ""
    echo "  Command options:"
    echo "    --package=<package> : Specify package name."
    echo "    --setenv : Add \"source setenv.\${shell_type}\" to rcfile."
    echo ""
    echo "  $0 --del-env-rc [options] <install_path> <setenv_filepath> <shell_type>"
    echo ""
    echo "  ----------------------------------------------------------------------------------------------------"
    echo ""
    echo "  $0 --pkg-in-dbinfo <install_path>"
    echo ""
    echo "  Does package in ascend_package_db.info. Echo true or false."
    echo ""
    echo "  Command options:"
    echo "    --package=<package> : Specify package name."
    echo ""
    echo "  ----------------------------------------------------------------------------------------------------"
    echo ""
    echo "  $0 --help"
    echo ""
    echo "  Print help messages."
}

# env rc相关命令
env_rc_commands() {
    local install_path
    local setenv_filepath
    local shell_type

    case "${OPERATE_TYPE}" in
    "add-env-rc")
        install_path="$1"
        setenv_filepath="$2"
        shell_type="$3"
        add_env_rc "${install_path}" "${setenv_filepath}" "${PACKAGE}" "${shell_type}" "${SETENV}" "${USERNAME}" "${USERGROUP}" "false" "${DOCKER_ROOT}"
        exit 0
        ;;
    "del-env-rc")
        install_path="$1"
        setenv_filepath="$2"
        shell_type="$3"
        del_env_rc "${install_path}" "${setenv_filepath}" "${shell_type}" "${USERNAME}" "${DOCKER_ROOT}"
        exit 0
        ;;
    esac
}

# db.info相关命令
dbinfo_commands() {
    local install_path

    case "${OPERATE_TYPE}" in
    "pkg-in-dbinfo")
        install_path="$1"
        pkg_in_dbinfo "${install_path}" "${PACKAGE}"
        exit 0
    esac
}

# spc相关命令
spc_commands() {
    local install_path="$1"
    local filelist_path="$2"
    local filelist_spc_path="$3"
    local version_dir="$4"
    local ret

    case "${OPERATE_TYPE}" in
    "spc_install")
        install_patch "${install_path}" "${filelist_path}" "${filelist_spc_path}" "${version_dir}"
        exit $?
        ;;
    "spc_rollback")
        rollback_patch "${install_path}" "${filelist_path}" "${version_dir}"
        exit 0
        ;;
    "spc_uninstall")
        uninstall_patch "${install_path}" "${filelist_path}" "${version_dir}"
        exit 0
        ;;
    esac
}

# cann_uninstall总卸载脚本相关命令
cann_uninstall_commands() {
    local script_dir="$1"
    local ret

    case "${OPERATE_TYPE}" in
    "add-cann-uninstall")
        do_add_cann_uninstall "${INSTALL_PATH}" "${script_dir}" "${USERNAME}" "${USERGROUP}" "${INSTALL_FOR_ALL}"
        ret="$?" && [ ${ret} -ne 0 ] && exit 1
        exit 0
        ;;
    "del-cann-uninstall")
        do_del_cann_uninstall "${INSTALL_PATH}" "${script_dir}"
        ret="$?" && [ ${ret} -ne 0 ] && exit 1
        exit 0
        ;;
    esac
}

# 多版本创建相关命令
multi_version_commands() {
    case "${OPERATE_TYPE}" in
    "notify_create_softlink")
        notify_latest_manager_create_softlink "$INSTALL_PATH/$LATEST_DIR/var" "$PACKAGE" "$VERSION" "$VERSION_DIR" \
            "$INSTALL_FOR_ALL" "$DOCKER_ROOT"
        ret="$?" && [ $ret -ne 0 ] && exit 1
        exit 0
        ;;
    "notify_remove_softlink")
        notify_latest_manager_remove_softlink "$INSTALL_PATH/$LATEST_DIR/var" "$PACKAGE" "$VERSION" "$VERSION_DIR" \
            "$INSTALL_FOR_ALL" "$DOCKER_ROOT"
        ret="$?" && [ $ret -ne 0 ] && exit 1
        exit 0
        ;;
    esac
}

# 正式命令
formal_commands() {
    local install_type="$1"
    local install_path="$2"
    local filelist_path="$3"
    local input_feature input_chip
    local ret docker_root
    local feature_param thread_num

    if [ "$FEATURE" != "all" ]; then
        normalize_feature "input_feature" "$FEATURE" "${OPERATE_TYPE}"
    else
        if [ $# = 3 ]; then
            input_feature="all"
        else
            normalize_feature "input_feature" "$4" "${OPERATE_TYPE}"
        fi
    fi
    normalize_feature "input_chip" "$CHIP" "${OPERATE_TYPE}"

    pack_feature_param "feature_param" "${input_feature}" "${FEATURE_EXCLUDE_ALL}" "${input_chip}"

    # 移除docker_root右侧的/
    rstrip_path "docker_root" "${DOCKER_ROOT}"

    if [ "$PARALLEL_LIMIT" = "true" ]; then
        get_thread_num "thread_num"
        init_fifo "PARALLEL_FIFO" "$thread_num"
    fi

    case "${OPERATE_TYPE}" in
    "install")
        do_install "${install_type}" "${install_path}" "${filelist_path}" "${PACKAGE}" "${feature_param}" "${docker_root}" "n"
        ret="$?" && [ $ret -ne 0 ] && return $ret
        ;;
    "uninstall")
        # 统一使用full模式卸载文件。两个包共用部分block（如opp和opp_kernel），以不同的模式安装时（如：run/full）
        # full模式的包先卸载，run模式的包后卸载，确保能够完整卸载。
        do_uninstall "full" "${install_path}" "${filelist_path}" "${PACKAGE}" "${feature_param}" "${docker_root}"
        ret="$?" && [ $ret -ne 0 ] && return $ret
        ;;
    "simple_install")
        # 简化安装模式
        do_install "${install_type}" "${install_path}" "${filelist_path}" "${PACKAGE}" "${feature_param}" "" "y"
        ret="$?" && [ $ret -ne 0 ] && return $ret
        ;;
    "simple_uninstall")
        # 简化卸载模式
        do_uninstall "full" "${install_path}" "${filelist_path}" "${PACKAGE}" "${feature_param}" "" "y"
        ret="$?" && [ $ret -ne 0 ] && return $ret
        ;;
    "mkdir")
        do_create_dirs "all" "$1" "$2" "$3" "${PACKAGE}" "${feature_param}"
        ret="$?" && [ $ret -ne 0 ] && return $ret
        ;;
    "makedir")
        do_create_dirs "mkdir" "$1" "$2" "$3" "${PACKAGE}" "${feature_param}"
        ret="$?" && [ $ret -ne 0 ] && return $ret
        ;;
    "copy")
        do_copy_files "$1" "$2" "$3" "${PACKAGE}" "${feature_param}"
        ret="$?" && [ $ret -ne 0 ] && return $ret
        ;;
    "chmoddir")
        do_chmod_file_dir "$install_type" "$2" "$3" "${feature_param}" "${PACKAGE}"
        ret="$?" && [ $ret -ne 0 ] && return $ret
        ;;
    "restoremod")
        do_create_dirs "resetmod" "$1" "$2" "$3" "${PACKAGE}" "${feature_param}"
        ret="$?" && [ $ret -ne 0 ] && return $ret
        ;;
    "remove")
        do_remove "$install_type" "$2" "$filelist_path" "${PACKAGE}" "${feature_param}"
        ret="$?" && [ $ret -ne 0 ] && return $ret
        ;;
    "print")
        print_install_content "$1" "$2" "${PRINT_OPERATE_TYPE}" "$3" "${feature_param}"
        ret="$?" && [ $ret -ne 0 ] && return $ret
        ;;
    esac

    return 0
}

# 正式命令并加锁
formal_commands_with_lock() {
    local install_path="$2"
    local lockfile docker_root install_path_real ret

    if [ "${OPERATE_TYPE}" = "install" ] || [ "${OPERATE_TYPE}" = "uninstall" ]; then
        if [ "${WITH_DOCKER_ROOT_PREFIX}" = "y" ]; then
            install_path_real="${install_path}"
        else
            # 移除docker_root右侧的/
            rstrip_path "docker_root" "${DOCKER_ROOT}"
            install_path_real="${docker_root}${install_path}"
        fi
    else
        install_path_real="${install_path}"
    fi
    lockfile="${install_path_real}/ascend.lock"

    if [ ! -d "${install_path_real}" ]; then
        # 兼容覆盖安装场景，包安装目录被删除情况
        mkdir -p "${install_path_real}"
        # 安装目录设置合适的权限
        change_mod "${install_path_real}" "750" "${INSTALL_FOR_ALL}"
        check_ret_error "$?" "Change mod ${install_path_real} failed!"
        ret="$?" && [ $ret -ne 0 ] && return $ret
    fi

    # 使用文件描述符9作为并发锁
    (
        flock -n 9
        if [ $? -ne 0 ]; then
            log "ERROR" "Get ${install_path_real} lockfile failed! There may be another process also installing in the directory."
            exit 1
        fi
        formal_commands "$@"
    ) 9> "${lockfile}"
    ret="$?"
    rm -f "${lockfile}"

    if [ $ret -ne 0 ]; then
        exit 1
    fi
}

# 第一个参数为source，标记为source场景，不执行主流程
[ "$1" = "source" ] && return 0

# 全局变量
# change_mod_and_own_files与change_mod_and_own_dirs函数中会使用
INSTALL_FOR_ALL=""
SETENV=""
SET_CANN_UNINSTALL=""
IS_UPGRADE=""
IS_RECREATE_SOFTLINK=""
WITH_DOCKER_ROOT_PREFIX=""
FEATURE_EXCLUDE_ALL="n"
REMOVE_INSTALL_INFO="n"  # 卸载时移除ascend_install.info文件
USE_SHARE_INFO="n"
CHIP="all"
FEATURE="all"
INCREMENT="n"  # 增量安装

OPERATE_TYPE=""
PACKAGE=""
USERNAME=""
USERGROUP=""
VERSION=""
VERSION_DIR=""
VERSION_FILE=""
CUSTOM_OPTIONS=""
# 宿主机打docker包场景，配置docker文件系统根路径
DOCKER_ROOT=""
# 安装路径
INSTALL_PATH=""
# 输入的latest_dir
INPUT_LATEST_DIR=""

while true; do
    case "$1" in
    --spc-install | -i)
        OPERATE_TYPE="spc_install"
        shift
        ;;
    --spc-rollback | -l)
        OPERATE_TYPE="spc_rollback"
        shift
        ;;
    --spc-uninstall | -u)
        OPERATE_TYPE="spc_uninstall"
        shift
        ;;
    --install)
        OPERATE_TYPE="install"
        shift
        ;;
    --uninstall)
        OPERATE_TYPE="uninstall"
        shift
        ;;
    --simple-install)
        OPERATE_TYPE="simple_install"
        shift
        ;;
    --simple-uninstall)
        OPERATE_TYPE="simple_uninstall"
        shift
        ;;
    --copy | -c)
        OPERATE_TYPE="copy"
        shift
        ;;
    --mkdir | -m)
        OPERATE_TYPE="mkdir"
        shift
        ;;
    --makedir | -d)
        OPERATE_TYPE="makedir"
        shift
        ;;
    --chmoddir | -o)
        OPERATE_TYPE="chmoddir"
        shift
        ;;
    --restoremod | -e)
        OPERATE_TYPE="restoremod"
        shift
        ;;
    --remove | -r)
        OPERATE_TYPE="remove"
        shift
        ;;
    --add-cann-uninstall)
        OPERATE_TYPE="add-cann-uninstall"
        shift
        ;;
    --del-cann-uninstall)
        OPERATE_TYPE="del-cann-uninstall"
        shift
        ;;
    --add-env-rc)
        OPERATE_TYPE="add-env-rc"
        shift
        ;;
    --del-env-rc)
        OPERATE_TYPE="del-env-rc"
        shift
        ;;
    --pkg-in-dbinfo)
        OPERATE_TYPE="pkg-in-dbinfo"
        shift
        ;;
    --add-pkg-blocks)
        OPERATE_TYPE="add-pkg-blocks"
        shift
        ;;
    --del-pkg-blocks)
        OPERATE_TYPE="del-pkg-blocks"
        shift
        ;;
    --create-latest-softlink)
        OPERATE_TYPE="notify_create_softlink"
        shift
        ;;
    --create-package-latest-softlink)
        OPERATE_TYPE="notify_create_softlink"
        shift
        ;;
    --remove-latest-softlink)
        OPERATE_TYPE="notify_remove_softlink"
        shift
        ;;
    --remove-package-latest-softlink)
        OPERATE_TYPE="notify_remove_softlink"
        shift
        ;;
    --print-blocks)
        OPERATE_TYPE="print-blocks"
        shift
        ;;
    --print=*)
        OPERATE_TYPE="print"
        PRINT_OPERATE_TYPE="$(echo "$1" | cut -d"=" -f2)"
        shift
        ;;
    --print| -p)
        OPERATE_TYPE="print"
        PRINT_OPERATE_TYPE="copy mkdir move"
        shift
        ;;
    --install-path=*)
        INSTALL_PATH=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --version=*)
        VERSION=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --version-dir=*)
        VERSION_DIR=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --version-file=*)
        VERSION_FILE=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --latest-dir=*)
        INPUT_LATEST_DIR=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --username=*)
        USERNAME=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --usergroup=*)
        USERGROUP=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --custom-options=*)
        CUSTOM_OPTIONS=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
    --arch=*)
        PKG_ARCH=$(echo "$1" | cut -d"=" -f2-)
        shift
        if [ "$PKG_ARCH" = "" ]; then
            log "ERROR" "The --arch option should not be an empty string."
            exit 1
        fi
        ;;
    --upgrade)
        IS_UPGRADE="y"
        shift
        ;;
    --recreate-softlink)
        IS_RECREATE_SOFTLINK="y"
        shift
        ;;
    --install_for_all)
        INSTALL_FOR_ALL="y"
        shift
        ;;
    --setenv)
        SETENV="y"
        shift
        ;;
    --set-cann-uninstall)
        SET_CANN_UNINSTALL="y"
        shift
        ;;
    --package=*)
        PACKAGE=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --docker-root=*)
        DOCKER_ROOT=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --chip=*)
        CHIP="$(echo "$1" | cut -d"=" -f2)"
        shift
        ;;
    --feature=*)
        FEATURE="$(echo "$1" | cut -d"=" -f2)"
        shift
        ;;
    --with-docker-root-prefix)
        WITH_DOCKER_ROOT_PREFIX="y"
        shift
        ;;
    --feature-exclude-all)
        FEATURE_EXCLUDE_ALL="y"
        shift
        ;;
    --remove-install-info)
        REMOVE_INSTALL_INFO="y"
        shift
        ;;
    --use-share-info)
        USE_SHARE_INFO="y"
        shift
        ;;
    --increment)
        INCREMENT="y"
        shift
        ;;
    -h | --help)
        help_info
        exit 0
        ;;
    -*)
        echo Unrecognized input options : "$1"
        help_info
        exit 1
        ;;
    *)
        break
        ;;
    esac
done

set_global_vars
env_rc_commands "$@"
dbinfo_commands "$@"
spc_commands "$@"
cann_uninstall_commands "$@"
multi_version_commands "$@"

if [ $# -lt 3 ]; then
    log "ERROR" "It's too few input params: $*"
    exit 1
fi

formal_commands_with_lock "$@"
exit $?
