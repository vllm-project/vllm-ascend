#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

req_ver_path=$1
_CURR_PATH=$(dirname $(readlink -f $0))
_DEFAULT_INSTALL_PATH="/usr/local/Ascend"
FILE_NOT_EXIST="0x0080"

getdate() {
  _cur_date=$(date +"%Y-%m-%d %H:%M:%S")
  echo "${_cur_date}"
}

logandprint() {
  echo "[OpsTransformer] [$(getdate)] ""$1"
}

check_path_pre() {
  in_checkpath_0="$1"
  in_checkpath_1=$(echo ${in_checkpath_0} | cut -d"=" -f2)
  if [ "${in_checkpath_1}" = "" ]; then
    logandprint "[WARNING]: please input correct path"
    exit 1
  fi
  arr=$(echo ${in_checkpath_1} | awk '{split($0,arr," ");for(i in arr) print arr[i]}')
  index=0
  for i in $arr; do
    id=$((${id:=-1} + 1))
    eval arr_$id=$i
    index=$(expr $index + 1)
  done
  len="${index}"
  b=0
  for i in $(seq 0 ${len}); do
    select_last_dir_component "$(eval echo '$'arr_$i)"
    ret=$last_component
    if [ "${ret}" != "" ]; then
      eval checked_path_temp_$b=$(eval echo '$'arr_$i)
      b=$(expr $b + 1)
    fi
  done
  check_all_path=""
  for i in $(seq 0 ${len}); do
    check_all_path="$(eval echo '$'checked_path_temp_$i) $check_all_path"
  done
  checked_path="${check_all_path}"
  return
}

select_last_dir_component() {
  path="$1"
  last_component=$(basename "${path}")
  if [ "${last_component}" = "atc" ]; then
    last_component="atc"
    return
  elif [ "${last_component}" = "fwkacllib" ]; then
    last_component="fwkacllib"
    return
  elif [ "${last_component}" = "compiler" ]; then
    last_component="compiler"
    return
  elif [ "${last_component}" = "fwkplugin" ]; then
    last_component="fwkplugin"
    return
  else
    last_component="atc or fwkacllib or compiler"
    return
  fi
}

check_version_file() {
  pkg_path="$1"
  component_ret="$2"
  run_pkg_path_temp=$(dirname "${pkg_path}")
  run_pkg_path="${run_pkg_path_temp}""/${component_ret}"
  version_file="${run_pkg_path}""/version.info"
  if [ -f "${version_file}" ]; then
    echo "${version_file}" 2 >>/dev/null
  else
    logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST}; The [${component_ret}] version.info in path [${pkg_path}] not exists."
    exit 1
  fi
  return
}

check_opp_version_file() {
  if [ -f "${_CURR_PATH}/../../version.info" ]; then
    ver_info="${_CURR_PATH}/../../version.info"
  # ops_transformer/version.info -> ops_transformer
  elif [ -f "${_DEFAULT_INSTALL_PATH}/ops_transformer/version.info" ]; then
    ver_info="${_DEFAULT_INSTALL_PATH}/ops_transformer/version.info"
  else
    logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST}; The [ops_transformer] version.info not exists."
  fi
  return
}

check_relation() {
  opp_ver_info="$1"
  req_pkg_name="$2"
  req_pkg_version="$3"
  _COMMON_INC_FILE="${_CURR_PATH}/common_func.inc"
  if [ -f "${_COMMON_INC_FILE}" ]; then
    . "${_COMMON_INC_FILE}"
    check_pkg_ver_deps "${opp_ver_info}" "${req_pkg_name}" "${req_pkg_version}"
    ret_situation=$ver_check_status
  else
    logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST}; The ${_COMMON_INC_FILE} not exists."
  fi
  return
}

show_relation() {
  relation_situation="$1"
  req_pkg_name_val="$2"
  req_pkg_path="$3"
  if [ "$relation_situation" = "SUCC" ]; then
    logandprint "[INFO]: Relationship of ops_transformer with ${req_pkg_name_val} in path ${req_pkg_path} check successfully"
    return 0
  else
    logandprint "[WARNING]: Relationship of ops_transformer with ${req_pkg_name_val} in path ${req_pkg_path} check failed. \
do you want to continue.  [y/n] "
    while true; do
      read yn
      if [ "$yn" == "n" ]; then
        echo "stop check!"
        exit 1
      elif [ "$yn" = y ]; then
        break
      else
        echo "[WARNING]: Input error, please input y or n to choose!"
      fi
    done
  fi
}

version_check() {
  path_val="$1"
  #get ops_transformer version
  check_opp_version_file
  ret_check_opp_version_file=$ver_info
  #get checked path
  check_path_pre "${path_val}"
  ret_check_path_pre=$checked_path
  if [ "${ret_check_path_pre}" != "" ]; then
    for var in ${ret_check_path_pre}; do
      # select_last_dir_component "${var}"
      # component_ret=$last_component
      #get atc or fwkacllib name
      select_last_dir_component "${var}"
      ret_last_component=$last_component
      #get the version of atc/fwkacllib
      check_version_file "${var}" "${ret_last_component}"
      ret_check_version_file=$version_file
      #check relation
      check_relation "${ret_check_opp_version_file}" "${ret_last_component}" "${ret_check_version_file}"
      ret_check_relation=$ret_situation
      #show relation
      show_relation "${ret_check_relation}" "${ret_last_component}" "${var}"
    done
  fi
}

version_check "${req_ver_path}"
exit 0
