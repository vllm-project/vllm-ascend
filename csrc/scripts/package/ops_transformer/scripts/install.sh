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
# error number and description
OPERATE_FAILED="0x0001"
PARAM_INVALID="0x0002"
FILE_NOT_EXIST="0x0080"
FILE_NOT_EXIST_DES="File not found."
OPP_COMPATIBILITY_CEHCK_ERR="0x0092"
OPP_COMPATIBILITY_CEHCK_ERR_DES="OppTransformer compatibility check error."
PERM_DENIED="0x0093"
PERM_DENIED_DES="Permission denied."

OPP_PLATFORM_DIR=ops_transformer
OPP_PLATFORM_UPPER=$(echo "${OPP_PLATFORM_DIR}" | tr '[:lower:]' '[:upper:]')
CURR_OPERATE_USER="$(id -nu 2>/dev/null)"
CURR_OPERATE_GROUP="$(id -ng 2>/dev/null)"
# defaults for general user
if [ "$(id -u)" != "0" ]; then
  DEFAULT_INSTALL_PATH="${HOME}/Ascend"
else
  IS_FOR_ALL="y"
  DEFAULT_INSTALL_PATH="/usr/local/Ascend"
fi

# run package's files info, CURR_PATH means current temp path
CURR_PATH=$(dirname $(readlink -f $0))
INSTALL_SHELL_FILE="${CURR_PATH}/opp_install.sh"
RUN_PKG_INFO_FILE="${CURR_PATH}/../scene.info"
VERSION_INFO_FILE="${CURR_PATH}/../version.info"
COMMON_INC_FILE="${CURR_PATH}/common_func.inc"
VERCHECK_FILE="${CURR_PATH}/ver_check.sh"
VERSION_COMPAT_FUNC_PATH="${CURR_PATH}/version_compatiable.inc"
COMMON_FUNC_V2_PATH="${CURR_PATH}/common_func_v2.inc"
VERSION_CFG_PATH="${CURR_PATH}/version_cfg.inc"
OPP_COMMON_FILE="${CURR_PATH}/opp_common.sh"

. "${VERSION_COMPAT_FUNC_PATH}"
. "${COMMON_INC_FILE}"
. "${COMMON_FUNC_V2_PATH}"
. "${VERSION_CFG_PATH}"
. "${OPP_COMMON_FILE}"

ARCH_INFO=$(grep -e "arch" "$RUN_PKG_INFO_FILE" | cut --only-delimited -d"=" -f2-)
# 包内路径
GRAPH_SO_PATH="${CURR_PATH}/../../../../${OPP_PLATFORM_DIR}/built-in/op_graph/lib/linux/${ARCH_INFO}/libopgraph_transformer.so"
HOST_SO_PATH="${CURR_PATH}/../../../../${OPP_PLATFORM_DIR}/built-in/op_impl/ai_core/tbe/op_host/lib/linux/${ARCH_INFO}/libophost_transformer.so"

# defaluts info determinated by user's inputs
ASCEND_INSTALL_INFO="ascend_install.info"
TARGET_INSTALL_PATH="${DEFAULT_INSTALL_PATH}" #--input-path
TARGET_USERNAME="${CURR_OPERATE_USER}"
TARGET_USERGROUP="${CURR_OPERATE_GROUP}"
TARGET_VERSION_DIR="" # TARGET_INSTALL_PATH + PKG_VERSION_DIR
TARGET_SHARED_INFO_DIR=""

# keys of infos in ascend_install.info
KEY_INSTALLED_UNAME="USERNAME"
KEY_INSTALLED_UGROUP="USERGROUP"
KEY_INSTALLED_TYPE="${OPP_PLATFORM_UPPER}_INSTALL_TYPE"
KEY_INSTALLED_PATH="${OPP_PLATFORM_UPPER}_INSTALL_PATH_VAL"
KEY_INSTALLED_VERSION="${OPP_PLATFORM_UPPER}_VERSION"
KEY_INSTALLED_FEATURE="${OPP_PLATFORM_UPPER}_INSTALL_FEATURE"
KEY_INSTALLED_CHIP="${OPP_PLATFORM_UPPER}_INSTALL_CHIP"

# keys of infos in run package
KEY_RUNPKG_VERSION="Version"

# init install cmd status, set default as n
CMD_LIST="$*"
IS_UNINSTALL=n
IS_INSTALL=n
IS_UPGRADE=n
IS_QUIET=n
IS_INPUT_PATH=n
IS_CHECK=n
IN_INSTALL_TYPE=""
IN_INSTALL_PATH=""
IS_DOCKER_INSTALL=n
IS_SETENV=n
DOCKER_ROOT=""
CONFLICT_CMD_NUMS=0
IN_FEATURE="All"

# log functions
# start info before shell executing
startlog() {
  echo "[OpsTransformer] [$(getdate)] [INFO]: Start Time: $(getdate)"
}

exitlog() {
  echo "[OpsTransformer] [$(getdate)] [INFO]: End Time: $(getdate)"
}

#check ascend_install.info for the change in code warning
get_installed_info() {
  local key="$1"
  local res=""
  if [ -f "${INSTALL_INFO_FILE}" ]; then
    chmod 644 "${INSTALL_INFO_FILE}" >/dev/null 2>&1
    res=$(cat ${INSTALL_INFO_FILE} | grep "${key}" | awk -F = '{print $2}')
  fi
  echo "${res}"
}

clean_before_reinstall() {
  local installed_path=$(get_installed_info "${KEY_INSTALLED_PATH}")
  local existed_files=$(find ${TARGET_SHARED_INFO_DIR}/${OPP_PLATFORM_DIR} -type f -print 2>/dev/null)
  if [ -z "${existed_files}" ]; then
    logandprint "[INFO]: Directory is empty, directly install opp module."
    return 0
  fi

  if [ "${IS_QUIET}" = "y" ]; then
    logandprint "[WARNING]: Directory has file existed or installed opp\
 module, are you sure to keep installing opp module in it? y"
  else
    if [ ! -f "${INSTALL_INFO_FILE}" ]; then
      logandprint "[INFO]: Directory has file existed, do you want to continue? [y/n]"
    else
      logandprint "[INFO]: Opp package has been installed on the path $(get_installed_info "${KEY_INSTALLED_PATH}"),\
 the version is $(get_installed_info "${KEY_INSTALLED_VERSION}"),\
 and the version of this package is ${RUN_PKG_VERSION}, do you want to continue? [y/n]"
    fi
    while true; do
      read yn
      if [ "$yn" = "n" ]; then
        logandprint "[INFO]: Exit to install opp module."
        exitlog
        exit 0
      elif [ "$yn" = "y" ]; then
        break
      else
        echo "[WARNING]: Input error, please input y or n to choose!"
      fi
    done
  fi

  if [ "${installed_path}" = "${TARGET_VERSION_DIR}" ]; then
    logandprint "[INFO]: Clean the installed opp module before install."
    if [ ! -f "${UNINSTALL_SHELL_FILE}" ]; then
      logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST};ERR_DES:${FILE_NOT_EXIST_DES}.The file\
 (${UNINSTALL_SHELL_FILE}) not exists. Please set the correct install \
 path or clean the previous version opp install info (${INSTALL_INFO_FILE}) and then reinstall it."
      return 1
    fi
    bash "${UNINSTALL_SHELL_FILE}" "${TARGET_VERSION_DIR}" "upgrade" "${IS_QUIET}" ${IN_FEATURE} "${IS_DOCKER_INSTALL}" "${DOCKER_ROOT}" "$pkg_version_dir"
    if [ "$?" != 0 ]; then
      logandprint "[ERROR]: ERR_NO:${INSTALL_FAILED};ERR_DES:Clean the installed directory failed."
      return 1
    fi
  fi
  return 0
}

select_last_dir_component() {
  path="$1"
  last_component=$(basename ${path})
  if [ "${last_component}" = "atc" ]; then
    last_component="atc"
    return
  elif [ "${last_component}" = "fwkacllib" ]; then
    last_component="fwkacllib"
    return
  elif [ "${last_component}" = "compiler" ]; then
    last_component="compiler"
    return
  fi
}

# check_version_file() {
#   pkg_path="$1"
#   component_ret="$2"
#   run_pkg_path_temp=$(dirname "${pkg_path}")
#   run_pkg_path_temp2=${run_pkg_path_temp%/*}
#   run_pkg_path="${run_pkg_path_temp}""/${component_ret}"
#   run_pkg_path_temp2=${run_pkg_path%/*}
#   version_file="${run_pkg_path}""/version.info"
#   version_file_tmp="${run_pkg_path_temp2}""/version.info"
#   if [ -f "${version_file_tmp}" ]; then
#     version_file=${version_file_tmp}
#   fi
#   if [ -f "${version_file}" ]; then
#     echo "${version_file}" 2 >>/dev/null
#   else
#     logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST}; The [${component_ret}] version.info in path [${pkg_path}] not exists."
#     exitlog
#     exit 1
#   fi
#   return
# }

check_opp_version_file() {
  if [ -f "${CURR_PATH}/../../version.info" ]; then
    opp_ver_info="${CURR_PATH}/../../version.info"
  elif [ -f "${DEFAULT_INSTALL_PATH}/${OPP_PLATFORM_DIR}/share/info/version.info" ]; then
    opp_ver_info="${DEFAULT_INSTALL_PATH}/${OPP_PLATFORM_DIR}/share/info/version.info"
  else
    logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST}; The [${OPP_PLATFORM_DIR}] version.info not exists."
    exitlog
    exit 1
  fi
  echo "find opp_ver_info: ${opp_ver_info}"
  return
}


check_docker_path() {
  docker_path="$1"
  if [ "${docker_path}" != "/"* ]; then
    echo "[OpsTransformer] [ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Parameter --docker-root\
 must with absolute path that which is start with root directory /. Such as --docker-root=/${docker_path}"
    exitlog
    exit 1
  fi
  if [ ! -d "${docker_path}" ]; then
    echo "[OpsTransformer] [ERROR]: ERR_NO:${FILE_NOT_EXIST}; The directory:${docker_path} not exist, please create this directory."
    exitlog
    exit 1
  fi
}

judgment_path() {
  . "${COMMON_INC_FILE}"
  check_install_path_valid "${1}"
  if [ $? -ne 0 ]; then
    echo "[OpsTransformer][ERROR]: The opp install path ${1} is invalid, only characters in [a-z,A-Z,0-9,-,_] are supported!"
    exitlog
    exit 1
  fi
}

check_install_path() {
  TARGET_INSTALL_PATH="$1"
  # empty patch check
  if [ "x${TARGET_INSTALL_PATH}" = "x" ]; then
    echo "[OpsTransformer] [ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Parameter --install-path\
 not support that the install path is empty."
    exitlog
    exit 1
  fi
  # space check
  if echo "x${TARGET_INSTALL_PATH}" | grep -q " "; then
    echo "[OpsTransformer] [ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Parameter --install-path\
 not support that the install path contains space character."
    exitlog
    exit 1
  fi
  # delete last "/"
  local temp_path="${TARGET_INSTALL_PATH}"
  temp_path=$(echo "${temp_path%/}")
  if [ x"${temp_path}" = "x" ]; then
    temp_path="/"
  fi
  # covert relative path to absolute path
  local prefix=$(echo "${temp_path}" | cut -d"/" -f1 | cut -d"~" -f1)
  if [ "x${prefix}" = "x" ]; then
    TARGET_INSTALL_PATH="${temp_path}"
  else
    prefix=$(echo "${RUN_PATH}" | cut -d"/" -f1 | cut -d"~" -f1)
    if [ x"${prefix}" = "x" ]; then
      TARGET_INSTALL_PATH="${RUN_PATH}/${temp_path}"
    else
      echo "[OpsTransformer] [ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES: Run package path is invalid: $RUN_PATH"
      exitlog
      exit 1
    fi
  fi
  # covert '~' to home path
  local home=$(echo "${TARGET_INSTALL_PATH}" | cut -d"~" -f1)
  if [ "x${home}" = "x" ]; then
    local temp_path_value=$(echo "${TARGET_INSTALL_PATH}" | cut -d"~" -f2)
    if [ "$(id -u)" -eq 0 ]; then
      TARGET_INSTALL_PATH="/root$temp_path_value"
    else
      local home_path=$(eval echo "${USER}")
      home_path=$(echo "${home_path}%/")
      TARGET_INSTALL_PATH="$home_path$temp_path_value"
    fi
  fi
}

#get the dir of xxx.run
#opp_install_path_curr=`echo "$2" | cut -d"/" -f2- `
# cut first two params from *.run
get_run_path() {
  RUN_PATH=$(echo "$2" | cut -d"-" -f3-)
  if [ x"${RUN_PATH}" = x"" ]; then
    RUN_PATH=$(pwd)
  else
    # delete last "/"
    RUN_PATH=$(echo "${RUN_PATH%/}")
    if [ "x${RUN_PATH}" = "x" ]; then
      # root path
      RUN_PATH=$(pwd)
    fi
  fi
}

get_opts() {
  i=0
  while true
  do
    if [ "x$1" = "x" ]; then
      break
    fi
    if [ "$(expr substr "$1" 1 2)" = "--" ]; then
      i=$(expr $i + 1)
    fi
    if [ $i -gt 2 ]; then
      break
    fi
    shift 1
  done

  if [ "$*" = "" ]; then
    echo "[ERROR]: ERR_NO:${PARAM_INVALID}; ERR_DES:Unrecognized parameters.Try './xxx.run --help for more information.'"
    exitlog
    exit 1
  fi

  while true; do
    # skip 2 parameters avoid run pkg and directory as input parameter
    case "$1" in
      --full)
        IN_INSTALL_TYPE=$(echo ${1} | awk -F"--" '{print $2}')
        IS_INSTALL="y"
        CONFLICT_CMD_NUMS=$(expr $CONFLICT_CMD_NUMS + 1)
        shift
        ;;
      --upgrade)
        IS_UPGRADE="y"
        CONFLICT_CMD_NUMS=$(expr $CONFLICT_CMD_NUMS + 1)
        shift
        ;;
      --uninstall)
        IS_UNINSTALL="y"
        CONFLICT_CMD_NUMS=$(expr $CONFLICT_CMD_NUMS + 1)
        shift
        ;;
      --install-path=*)
        IS_INPUT_PATH="y"
        IN_INSTALL_PATH=$(echo ${1} | cut -d"=" -f2-)
        # check path
        judgment_path "${IN_INSTALL_PATH}"
        check_install_path "${IN_INSTALL_PATH}"
        shift
        ;;
      --quiet)
        IS_QUIET="y"
        shift
        ;;
      --install-for-all)
        IS_FOR_ALL="y"
        shift
        ;;
      -*)
        echo "[OpsTransformer] [ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Unsupported parameters [$1],\
 operation execute failed. Please use [--help] to see the useage."
        exitlog
        exit 1
        ;;
      *)
        break
        ;;
    esac
  done
}

# pre-check
check_opts() {
  if [ "${CONFLICT_CMD_NUMS}" != 1 ]; then
    echo "[OpsTransformer] [ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:\
 only support one type: full/run/devel/upgrade/uninstall/check, operation execute failed!\
 Please use [--help] to see the usage."
    exitlog
    exit 1
  fi
}

# init target_dir and log for install
init_env() {
  # create log folder and log file
  comm_init_log

  if is_version_dirpath "$TARGET_INSTALL_PATH"; then
    pkg_version_dir="$(basename "$TARGET_INSTALL_PATH")"
    TARGET_INSTALL_PATH="$(dirname "$TARGET_INSTALL_PATH")"
  else
    pkg_version_dir="cann"
  fi
  TARGET_VERSION_DIR="$TARGET_INSTALL_PATH/$pkg_version_dir"  # Splicing docker-root and install-path
  if [ "${IS_DOCKER_INSTALL}" = "y" ]; then
    # delete last "/"
    local temp_path_param="${DOCKER_ROOT}"
    local temp_path_val=$(echo "${temp_path_param%/}")
    if [ "x${temp_path_val}" = "x" ]; then
      temp_path_val="/"
    fi
    TARGET_VERSION_DIR=${temp_path_val}${TARGET_VERSION_DIR}
  fi

  TARGET_SHARED_INFO_DIR=${TARGET_VERSION_DIR}/share/info
  UNINSTALL_SHELL_FILE="${TARGET_SHARED_INFO_DIR}/${OPP_PLATFORM_DIR}/script/opp_uninstall.sh"
  INSTALL_INFO_FILE="${TARGET_SHARED_INFO_DIR}/${OPP_PLATFORM_DIR}/${ASCEND_INSTALL_INFO}"

  logandprint "[INFO]: Execute the opp run package."
  logandprint "[INFO]: OperationLogFile path: ${COMM_LOGFILE}."
  logandprint "[INFO]: Input params: $CMD_LIST"

  get_package_version "RUN_PKG_VERSION" "$VERSION_INFO_FILE"
  local installed_version=$(get_installed_info "${KEY_INSTALLED_VERSION}")
  if [ "${installed_version}" = "" ]; then
    logandprint "[INFO]: Version of installing opp module is ${RUN_PKG_VERSION}."
  else
    if [ "${RUN_PKG_VERSION}" != "" ]; then
      logandprint "[INFO]: Existed opp module version is ${installed_version},\
 the new opp module version is ${RUN_PKG_VERSION}."
    fi
  fi
}

check_pre_install() {
  local installed_user=$(get_installed_info "${KEY_INSTALLED_UNAME}")
  local installed_group=$(get_installed_info "${KEY_INSTALLED_UGROUP}")
  if [ "${installed_user}" != "" ] || [ "${installed_group}" != "" ]; then
    if [ "${installed_user}" != "${TARGET_USERNAME}" ] || [ "${installed_group}" != "${TARGET_USERGROUP}" ]; then
      logandprint "[ERROR]: The user and group are not same with last installation,\
  do not support overwriting installation!"
      exitlog
      exit 1
    fi
  fi
  
  if [ "${IS_UPGRADE}" = "y" ]; then
    if [ ! -e "${INSTALL_INFO_FILE}" ]; then
      logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST}; The directory:${TARGET_INSTALL_PATH} not install OpsTransformer, upgrade failed."
      exitlog
      exit 1
    fi
    IN_INSTALL_TYPE=$(get_installed_info "${KEY_INSTALLED_TYPE}")
  fi
}

#Support the installation script when the specified path (relative path and absolute path) does not exist
mkdir_install_path() {
  local base_dir=$(dirname ${TARGET_INSTALL_PATH})
  if [ ! -d ${base_dir} ]; then
    logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST}; The directory:${base_dir} not exist, please create this directory."
    exitlog
    exit 1
  fi

  if [ -d "${TARGET_INSTALL_PATH}" ]; then
    test -w ${TARGET_INSTALL_PATH} >>/dev/null 2>&1
    if [ "$?" -ne 0 ]; then
      #All paths exist with write permission
      logandprint "[ERROR]: ERR_NO:${PERM_DENIED};ERR_DES:${PERM_DENIED_DES}. The ${TARGET_USERNAME} do\
 access ${TARGET_INSTALL_PATH} failed, please reset the directory to a right permission."
      exit 1
    fi
  else
    test -w ${base_dir} >>/dev/null 2>&1
    if [ "$?" -ne 0 ]; then
      #All paths exist with write permission
      logandprint "[ERROR]: ERR_NO:${PERM_DENIED};ERR_DES:${PERM_DENIED_DES}. The ${TARGET_USERNAME} do\
 access ${base_dir} failed, please reset the directory to a right permission."
      exit 1
    else
      comm_create_dir "${TARGET_INSTALL_PATH}" "750" "${TARGET_USERNAME}:${TARGET_USERGROUP}" "${IS_FOR_ALL}"
    fi
  fi
}

install_package() {
  if [ "${IS_INSTALL}" = "n" ] && [ "${IS_UPGRADE}" = "n" ]; then
    return
  fi

  local architecture=$(uname -m)
  local graph_so_dir_path="${TARGET_VERSION_DIR}/opp/built-in/op_graph/lib/linux/${ARCH_INFO}"
  local host_so_dir_path="${TARGET_VERSION_DIR}/opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/${ARCH_INFO}"
  # check platform
  if [ "${architecture}" != "${ARCH_INFO}" ] ; then
    logandprint "[INFO]: the architecture of the run package is inconsistent with that of the current environment. "
    # 异构安装场景，拷贝so到指定目录
    if [ -d "${TARGET_VERSION_DIR}/opp/built-in/op_graph/lib/linux" ] ; then
      chmod u+w ${TARGET_VERSION_DIR}/opp/built-in/op_graph/lib/linux
    fi
    if [ -d "${TARGET_VERSION_DIR}/opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux" ] ; then
      chmod u+w ${TARGET_VERSION_DIR}/opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux
    fi
    mkdir -p ${graph_so_dir_path}
    mkdir -p ${host_so_dir_path}
    cp ${GRAPH_SO_PATH} ${graph_so_dir_path}
    cp ${HOST_SO_PATH} ${host_so_dir_path}
    chmod 755 ${graph_so_dir_path}/*
    chmod 755 ${host_so_dir_path}/*

    chmod u-w ${TARGET_VERSION_DIR}/opp/built-in/op_graph/lib/linux
    chmod u-w ${TARGET_VERSION_DIR}/opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux
    exit 0
  fi

  # use uninstall to clean the install folder
  clean_before_reinstall
  if [ "$?" != 0 ]; then
    comm_log_operation "Install" "${IN_INSTALL_TYPE}" "OpsTransformer" "$?" "${CMD_LIST}"
  fi

  bash "${INSTALL_SHELL_FILE}" "${TARGET_INSTALL_PATH}" "${TARGET_USERNAME}" "${TARGET_USERGROUP}" "${IN_FEATURE}" \
    "${IN_INSTALL_TYPE}" "${IS_FOR_ALL}" "${IS_SETENV}" "${IS_DOCKER_INSTALL}" "${DOCKER_ROOT}" "$pkg_version_dir"
  if [ "$?" != 0 ]; then
    comm_log_operation "Install" "${IN_INSTALL_TYPE}" "OpsTransformer" "$?" "${CMD_LIST}"
  fi
  if [ $(id -u) -eq 0 ]; then
    chown -R "root":"root" "${TARGET_SHARED_INFO_DIR}/${OPP_PLATFORM_DIR}/script" 2>/dev/null
    chown "root":"root" "${TARGET_SHARED_INFO_DIR}/${OPP_PLATFORM_DIR}" 2>/dev/null
    chmod -R 555 "${TARGET_SHARED_INFO_DIR}/${OPP_PLATFORM_DIR}/script" 2>/dev/null
    chmod 444 "${TARGET_SHARED_INFO_DIR}/${OPP_PLATFORM_DIR}/script/filelist.csv" 2>/dev/null
  else
    chmod -R 550 "${TARGET_SHARED_INFO_DIR}/${OPP_PLATFORM_DIR}/script" 2>/dev/null
    chmod 440 "${TARGET_SHARED_INFO_DIR}/${OPP_PLATFORM_DIR}/script/filelist.csv" 2>/dev/null
  fi
  comm_log_operation "Install" "${IN_INSTALL_TYPE}" "OpsTransformer" "$?" "${CMD_LIST}"
}

uninstall_package() {
  if [ "${IS_UNINSTALL}" = "n" ]; then
    return
  fi

  if [ ! -f "${UNINSTALL_SHELL_FILE}" ]; then
    logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST};ERR_DES:The file\
 (${UNINSTALL_SHELL_FILE}) not exists. Please make sure that the opp module\
 installed in (${TARGET_VERSION_DIR}) and then set the correct install path."
    uninstall_path=$(ls "${TARGET_INSTALL_PATH}" 2>/dev/null)
    if [ "${uninstall_path}" = "" ]; then
      rm -rf "${TARGET_INSTALL_PATH}"
    fi
    comm_log_operation "Uninstall" "${IN_INSTALL_TYPE}" "OpsTransformer" "$?" "${CMD_LIST}"
    exit 0
  fi

  # 如果是异构卸载
  local architecture=$(uname -m)
  if [ "${architecture}" != ${ARCH_INFO} ]; then
    target_arch=${ARCH_INFO}
  else
     # 判断异构so是否存在，存在则删除
    if [ "${architecture}" = "x86_64" ]; then
        target_arch="aarch64"
    else
        target_arch="x86_64"
    fi
  fi
  local graph_so_path="${TARGET_VERSION_DIR}/opp/built-in/op_graph/lib/linux/${target_arch}/libopgraph_transformer.so"
  local graph_so_dir_path="${TARGET_VERSION_DIR}/opp/built-in/op_graph/lib/linux/${target_arch}"
  local host_so_path="${TARGET_VERSION_DIR}/opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/${target_arch}/libophost_transformer.so"
  local host_so_dir_path="${TARGET_VERSION_DIR}/opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/${target_arch}"
  if [ -f "${graph_so_path}" ]; then
      rm -f "${graph_so_path}"
  fi
  if [ -f "${host_so_path}" ]; then
      rm -f "${host_so_path}"
  fi
  # 判断目录是否存在且是否为空
  if [ -d "${graph_so_dir_path}" ]; then
      if [ -z "$(ls -A "${graph_so_dir_path}")" ]; then
          rm -rf "${graph_so_dir_path}"
      fi
  fi
  if [ -d "${host_so_dir_path}" ]; then
      if [ -z "$(ls -A "${host_so_dir_path}")" ]; then
          rm -rf "${host_so_dir_path}"
      fi
  fi
  if [ "${architecture}" != ${ARCH_INFO} ]; then
      return
  fi

  bash "${UNINSTALL_SHELL_FILE}" "${TARGET_INSTALL_PATH}" "uninstall" "${IS_QUIET}" ${IN_FEATURE} "${IS_DOCKER_INSTALL}" "${DOCKER_ROOT}" "$pkg_version_dir"
  logandprint "[INFO]: Remove precheck info."

  comm_log_operation "Uninstall" "${IN_INSTALL_TYPE}" "OpsTransformer" "$?" "${CMD_LIST}"
}

main() {
  get_run_path "$@"

  startlog

  get_opts "$@"

  check_opts

  init_env

  check_pre_install

  mkdir_install_path

  install_package

  uninstall_package

}

main "$@"
exit 0
