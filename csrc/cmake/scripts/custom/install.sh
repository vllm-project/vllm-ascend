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

vendor_name=customize
targetdir=/usr/local/Ascend/opp
target_custom=0

sourcedir=$PWD/packages
vendordir=vendors/$vendor_name

QUIET="y"
INSTALL_FOR_ALL="n"


while true
do
    case $1 in
    --quiet)
        QUIET="y"
        shift
    ;;
    --install-path=*)
        INSTALL_PATH=$(echo $1 | cut -d"=" -f2-)
        INSTALL_PATH=${INSTALL_PATH%*/}
        shift
    ;;
    --install-for-all)
        INSTALL_FOR_ALL="y"
        shift
    ;;
    --*)
        shift
    ;;
    *)
        break
    ;;
    esac
done

log() {
    cur_date=`date +"%Y-%m-%d %H:%M:%S"`
    echo "[ops_custom] [$cur_date] "$1
}

if [ -n "${INSTALL_PATH}" ]; then
    if [[ ! "${INSTALL_PATH}" = /* ]]; then
        log "[ERROR] use absolute path for --install-path argument"
        exit 1
    fi
    if [ ! -d ${INSTALL_PATH} ]; then
        mkdir ${INSTALL_PATH} >> /dev/null 2>&1
        if [ $? -ne 0 ]; then
            log "[ERROR] create ${INSTALL_PATH}  failed"
            exit 1
        fi
    fi
    targetdir=${INSTALL_PATH}
elif [ -n "${ASCEND_CUSTOM_OPP_PATH}" ]; then
    if [[ "${ASCEND_CUSTOM_OPP_PATH}" == *:* ]]; then
        log "[ERROR] environment variable ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH} is set and \
        has multiple path in it (colon inside), which will cause the custom op installed incorrectly. \
        Please use the --install-path option to specify an installation path instead."
        exit 1
    fi
    if [ ! -d ${ASCEND_CUSTOM_OPP_PATH} ]; then
        mkdir -p ${ASCEND_CUSTOM_OPP_PATH} >> /dev/null 2>&1
        if [ $? -ne 0 ]; then
            log "[ERROR] create ${ASCEND_CUSTOM_OPP_PATH}  failed"
        fi
    fi
    targetdir=${ASCEND_CUSTOM_OPP_PATH}
else
    if [ "x${ASCEND_OPP_PATH}" == "x" ]; then
        log "[ERROR] env ASCEND_OPP_PATH no exist"
        exit 1
    fi
    targetdir="${ASCEND_OPP_PATH}"
fi

if [ ! -d $targetdir ];then
    log "[ERROR] $targetdir no exist"
    exit 1
fi

if [ ! -x $targetdir ] || [ ! -w $targetdir ] || [ ! -r $targetdir ];then
    log "[WARNING] The directory $targetdir does not have sufficient permissions. \
    Please check and modify the folder permissions (e.g., using chmod), \
    or use the --install-path option to specify an installation path and \
    change the environment variable ASCEND_CUSTOM_OPP_PATH to the specified path."
fi

upgrade()
{
    if [ ! -d ${sourcedir}/$vendordir/$1 ]; then
        log "[INFO] no need to upgrade ops $1 files"
        return 0
    fi

    if [ ! -d ${targetdir}/$vendordir/$1 ];then
        log "[INFO] create ${targetdir}/$vendordir/$1."
        mkdir -p ${targetdir}/$vendordir/$1
        if [ $? -ne 0 ];then
            log "[ERROR] create ${targetdir}/$vendordir/$1 failed"
            return 1
        fi
    else
        has_same_file=-1
        for file_a in ${sourcedir}/$vendordir/$1/*; do
            file_b=${file_a##*/};
            if [ "ls ${targetdir}/$vendordir/$1" = "" ]; then
                log "[INFO] ${targetdir}/$vendordir/$1 is empty !!"
		        return 1
	          fi
            grep -q $file_b <<<`ls ${targetdir}/$vendordir/$1`;
            if [[ $? -eq 0 ]]; then
                echo -n "${file_b} "
                has_same_file=0
            fi
        done
        if [ 0 -eq $has_same_file ]; then
            echo
            if test $QUIET = "n"; then
                echo "[INFO]: has old version in ${targetdir}/$vendordir/$1, \
                you want to Overlay Installation , please enter:[o]; \
                or replace directory installation , please enter: [r]; \
                or not install , please enter:[n]."

                while true
                do
                    read orn
                    if [ "$orn" = n ]; then
                        return 0
                    elif [ "$orn" = o ]; then
                        break;
                    elif [ "$orn" = r ]; then
                        [ -n "${targetdir}/$vendordir/$1/" ] && rm -rf "${targetdir}/$vendordir/$1"/*
                        break;
                    else
                        log "[ERROR] input error, please input again!"
                    fi
                done
            else
                [ -n "${targetdir}/$vendordir/$1/" ] && rm -rf "${targetdir}/$vendordir/$1"/*
            fi
        fi
        log "[INFO] replace or merge old ops $1 files ......"
    fi

    log "[INFO] copy new ops $1 files ......"
    if [ -d ${targetdir}/$vendordir/$1/ ]; then
        chmod -R +w "$targetdir/$vendordir/$1/" >/dev/null 2>&1
    fi
    cp -rf ${sourcedir}/$vendordir/$1/* $targetdir/$vendordir/$1/
    if [ $? -ne 0 ];then
        log "[ERROR] copy new $1 files failed"
        return 1
    fi

    return 0
}
upgrade_proto()
{
    if [ ! -f ${sourcedir}/$vendordir/custom.proto ]; then
        log "[INFO] no need to upgrade custom.proto files"
        return 0
    fi
    if [ ! -d ${targetdir}/$vendordir/framework/caffe ];then
        log "[INFO] create ${targetdir}/$vendordir/framework/caffe."
        mkdir -p ${targetdir}/$vendordir/framework/caffe
        if [ $? -ne 0 ];then
            log "[ERROR] create ${targetdir}/$vendordir/framework/caffe failed"
            return 1
        fi
    else
        if [ -f ${targetdir}/$vendordir/framework/caffe/custom.proto ]; then
            # 有老版本,判断是否要覆盖式安装
            if test $QUIET = "n"; then
                  echo "[INFO] ${targetdir}/$vendordir/framework/caffe has old version"\
                "custom.proto file. Do you want to replace? [y/n] "

                while true
                do
                    read yn
                    if [ "$yn" = n ]; then
                        return 0
                    elif [ "$yn" = y ]; then
                        break;
                    else
                        log "[ERROR] input error, please input again!"
                    fi
                done
            fi
        fi
        log "[INFO] replace old caffe.proto files ......"
    fi
    chmod -R +w "$targetdir/$vendordir/framework/caffe/" >/dev/null 2>&1
    cp -rf ${sourcedir}/$vendordir/custom.proto ${targetdir}/$vendordir/framework/caffe/
    if [ $? -ne 0 ];then
        log "[ERROR] copy new custom.proto failed"
        return 1
    fi
	log "[INFO] copy custom.proto success"

    return 0
}

upgrade_file()
{
    if [ ! -e ${sourcedir}/$vendordir/$1 ]; then
        log "[INFO] no need to upgrade ops $1 file"
        return 0
    fi

    log "[INFO] copy new $1 files ......"
    cp -f ${sourcedir}/$vendordir/$1 $targetdir/$vendordir/$1
    if [ $? -ne 0 ];then
        log "[ERROR] copy new $1 file failed"
        return 1
    fi

    return 0
}

delete_optiling_file()
{
  if [ ! -d ${targetdir}/vendors ];then
    log "[INFO] $1 not exist, no need to uninstall"
    return 0
  fi
  sys_info=$(uname -m)
  if [ ! -d ${sourcedir}/$vendordir/$1/ai_core/tbe/op_tiling/lib/linux/${sys_info} ];then
    rm -rf ${sourcedir}/$vendordir/$1/ai_core/tbe/op_tiling/liboptiling.so
  fi
  return 0
}

log "[INFO] copy uninstall sh success"

if [ ! -d ${targetdir}/vendors ];then
        log "[INFO] create ${targetdir}/vendors."
        mkdir -p ${targetdir}/vendors
        if [ $? -ne 0 ];then
            log "[ERROR] create ${targetdir}/vendors failed"
            exit 1
        fi
fi
chmod u+w ${targetdir}/vendors

log "[INFO] upgrade framework"
upgrade framework
if [ $? -ne 0 ];then
    exit 1
fi

log "[INFO] upgrade op proto"
upgrade op_proto
if [ $? -ne 0 ];then
    exit 1
fi

log "[INFO] upgrade op impl"
delete_optiling_file op_impl
upgrade op_impl
if [ $? -ne 0 ];then
    exit 1
fi

log "[INFO] upgrade op api"
upgrade op_api
if [ $? -ne 0 ];then
    exit 1
fi

log "[INFO] upgrade version.info"
upgrade_file version.info
if [ $? -ne 0 ];then
    exit 1
fi

upgrade_proto
if [ $? -ne 0 ];then
    exit 1
fi

# set the set_env.bash
if [ -n "${INSTALL_PATH}" ] && [ -d ${INSTALL_PATH} ]; then
    _ASCEND_CUSTOM_OPP_PATH=${targetdir}/${vendordir}
    bin_path="${_ASCEND_CUSTOM_OPP_PATH}/bin"
    set_env_variable="#!/bin/bash\nexport ASCEND_CUSTOM_OPP_PATH=${_ASCEND_CUSTOM_OPP_PATH}:\${ASCEND_CUSTOM_OPP_PATH}\nexport LD_LIBRARY_PATH=${_ASCEND_CUSTOM_OPP_PATH}/op_api/lib/:\${LD_LIBRARY_PATH}"
    if [ ! -d ${bin_path} ]; then
        mkdir -p ${bin_path} >> /dev/null 2>&1
        if [ $? -ne 0 ]; then
            log "[ERROR] create ${bin_path} failed"
            exit 1
        fi
    fi
    echo -e ${set_env_variable} > ${bin_path}/set_env.bash
    if [ $? -ne 0 ]; then
        log "[ERROR] write ASCEND_CUSTOM_OPP_PATH to set_env.bash failed"
        exit 1
    else
        log "[INFO] using requirements: when custom module install finished or before you run the custom module, \
        execute the command [ source ${bin_path}/set_env.bash ] to set the environment path"
    fi
else
    _ASCEND_CUSTOM_OPP_PATH=${targetdir}/${vendordir}
    config_file=${targetdir}/vendors/config.ini
    if [ ! -f ${config_file} ]; then
        touch ${config_file}
        chmod 640 ${config_file}
        echo "load_priority=$vendor_name" > ${config_file}
        if [ $? -ne 0 ];then
            log "[ERROR] echo load_priority failed"
            exit 1
        fi
    else
        found_vendors="$(grep -w "load_priority" "$config_file" | cut --only-delimited -d"=" -f2-)"
        found_vendor=$(echo $found_vendors | sed "s/\<$vendor_name\>//g" | tr ',' ' ')
        vendor=$(echo $found_vendor | tr -s ' ' ',')
        if [ "$vendor" != "" ]; then
            sed -i "/load_priority=$found_vendors/s@load_priority=$found_vendors@load_priority=$vendor_name,$vendor@g" "$config_file"
        fi
    fi
    if test $INSTALL_FOR_ALL = "y"; then
        chmod 755 ${config_file}
    fi
    log "[INFO] using requirements: when custom module install finished or before you run the custom module, \
        execute the command [ export LD_LIBRARY_PATH=${_ASCEND_CUSTOM_OPP_PATH}/op_api/lib/:\${LD_LIBRARY_PATH} ] to set the environment path"
fi

if [ -d ${targetdir}/$vendordir/op_impl/cpu/aicpu_kernel/impl/ ]; then
    chmod -R 440 ${targetdir}/$vendordir/op_impl/cpu/aicpu_kernel/impl/* >/dev/null 2>&1
fi

echo "SUCCESS"
exit 0
