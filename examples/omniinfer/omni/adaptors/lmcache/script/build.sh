 #!/bin/bash

set -e

BUILD_ROOT="$(dirname "$(realpath "$0")")"
cd $BUILD_ROOT
if [ ! -d "dist" ]; then
    mkdir dist
fi

if [ ! -d "lib" ]; then
    mkdir lib
fi

if [ $(id -u) -ne 0 ]; then
    echo "Require root permission, try sudo ./build.sh"
    exit 1
fi

git config --global http.sslVerify false
git clone https://github.com/LMCache/LMCache.git

PATCH_ROOT=${1:-$BUILD_ROOT/../patch/}
LMCACHE_PATH=${2:-$BUILD_ROOT/LMCache}
MOONCAKE_PATH=${3:-$BUILD_ROOT/Mooncake}

# install lmcache and ascend_lmcache
cd ${LMCACHE_PATH}
git reset --hard
git clean -fd
git checkout v0.3.2
git apply --whitespace=nowarn $PATCH_ROOT/npu_adaptor.patch

export NO_CUDA_EXT=1
python setup.py bdist_wheel
mv dist/lmcache* $BUILD_ROOT/dist

cd $BUILD_ROOT/..
python setup.py bdist_wheel
mv dist/ascend* $BUILD_ROOT/dist

# install mooncake
bash $BUILD_ROOT/install_mooncake.sh
bash $BUILD_ROOT/copy_library.sh $BUILD_ROOT/lib

cd $MOONCAKE_PATH
sed -i 's/^cp build\/mooncake-transfer-engine\/example\/transfer_engine_bench mooncake-wheel\/mooncake\//#&/' ./scripts/build_wheel.sh
sed -i 's/^pip install --upgrade pip/#&/' ./scripts/build_wheel.sh
sed -i 's/^pip install build setuptools wheel auditwheel/#&/' ./scripts/build_wheel.sh
sed -i '/auditwheel repair ${OUTPUT_DIR}\/\*.whl/,/-w ${REPAIRED_DIR}\/ --plat ${PLATFORM_TAG}/s/^/#/' ./scripts/build_wheel.sh
sed -i '/rm -f ${OUTPUT_DIR}\/\*.whl/,/mv ${REPAIRED_DIR}\/\*.whl ${OUTPUT_DIR}\//s/^/#/' ./scripts/build_wheel.sh
bash ./scripts/build_wheel.sh
cp ${MOONCAKE_PATH}/mooncake-wheel/dist/mooncake_transfer_engine-*.whl $BUILD_ROOT/dist