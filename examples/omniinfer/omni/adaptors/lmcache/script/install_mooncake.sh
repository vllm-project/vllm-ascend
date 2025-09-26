#!/bin/bash

BUILD_ROOT="$(dirname "$(realpath "$0")")"
cd $BUILD_ROOT

if [ $(id -u) -ne 0 ]; then
    echo "Require root permission, try sudo ./build.sh"
    exit 1
fi

# install mooncake
PATCH_ROOT=${1:-$BUILD_ROOT/../patch/}
MOONCAKE_PATH=${2:-$BUILD_ROOT/Mooncake}
git clone https://github.com/kvcache-ai/Mooncake.git

cd ${MOONCAKE_PATH}
git reset --hard
git clean -fd
git checkout v0.3.5
git apply --whitespace=nowarn $PATCH_ROOT/mooncake_install.patch
git apply --whitespace=nowarn $PATCH_ROOT/bug_segment_port.patch

SYSTEM_PACKAGES="rdma-core-devel \
    gflags-devel \
    yaml-cpp-devel \
    gtest-devel \
    jsoncpp-devel \
    libunwind-devel \
    numactl-devel \
    boost-devel \
    boost-system \
    boost-thread \
    openssl-devel \
    grpc-devel \
    protobuf-devel \
    protobuf-compiler \
    libcurl-devel \
    hiredis-devel \
    patchelf"

yum install -y $SYSTEM_PACKAGES


echo "Installing yalantinglibs"

if [ ! -d "${MOONCAKE_PATH}/thirdparties" ]; then
    mkdir -p "${MOONCAKE_PATH}/thirdparties"
fi

cd "${MOONCAKE_PATH}/thirdparties"
if [ -d "yalantinglibs" ]; then
    echo -e "yalantinglibs directory already exists. Removing for fresh install..."
    rm -rf yalantinglibs
fi

git config --global http.sslVerify false
git clone https://github.com/alibaba/yalantinglibs.git

cd yalantinglibs
git checkout 0.5.1
mkdir -p build
cd build

cmake .. -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF
cmake --build . -j$(nproc)
cmake --install .

echo "Installing glog"
cd "${MOONCAKE_PATH}/thirdparties"
git clone https://github.com/google/glog.git

cd glog
git checkout v0.7.1

cmake -DWITH_GTEST=OFF -S . -B build -G "Unix Makefiles"
cmake --build build --target install -j$(nproc)

# Check if .gitmodules exists
if [ -f "${MOONCAKE_PATH}/.gitmodules" ]; then
    FIRST_SUBMODULE=$(grep "path" ${MOONCAKE_PATH}/.gitmodules | head -1 | awk '{print $3}')
    cd "${MOONCAKE_PATH}"

    if [ -d "${MOONCAKE_PATH}/${FIRST_SUBMODULE}/.git" ] || [ -f "${MOONCAKE_PATH}/${FIRST_SUBMODULE}/.git" ]; then
        echo -e "Git submodules already initialized. Skipping..."
    else
        git submodule update --init
    fi
else
    echo -e "No .gitmodules file found. Skipping..."
    exit 1
fi

GOVER=1.23.8
echo "Installing Go $GOVER"
install_go() {
    ARCH=$(uname -m)
    if [ "$ARCH" = "aarch64" ]; then
        ARCH="arm64"
    elif [ "$ARCH" = "x86_64" ]; then
        ARCH="amd64"
    else
        echo "Unsupported architecture: $ARCH"
        exit 1
    fi
    # Download Go
    wget -q --show-progress http://mirrors.aliyun.com/golang/go$GOVER.linux-$ARCH.tar.gz

    # Install Go
    tar -C /usr/local -xzf go$GOVER.linux-$ARCH.tar.gz

    # Clean up downloaded file
    rm -f go$GOVER.linux-$ARCH.tar.gz
}

# Check if Go is already installed
if command -v go &> /dev/null; then
      GO_VERSION=$(go version | awk '{print $3}')
      if [[ "$GO_VERSION" == "go$GOVER" ]]; then
          echo -e "GO $GOVER is already installed. Skipping..."
      else
          echo -e "Found Go $GO_VERSION. Will install Go $GOVER..."
          install_go
      fi
else
    install_go
fi

# Add Go to PATH if not already there
if ! grep -q "export PATH=\$PATH:/usr/local/go/bin" ~/.bashrc; then
    echo -e "Adding Go to your PATH in ~/.bashrc"
    echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
    echo -e "Please run 'source ~/.bashrc' or start a new terminal to use Go"
fi

source ~/.bashrc

go env -w GO111MODULE=on
go env -w GOPROXY=https://mirrors.huaweicloud.com/repository/goproxy/
go env -w GONOSUMDB=*

if [ ! -d "${MOONCAKE_PATH}/build" ]; then
    mkdir -p "${MOONCAKE_PATH}/build"
fi
cd ${MOONCAKE_PATH}/build

cmake -DCMAKE_POLICY_VERSION_MINIMUM=4.0 -DUSE_ETCD=ON -DSTORE_USE_ETCD=ON -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF ..
make -j"$((($(nproc) - 2)))"

make install
ldconfig