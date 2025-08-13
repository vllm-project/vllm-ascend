#!/bin/bash
# Usage: ./install_nixl.sh [--force]

FORCE=false
if [ "$1" == "--force" ]; then
    FORCE=true
fi

SUDO=false
if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    SUDO=true
fi

# A function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}
if command_exists apt; then
    if $SUDO; then
        sudo apt install -y build-essential cmake pkg-config
    else
        apt install -y build-essential cmake pkg-config
    fi
elif command_exists dnf; then
    if $SUDO; then
        sudo dnf install -y gcc-c++ cmake pkg-config
    else
        dnf install -y gcc-c++ cmake pkg-config
    fi
else
    echo "No apt or dnf package manager detected"
    exit 1
fi

ARCH=$(uname -m)

ROOT_DIR="/usr/local"
mkdir -p "$ROOT_DIR"
UCX_HOME="$ROOT_DIR/ucx"
NIXL_HOME="$ROOT_DIR/nixl"

export PATH="$UCX_HOME/bin:$NIXL_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$UCX_HOME/lib:$NIXL_HOME/lib/$ARCH-linux-gnu:$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$PKG_CONFIG_PATH:$UCX_HOME/lib/pkgconfig"

TEMP_DIR="nixl_installer"
mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

pip install meson ninja pybind11

if ! command -v ucx_info &> /dev/null || [ "$FORCE" = true ]; then
    echo "Installing UCX"
    wget https://github.com/openucx/ucx/releases/download/v1.18.0/ucx-1.18.0.tar.gz
    tar xzf ucx-1.18.0.tar.gz; rm ucx-1.18.0.tar.gz
    cd ucx-1.18.0
    
    ./configure  --prefix=$UCX_HOME                \
                --enable-shared                    \
                --disable-static                   \
                --disable-doxygen-doc              \
                --enable-optimizations             \
                --enable-cma                       \
                --enable-devel-headers             \
                --with-dm                          \
                --with-verbs                       \
                --enable-mt
    make -j
    make -j install-strip
    
    if $SUDO; then
        echo "Running ldconfig with sudo"
        sudo ldconfig
    else
        echo "Skipping ldconfig - sudo not available"
        echo "Please run 'sudo ldconfig' manually if needed"
    fi

    cd ..
else
    echo "Found existing UCX. Skipping UCX installation"  
fi

if ! command -v nixl_test &> /dev/null || [ "$FORCE" = true ]; then
    echo "Installing NIXL"
    wget https://github.com/ai-dynamo/nixl/archive/refs/tags/0.4.1.tar.gz
    tar xzf 0.4.1.tar.gz; rm 0.4.1.tar.gz
    cd nixl-0.4.1
    meson setup build --prefix=$NIXL_HOME -Ducx_path=$UCX_HOME -Ddisable_gds_backend=true
    cd build
    ninja
    ninja install
    cd ..
    pip install .
    cd ..
else
    echo "Found existing NIXL. Skipping NIXL installation"  
fi
