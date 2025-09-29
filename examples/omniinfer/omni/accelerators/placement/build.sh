#!/bin/bash

GOOGLE_TEST_VERSION="googletest-1.16.0"
# Get the directory where the script is located (absolute path)
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script is located in: $CURRENT_DIR"

# Function to check for required tools
check_dependencies() {
    for cmd in unzip cmake make python3 pytest; do
        if ! command -v "$cmd" &> /dev/null; then
            echo "Error: $cmd is not installed. Please install it."
            exit 1
        fi
    done
}

# Function to install Google Test
install_gtest() {
    echo "Installing Google Test ${GOOGLE_TEST_VERSION}"
    local DEST="${CURRENT_DIR}/omni_placement/cpp/test/${GOOGLE_TEST_VERSION}"

    if [ -d "$DEST" ]; then
        echo "Google Test is already installed at $DEST"
        return
    fi

    if [ ! -f "${GOOGLE_TEST_VERSION}.zip" ]; then
        echo "Error: ${GOOGLE_TEST_VERSION}.zip not found in $(pwd)"
        exit 1
    fi

    unzip -n "${GOOGLE_TEST_VERSION}.zip" || {
        echo "Error: Failed to unzip ${GOOGLE_TEST_VERSION}.zip"
        exit 1
    }

    cd "$DEST" || {
        echo "Error: Failed to cd into $DEST"
        exit 1
    }

    echo "Building Google Test in: $(pwd)"
    cmake . && make && sudo make install || {
        echo "Error: Google Test build/install failed"
        exit 1
    }
    echo "Google Test installed successfully"

    cd "$CURRENT_DIR/omni_placement/cpp/test" || {
        echo "Error: Failed to return to test directory"
        exit 1
    }
}

# Function to build the C++ library
setup_omni_placement_clib() {
    echo "Setting up omni_placement C++ library"
    cd "${CURRENT_DIR}/omni_placement/cpp" || {
        echo "Error: Failed to cd into ${CURRENT_DIR}/omni_placement/cpp"
        exit 1
    }

    python3 setup.py build_ext --inplace || {
        echo "Error: Failed to build extension"
        exit 1
    }

    # python3 setup.py install || {
    #     echo "Error: Failed to install library"
    #     exit 1
    # }

    echo "C++ library setup completed"
}

pip_install_omni_placement_whl(){
    cd "${CURRENT_DIR}" || {
        echo "Error: Failed to cd into ${CURRENT_DIR}"
        exit 1
    }
    pip install -e . || {
        echo "Error: Failed to pip install omni_placement"
        exit 1
    }
}

# Function to run C++ unit tests with optional clean parameter
run_cpp_unittest() {
    local clean_flag="$1"  # Capture the clean parameter

    echo "Running C++ tests"
    cd "${CURRENT_DIR}/omni_placement/cpp/test" || {
        echo "Error: Failed to cd into ${CURRENT_DIR}/omni_placement/cpp/test"
        exit 1
    }

    install_gtest

    cmake .

    # If clean parameter is provided, run make clean first
    if [ "$clean_flag" = "clean" ]; then
        echo "Cleaning previous build..."
        make clean || {
            echo "Error: 'make clean' failed"
            exit 1
        }
    fi

    make || {
        echo "Error: Make failed for tests"
        exit 1
    }

    ./test_placement || {
        echo "Error: test_placement execution failed"
        exit 1
    }

    echo "C++ tests completed successfully"
}

# Function to run Python tests
run_pytest() {
    echo "Running Python tests"
    cd "$CURRENT_DIR" || {
        echo "Error: Failed to cd into $CURRENT_DIR"
        exit 1
    }

    # apply patch to vllm_npu
    bash ./scripts/copy_dsv3_to_vllm_npu.sh

    pytest --ignore=omni_placement/cpp/test/googletest-1.16.0/ --ignore=examples/ || {
        echo "Error: pytest failed"
        exit 1
    }

    echo "Python unit tests finished"
}

# Main execution with clean parameter handling
check_dependencies

# source environment variable
source ~/.bashrc

# Check if 'clean' is passed as an argument
if [ "$1" = "clean" ]; then
    setup_omni_placement_clib
    pip_install_omni_placement_whl
    run_cpp_unittest "clean"  # Pass clean to run_cpp_unittest
    run_pytest
else
    setup_omni_placement_clib
    pip_install_omni_placement_whl
    run_cpp_unittest ""       # Run without clean
    run_pytest
fi