#!/bin/bash

START_DIR=$PWD
# Get the absolute path of the directory where the script is located
SOURCE_ROOT="$(dirname "$(realpath "$0")")"

# Function to safely change directory
safe_cd() {
    cd "$1" || { echo -e "\033[31mError: Failed to cd into $1\033[0m"; exit 1; }
}

# Function to check for required tools and install cmake if missing
check_dependencies() {
    echo "Checking for required dependencies..."
    for cmd in unzip make python3 pytest; do
        if ! command -v "$cmd" &> /dev/null; then
            echo -e "\033[31mError: $cmd is not installed. Please install it.\033[0m"
            exit 1
        fi
    done

    if command -v cmake &> /dev/null; then
        echo -e "\033[32mFound existing cmake installation.\033[0m"
    else
        echo -e "\033[33mWarning: cmake is not installed. Attempting to install with yum...\033[0m"
        yum install -y cmake || {
            echo -e "\033[31mError: Failed to install cmake using yum. Please install it manually.\033[0m"
            exit 1
        }
        echo -e "\033[32mcmake installed successfully.\033[0m"
    fi
}

# Function to source CANN environment and set environment variables
source_cann_env() {
    echo "Sourcing ~/.bashrc for CANN and torch-npu environment..."
    if [ -f ~/.bashrc ]; then
        source ~/.bashrc
        echo -e "\033[32m~/.bashrc sourced successfully.\033[0m"
        
        # Ensure ASCEND_TOOLKIT_HOME environment variable is set for CMakeLists.txt usage
        if [ -z "$ASCEND_TOOLKIT_HOME" ]; then
            if [ -d "/usr/local/Ascend/ascend-toolkit/latest" ]; then
                export ASCEND_TOOLKIT_HOME="/usr/local/Ascend/ascend-toolkit/latest"
                echo -e "\033[32mSet ASCEND_TOOLKIT_HOME=$ASCEND_TOOLKIT_HOME\033[0m"
            elif [ -d "/usr/local/Ascend/ascend-toolkit" ]; then
                # Find the latest version directory
                LATEST_VERSION=$(find /usr/local/Ascend/ascend-toolkit -maxdepth 1 -type d -name "*.*" | sort -V | tail -1)
                if [ -n "$LATEST_VERSION" ]; then
                    export ASCEND_TOOLKIT_HOME="$LATEST_VERSION"
                    echo -e "\033[32mSet ASCEND_TOOLKIT_HOME=$ASCEND_TOOLKIT_HOME\033[0m"
                fi
            fi
        else
            echo -e "\033[32mASCEND_TOOLKIT_HOME already configured: $ASCEND_TOOLKIT_HOME\033[0m"
        fi
        
        # Output important environment variables for debugging
        echo "Environment variables for debugging:"
        echo "ASCEND_TOOLKIT_HOME: $ASCEND_TOOLKIT_HOME"
        echo "LD_LIBRARY_PATH contains Ascend: $(echo $LD_LIBRARY_PATH | grep -o ascend | head -1 || echo 'No')"
        
    else
        echo -e "\033[33mWarning: ~/.bashrc not found. CANN environment may not be properly configured.\033[0m"
    fi
}

# Function to check NPU availability from Python
check_python_npu() {
    echo "Checking NPU availability from Python..."
    python3 -c "
import sys
try:
    import torch_npu
    print('[OK] torch_npu available')
    sys.exit(0)
except ImportError:
    try:
        import acl
        print('[OK] acl (Ascend Computing Language) available')
        sys.exit(0)
    except ImportError:
        print('[FAIL] Neither torch_npu nor acl available')
        sys.exit(1)
" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo -e "\033[32m[OK] Python NPU libraries available.\033[0m"
        return 0
    else
        echo -e "\033[33m[FAIL] Python NPU libraries not available.\033[0m"
        return 1
    fi
}

# Enhanced function to check for NPU availability using multiple methods
check_npu_presence() {
    local npu_detected=false
    local detection_methods=0
    
    echo "Checking NPU availability using multiple detection methods..."
    
    # Method 1: npu-smi command
    if command -v npu-smi &> /dev/null && npu-smi info &> /dev/null; then
        echo -e "\033[32m[OK] NPU detected via npu-smi command.\033[0m"
        npu_detected=true
        ((detection_methods++))
    else
        echo -e "\033[33m[FAIL] npu-smi command not available or failed.\033[0m"
    fi
    
    # Method 2: Check device files
    if [ -e /dev/davinci0 ] || [ -e /dev/davinci_manager ] || [ -e /dev/devmm_svm ] || [ -e /dev/hisi_hdc ]; then
        echo -e "\033[32m[OK] NPU device files detected.\033[0m"
        npu_detected=true
        ((detection_methods++))
    else
        echo -e "\033[33m[FAIL] No NPU device files found.\033[0m"
    fi
    
    # Method 3: Check driver modules
    if lsmod | grep -E "(davinci|ascend|hisi)" &> /dev/null; then
        echo -e "\033[32m[OK] NPU driver modules loaded.\033[0m"
        npu_detected=true
        ((detection_methods++))
    else
        echo -e "\033[33m[FAIL] No NPU driver modules found.\033[0m"
    fi
    
    # Method 4: Check PCI devices
    if lspci | grep -i "19e5:" &> /dev/null; then
        echo -e "\033[32m[OK] Huawei Ascend NPU PCI device detected.\033[0m"
        npu_detected=true
        ((detection_methods++))
    else
        echo -e "\033[33m[FAIL] No Huawei NPU PCI device found.\033[0m"
    fi
    
    # Method 5: Check CANN environment
    if [ -n "$ASCEND_OPP_PATH" ] || ([ -n "$LD_LIBRARY_PATH" ] && echo "$LD_LIBRARY_PATH" | grep -i ascend &> /dev/null); then
        echo -e "\033[32m[OK] CANN environment variables detected.\033[0m"
        npu_detected=true
        ((detection_methods++))
    else
        echo -e "\033[33m[FAIL] CANN environment not properly configured.\033[0m"
    fi
    
    # Method 6: Check Python NPU libraries
    if check_python_npu; then
        npu_detected=true
        ((detection_methods++))
    fi
    
    # Summary
    if [ "$npu_detected" = true ]; then
        echo -e "\033[32mNPU detected using $detection_methods method(s). Unit tests with NPU will be enabled.\033[0m"
        return 0
    else
        echo -e "\033[33mWarning: NPU not detected by any method. Skipping unit tests with NPU.\033[0m"
        return 1
    fi
}

# Function to install Google Test and Google Mock if not found
install_gtest() {
    # 1. Check for system packages (gtest-devel and gmock-devel)
    if rpm -q gtest-devel &> /dev/null && rpm -q gmock-devel &> /dev/null; then
        echo -e "\033[32mFound system-installed 'gtest-devel' and 'gmock-devel'. Skipping local installation.\033[0m"
        return 0
    fi

    # 2. Check for manual installation (libgtest.a and libgmock.a)
    if [ -f /usr/local/lib/libgtest.a ] && [ -f /usr/local/lib/libgmock.a ]; then
        echo -e "\033[32mFound manually installed Google Test and Google Mock in /usr/local. Skipping local installation.\033[0m"
        return 0
    fi

    # 3. Last Resort: Install from source
    local GOOGLE_TEST_VERSION="googletest-1.16.0"
    echo "No system-wide Google Test/Mock found. Installing ${GOOGLE_TEST_VERSION} locally."
    local DEST_DIR="${SOURCE_ROOT}/omni_placement/cpp/test"
    local GTEST_SRC_DIR="${DEST_DIR}/${GOOGLE_TEST_VERSION}"

    safe_cd "${DEST_DIR}"

    if [ ! -f "${GOOGLE_TEST_VERSION}.zip" ]; then
        echo -e "\033[31mError: ${GOOGLE_TEST_VERSION}.zip not found in $(pwd)\033[0m"
        echo -e "\033[33mPlease ensure the gtest source zip is present before running.\033[0m"
        exit 1
    fi

    if [ ! -d "$GTEST_SRC_DIR" ]; then
        unzip -q -n "${GOOGLE_TEST_VERSION}.zip" -d "${DEST_DIR}" || {
            echo -e "\033[31mError: Failed to unzip ${GOOGLE_TEST_VERSION}.zip\033[0m"
            exit 1
        }
    fi

    safe_cd "$GTEST_SRC_DIR"
    echo "Building Google Test in: $(pwd)"
    
    mkdir -p build && safe_cd build

    cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. && make && make install || {
        echo -e "\033[31mError: Google Test build/install failed. Check build permissions.\033[0m"
        exit 1
    }
    echo -e "\033[32mGoogle Test installed to /usr/local successfully\033[0m"

    safe_cd "${SOURCE_ROOT}/omni_placement/cpp/test"
}

# Function to pip install omni_placement
pip_install_omni_placement_whl() {
    safe_cd "${SOURCE_ROOT}"
    pip install -e . || {
        echo -e "\033[31mError: Failed to pip install omni_placement\033[0m"
        exit 1
    }
    echo -e "\033[32momni_placement installed successfully\033[0m"
}

# Function to run C++ tests with conditional hardware test build
run_cpp_tests() {
    echo "--- Running C++ Tests ---"
    safe_cd "${SOURCE_ROOT}/omni_placement/cpp/test"

    install_gtest
    
    # Always start with a clean build directory
    echo "Recreating a clean build directory for C++ tests..."
    if [ -d "build" ]; then
        rm -rf build
    fi
    mkdir -p build && safe_cd build

    # Output current environment information for debugging
    echo "=== Environment Debug Info ==="
    echo "Current working directory: $(pwd)"
    echo "ASCEND_TOOLKIT_HOME: $ASCEND_TOOLKIT_HOME"
    echo "Python3 path: $(which python3)"
    
    # Check if key files exist
    if [ -n "$ASCEND_TOOLKIT_HOME" ]; then
        echo "Checking Ascend installation:"
        [ -f "$ASCEND_TOOLKIT_HOME/include/acl/acl.h" ] && echo "[OK] acl.h found" || echo "[FAIL] acl.h not found"
        [ -f "$ASCEND_TOOLKIT_HOME/lib64/libascendcl.so" ] && echo "[OK] libascendcl.so found" || echo "[FAIL] libascendcl.so not found"
        [ -f "$ASCEND_TOOLKIT_HOME/lib64/libhccl.so" ] && echo "[OK] libhccl.so found" || echo "[FAIL] libhccl.so not found"
    fi
    echo "=== End Debug Info ==="

    # Determine if we should build hardware tests
    local CMAKE_OPTIONS=""
    if check_npu_presence; then
        CMAKE_OPTIONS="-DBUILD_HARDWARE_TESTS=ON"
        echo "NPU detected. Unit tests with NPU will be built."
    else
        echo "NPU not detected. Only CPU-based unit tests will be built."
    fi

    echo "Configuring the project with cmake... Options: ${CMAKE_OPTIONS:-None}"
    cmake ${CMAKE_OPTIONS} .. || {
        echo -e "\033[31mError: cmake configuration failed\033[0m"
        echo "=== CMake Error Debug ==="
        echo "Please check the CMakeFiles/CMakeError.log for details"
        if [ -f "CMakeFiles/CMakeError.log" ]; then
            echo "Last 20 lines of CMakeError.log:"
            tail -20 CMakeFiles/CMakeError.log
        fi
        echo "=== End CMake Error Debug ==="
        exit 1
    }

    echo "Building the tests with make..."
    make || {
        echo -e "\033[31mError: Make failed for tests\033[0m"
        exit 1
    }

    echo "--- Executing Unit Tests (No Hardware Required) ---"
    ./unit_tests || {
        echo -e "\033[31mError: C++ unit_tests execution failed\033[0m"
        exit 1
    }
    echo -e "\033[32mC++ unit tests completed successfully\033[0m"

    # Conditionally execute hardware tests if they were built
    if [ -f "./unit_tests_with_npu" ]; then
        echo "--- Executing Unit Tests with NPU (NPU Required) ---"
        ./unit_tests_with_npu || {
            echo -e "\033[31mError: C++ unit_tests_with_npu execution failed\033[0m"
            exit 1
        }
        echo -e "\033[32mC++ unit tests with NPU completed successfully\033[0m"
    else
        echo -e "\033[33mUnit tests with NPU were not built (NPU not detected or BUILD_HARDWARE_TESTS=OFF)\033[0m"
    fi
    
    safe_cd ..
}

# Function to run Python tests with fixed test list
run_pytest() {
    local TEST_LIST="tests/test_moe_weights.py tests/test_ada_router_optimizer.py"
    echo "--- Running Python Tests ---"
    echo "Test list: $TEST_LIST"
    safe_cd "${SOURCE_ROOT}"

    for test_file in $TEST_LIST; do
        if [ ! -f "$test_file" ]; then
            echo -e "\033[31mError: Test file $test_file not found\033[0m"
            exit 1
        fi
    done

    pytest $TEST_LIST || {
        echo -e "\033[31mError: pytest failed on $TEST_LIST\033[0m"
        exit 1
    }
    echo -e "\033[32mPython unit tests finished\033[0m"
}

# --- Main execution ---
echo "Starting build and test process..."

check_dependencies

# Source CANN and torch-npu environment from ~/.bashrc
source_cann_env

safe_cd "${SOURCE_ROOT}"

# Simplified execution flow
pip_install_omni_placement_whl
run_cpp_tests
run_pytest

# Return to the directory where the script was started
safe_cd "$START_DIR"
echo -e "\033[32mBuild and test process finished.\033[0m"