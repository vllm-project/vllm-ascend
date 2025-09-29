START_DIR=$PWD

BUILD_ROOT="$(dirname $(dirname "$(realpath "$0")"))"

# Function to display help information
print_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help                        Display this help information"
    echo "  --ci                              CI build"
    exit 0
}

# 解析长选项
parse_long_option() {
    case "$1" in
        --ci)
            USE_MOCK_MODEL="$2"
            ;;
    esac
    return 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            print_help
            ;;
        --*)
            parse_long_option "$1" "$2"
            shift 2
            ;;
        *)
            echo "未知选项: $1" >&2
            print_help
            ;;
    esac
done

cd $BUILD_ROOT/build
if [ ! -d "dist" ]; then
    mkdir dist
fi

cd $BUILD_ROOT/infer_engines

if [ "$USE_MOCK_MODEL" = "1" ]; then
    # Patch vllm
    ./bash_install_code.sh

    cd vllm
    git apply $BUILD_ROOT/tests/reduce_the_num_of_hidden_layers_of_deepseek_v3_w8a8.patch
else
    # Patch vllm
    ./bash_install_code.sh

    cd vllm
fi

VLLM_TARGET_DEVICE=empty python setup.py bdist_wheel
mv dist/vllm* $BUILD_ROOT/build/dist

cd $BUILD_ROOT
python -m build
mv dist/omni_i* $BUILD_ROOT/build/dist

cd $BUILD_ROOT/omni/accelerators/sched/global_proxy/build/
bash build.sh

cd $BUILD_ROOT/omni/accelerators/sched/omni_proxy/
bash build.sh

cd $BUILD_ROOT/omni/adaptors/vllm/cpp
python setup.py bdist_wheel
mv dist/omni_vllm* $BUILD_ROOT/build/dist

cd $PWD
