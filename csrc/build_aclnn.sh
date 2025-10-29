ROOT_DIR=$1

source $(dirname $ASCEND_HOME_PATH)/set_env.sh

cd csrc/dispatch_ffn_combine

# Clean
rm -rf build_out

# Build
bash build.sh

# Install
rm -rf $ROOT_DIR/CANN
mkdir $ROOT_DIR/CANN

bash ./build_out/custom_opp_ubuntu_aarch64.run  --install-path=$ROOT_DIR/CANN
source $ROOT_DIR/CANN/vendors/hwcomputing/bin/set_env.bash
