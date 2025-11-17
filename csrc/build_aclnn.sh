ROOT_DIR=$1

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  echo "[build_aclnn] ASCEND_HOME_PATH is not set; CANN toolkit/driver not found. Stop building."
  exit 0
fi

ASCEND_SET_ENV="$(dirname "$ASCEND_HOME_PATH")/set_env.sh"
if [[ ! -f "$ASCEND_SET_ENV" ]]; then
  echo "[build_aclnn] Cannot find CANN environment script at $ASCEND_SET_ENV. Stop building."
  exit 0
fi

source "$ASCEND_SET_ENV"

cd csrc/dispatch_ffn_combine

# Clean
rm -rf build_out

# Build
bash build.sh

# Install
rm -rf $ROOT_DIR/CANN
mkdir $ROOT_DIR/CANN

bash ./build_out/custom_opp_dispatch_ffn_combine.run  --install-path=$ROOT_DIR/CANN
source $ROOT_DIR/CANN/vendors/hwcomputing/bin/set_env.bash
