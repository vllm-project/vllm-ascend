#!/bin/bash
# A5 / chip950 ATK entry (align BSA tests/950 layout).

function set_up_env () {
    source /home/w00841359/venv311/bin/activate
    if [ -n "${ASCEND_TOOLKIT_HOME:-}" ]; then
        # Prefer toolkit setenv when available
        if [ -f "$ASCEND_TOOLKIT_HOME/bin/setenv.bash" ]; then
            # shellcheck disable=SC1090
            source "$ASCEND_TOOLKIT_HOME/bin/setenv.bash"
        elif [ -f "$ASCEND_TOOLKIT_HOME/set_env.sh" ]; then
            # shellcheck disable=SC1090
            source "$ASCEND_TOOLKIT_HOME/set_env.sh"
        fi
    fi
    # Required: custom opp discovery root (toolkit opp, not vendors/custom_*)
    export ASCEND_CUSTOM_OPP_PATH=$ASCEND_TOOLKIT_HOME/opp
    if [ -d "$ASCEND_TOOLKIT_HOME/opp/vendors/custom_transformer/op_api/lib" ]; then
        export LD_LIBRARY_PATH=$ASCEND_TOOLKIT_HOME/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}
    fi
}

function atk_task_gen () {
    atk case -f op_generic_block_sparse_attention.yaml -p generator_genericblocksparseattention.py
}

function atk_execute () {
    atk task -c all_op_generic_block_sparse_attention.json -n nodes.yaml \
        -p aclnn_genericblocksparseattention.py -to 9999
}

set_up_env
# atk_task_gen
# atk_execute
