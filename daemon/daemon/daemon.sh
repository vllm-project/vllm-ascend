export PYTHONPATH=/vllm-workspace/bifrost/daemon/:$PYTHONPATH
export ASCEND_RT_VISIBLE_DEVICES="3,4"

# daemon setup script
# export MDAEMON_LOOP_INTERVAL_MS='1'
export MDAEMON_LOOP_INTERVAL_US='2'
export MDAEMON_DEBUG_LOG='0'
export MDAEMON_EXTEND_THRESHOLD_inGiB='1'
export MDAEMON_REMOVE_THRESHOLD_inGiB='20'
export MDAEMON_EXTEND_BYTES_inMiB='256'
export MDAEMON_REMOVE_BYTES_inMiB='256'
export MDAEMON_DEVICE_TOTAL_CAP_GB='35'

mdaemon --pool 0:16:1.2:12.5 --pool 0:8:1 --pool 0:4:2 --pool 0:2:1 \
        --pool 1:16:1 --pool 1:8:1 --pool 1:4:2 --pool 1:2:1 \
        --refresh-ms 50 --control-enable --control-host 127.0.0.1 --control-port 18080