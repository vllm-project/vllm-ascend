#!/bin/bash
PROXY_SERVER_SCRIPT=$1
HOST=$2
PORT=$3
PREFILL_PORT=$4
DECODE_PORT=$5

python ${PROXY_SERVER_SCRIPT} \
    --host ${HOST} \
    --port ${PORT} \
    --prefiller-host ${HOST} \
    --prefiller-port ${PREFILL_PORT} \
    --decoder-host ${HOST} \
    --decoder-port ${DECODE_PORT} &