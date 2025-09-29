#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

set -e

nginx_conf_file="/usr/local/nginx/conf/nginx.conf"
listen_port="7150"
core_num="4"
start_core_index="0"
prefill_endpoints=""
decode_endpoints=""
log_file="/tmp/nginx_error.log"
log_level="notice"
omni_proxy_pd_policy="sequential"
omni_proxy_model_path=""

dry_run=false
stop=false
rollback=false

print_help() {
    echo "Usage:"
    echo "  $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --nginx-conf-file <path>        Path to nginx config file (default: /usr/local/nginx/conf/nginx.conf)"
    echo "  --listen-port <PORT>            Listening port (default: 7150)"
    echo "  --core-num <N>                  Number of CPU cores to use (default: 4)"
    echo "  --start-core-index <N>          Starting CPU core index (default: 0)"
    echo "  --prefill-endpoints <list>      Comma-separated backend servers for prefill"
    echo "  --decode-endpoints <list>       Comma-separated backend servers for decode"
    echo "  --log-file <path>               Log file path (default: /tmp/nginx_error.log)"
    echo "  --log-level <LEVEL>             Log level (default: notice)"
    echo "  --omni-proxy-pd-policy <policy> sequential or parallel (default: sequential)"
    echo "  --omni-proxy-model-path <path>  Path to model directory (default: unset)"
    echo "  --dry-run                       Only generate nginx config, do not start nginx"
    echo "  --stop                          Stop nginx"
    echo "  --rollback                      Rollback nginx config if backup exists (must be used with --stop)"
    echo "  --help                          Show this help message"
    echo ""
    echo "EXAMPLE:"
    echo "  bash $0 \\"
    echo "      --nginx-conf-file /usr/local/nginx/conf/nginx.conf \\"
    echo "      --listen-port 7000 \\"
    echo "      --core-num 4 \\"
    echo "      --prefill-endpoints 127.0.0.1:9000,127.0.0.2:9001 \\"
    echo "      --decode-endpoints 127.0.0.3:9100,127.0.0.3:9101 \\"
    echo "      --omni-proxy-pd-policy  sequential \\"
    echo "      --omni-proxy-model-path /data/models/deepseek"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nginx-conf-file)
            nginx_conf_file="$2"
            shift 2
            ;;
        --listen-port)
            listen_port="$2"
            shift 2
            ;;
        --core-num)
            core_num="$2"
            shift 2
            ;;
        --start-core-index)
            start_core_index="$2"
            shift 2
            ;;
        --prefill-endpoints)
            prefill_endpoints="$2"
            shift 2
            ;;
        --decode-endpoints)
            decode_endpoints="$2"
            shift 2
            ;;
        --log-file)
            log_file="$2"
            shift 2
            ;;
        --log-level)
            log_level="$2"
            shift 2
            ;;
        --omni-proxy-pd-policy)
            if [[ "$2" != "sequential" && "$2" != "parallel" ]]; then
                echo "Error: --omni-proxy-pd-policy must be 'sequential' or 'parallel'"
                exit 1
            fi
            omni_proxy_pd_policy="$2"
            shift 2
            ;;
        --omni-proxy-model-path)
            omni_proxy_model_path="$2"
            shift 2
            ;;
        --dry-run)
            dry_run=true
            shift 1
            ;;
        --stop)
            stop=true
            shift 1
            ;;
        --rollback)
            rollback=true
            shift 1
            ;;
        --help|-h)
            print_help
            ;;
        *)
            echo "Unknown argument: $1"
            print_help
            ;;
    esac
done

function stop_nginx() {
    while pgrep nginx > /dev/null; do
        echo "Stopping existing nginx ..."
        pgrep nginx | xargs kill -15
        sleep 1
    done
    echo "Nginx stopped."
}

function start_nginx() {
    local nginx_conf_file="$1"
    # nginx -t -c "$nginx_conf_file"
    if [ $? -ne 0 ]; then
        echo "Error: nginx config $nginx_conf_file is invalid. Exiting."
        exit 1
    fi
    echo "Starting nginx with config $nginx_conf_file..."
    nginx -c "$nginx_conf_file"
}

function rollback_nginx_conf() {
    local nginx_conf_file="$1"
    local backup_file="${nginx_conf_file}_bak"
    if [[ -f "$backup_file" ]]; then
        \cp "$backup_file" "$nginx_conf_file"
        echo "Rolled back nginx config to $backup_file"
    else
        echo "No backup config found to rollback."
    fi
}

function gen_affinity_masks() {
    local count=$1
    local masks=()
    for ((i=0; i<count; i++)); do
        local mask=""
        for ((j=0; j<count; j++)); do
            if ((j == i)); then
                mask="${mask}1"
            else
                mask="${mask}0"
            fi
        done
        while ((${#mask} < 16)); do
            mask="0${mask}"
        done
        masks+=("$mask")
    done
    echo "${masks[@]}"
}

function gen_upstream_block() {
    local name="$1"
    local endpoints="$2"
    local block="    upstream $name {\n"
    IFS=',' read -ra list <<< "$endpoints"
    for addr in "${list[@]}"; do
        block+="        server $addr max_fails=3 fail_timeout=10s;\n"
    done
    block+="    }"
    echo -e "$block"
}

function generate_nginx_conf() {
    affinity_masks=$(gen_affinity_masks "$core_num")
    # backup config if exists
    if [[ -f "$nginx_conf_file" ]]; then
        \cp -n "$nginx_conf_file" "${nginx_conf_file}_bak"
    fi

    cat > "$nginx_conf_file" <<EOF
load_module /usr/local/nginx/modules/ngx_http_omni_proxy_module.so;
load_module /usr/local/nginx/modules/ngx_http_set_request_id_module.so;

env PYTHONHASHSEED;
env TORCH_DEVICE_BACKEND_AUTOLOAD;
env VLLM_PLUGINS;
env LD_LIBRARY_PATH;
user root;

worker_processes $core_num;
worker_rlimit_nofile 102400;
worker_cpu_affinity $affinity_masks;

error_log  $log_file  $log_level;

events {
    use epoll;
    accept_mutex off;
    multi_accept on;
    worker_connections 4096;
}

http {
    proxy_http_version 1.1;
    tcp_nodelay on;
    keepalive_requests 1000;
    keepalive_timeout 300;
    client_max_body_size 10M;
    client_body_buffer_size 1M;

    proxy_read_timeout 14400s;
    proxy_connect_timeout 600s;
    proxy_send_timeout 600s;

$(gen_upstream_block "prefill_endpoints" "$prefill_endpoints")
    
$(gen_upstream_block "decode_endpoints" "$decode_endpoints")

    server {
        listen $listen_port reuseport;
        server_name localhost;

        location /v1 {
            set_request_id on;
            omni_proxy decode_endpoints;
            omni_proxy_pd_policy $omni_proxy_pd_policy;
EOF

    if [[ -n "$omni_proxy_model_path" ]]; then
        cat >> "$nginx_conf_file" <<EOF
            omni_proxy_model_path $omni_proxy_model_path;
            omni_proxy_vllm_kv_port_offset 100;
EOF
    fi

    cat >> "$nginx_conf_file" <<EOF
            chunked_transfer_encoding off;
            proxy_buffering off;
            send_timeout 1h;
            postpone_output 0;
        }

        location = /omni_proxy/metrics {
            omni_proxy_metrics on;
            default_type text/plain;
        }

        location ~ ^/prefill_sub(?<orig>/.*)\$ {
            internal;
            proxy_pass http://prefill_endpoints\$orig\$is_args\$args;
            subrequest_output_buffer_size 1M;
        }
    }
}
EOF

    echo "nginx.conf generated at $nginx_conf_file"
}

function do_start() {
    if [[ -z "$prefill_endpoints" || -z "$decode_endpoints" ]]; then
        echo "Error: --prefill-endpoints and --decode-endpoints are required"
        exit 1
    fi

    generate_nginx_conf

    if [ "$dry_run" = true ]; then
        echo "Dry run complete. Configuration generated at $nginx_conf_file."
        exit 0
    fi

    stop_nginx
    start_nginx "$nginx_conf_file"
}

function do_stop() {
    stop_nginx
    if [ "$rollback" = true ]; then
        rollback_nginx_conf "$nginx_conf_file"
    fi
}

function main() {
    if [ "$stop" = false ]; then
        do_start
    else
        do_stop
    fi
}

main