Name:           omni-proxy
Version:        1.0
Release:        1%{?dist}
Summary:        Nginx custom global proxy modules

License:        MIT
Source0:        nginx-1.28.0.tar.gz
Source1:        global_proxy.tar.gz

BuildRequires:  gcc make zlib-devel pcre-devel openssl-devel

Conflicts:      nginx

%description
Custom global proxy modules for Nginx, built for PD Disaggregation.

%prep
%setup -q -c -T -a 0
tar -zxf %{SOURCE1} -C .

%build
cd nginx-1.28.0
CFLAGS="-O2" ./configure --sbin-path=/usr/sbin \
    --add-dynamic-module=../global_proxy/modules/ngx_http_prefill_module \
    --add-dynamic-module=../global_proxy/modules/ngx_http_prefill_refactor_module \
    --add-dynamic-module=../global_proxy/modules/ngx_http_set_request_id_module \
    --add-dynamic-module=../global_proxy/modules/ngx_http_internal_metrics_module \
    --add-dynamic-module=../global_proxy/lb_sdk/modules/ngx_http_upstream_length_balance_module \
    --add-dynamic-module=../global_proxy/lb_sdk/modules/ngx_http_upstream_greedy_timeout_module \
    --add-dynamic-module=../global_proxy/lb_sdk/modules/ngx_http_upstream_prefill_score_balance_module \
    --add-dynamic-module=../global_proxy/lb_sdk/modules/ngx_http_upstream_weighted_least_active_module \
    --add-dynamic-module=../global_proxy/lb_sdk/modules/ngx_http_upstream_pd_score_balance_module \
    --add-dynamic-module=../global_proxy/lb_sdk/modules/ngx_http_upstream_auto_balance_controller_module \
    --add-dynamic-module=../global_proxy/lb_sdk/modules/ngx_http_upstream_least_total_load_module \
    --without-http_gzip_module \
    --with-ld-opt="-lm"
make -j
make modules

%install
mkdir -p %{buildroot}/usr/sbin
cp nginx-1.28.0/objs/nginx %{buildroot}/usr/sbin/nginx

mkdir -p %{buildroot}/usr/local/nginx/
cp -a nginx-1.28.0/conf  %{buildroot}/usr/local/nginx/
cp -a nginx-1.28.0/html  %{buildroot}/usr/local/nginx/
cp -a nginx-1.28.0/objs/nginx %{buildroot}/usr/local/nginx/nginx
mkdir -p %{buildroot}/usr/local/nginx/logs/
mkdir -p %{buildroot}/usr/local/nginx/modules/
cp nginx-1.28.0/objs/*.so %{buildroot}/usr/local/nginx/modules/

%files
/usr/sbin/nginx
/usr/local/nginx/nginx
/usr/local/nginx/conf/*
/usr/local/nginx/html/*
%dir /usr/local/nginx/logs
/usr/local/nginx/modules/*.so

%changelog
* Wed Jul 31 2025 Huawei - 1.0-1
- Initial build