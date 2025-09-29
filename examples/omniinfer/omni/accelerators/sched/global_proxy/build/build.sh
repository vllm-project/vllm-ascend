#!/bin/bash
set -e

PKG_VERSION=1.0
PKG_RELEASE=1
NGINX_VERSION=1.28.0

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
RPMBUILD=$WORKDIR/rpmbuild

mkdir -p $WORKDIR/SOURCES

rm -rf $RPMBUILD
mkdir -p $RPMBUILD/{BUILD,RPMS,SOURCES,SPECS,SRPMS}

if [ ! -f SOURCES/nginx-${NGINX_VERSION}.tar.gz ]; then
    echo "nginx-${NGINX_VERSION}.tar.gz not found, downloading..."
    wget --no-check-certificate https://nginx.org/download/nginx-${NGINX_VERSION}.tar.gz -O SOURCES/nginx-${NGINX_VERSION}.tar.gz
fi

if [ ! -f SOURCES/global_proxy.tar.gz ]; then
    echo "global_proxy.tar.gz not found, creating..."
    tar --exclude=build -czf SOURCES/global_proxy.tar.gz -C ../.. global_proxy
fi

cp SOURCES/global_proxy.tar.gz $RPMBUILD/SOURCES/
cp SOURCES/nginx-${NGINX_VERSION}.tar.gz $RPMBUILD/SOURCES/
cp SPECS/omni-proxy.spec $RPMBUILD/SPECS/

rpmbuild --define "_topdir $RPMBUILD" --define "debug_package %{nil}" -ba $RPMBUILD/SPECS/omni-proxy.spec

ARCH=$(uname -m)
echo "RPM Packages has been built in $RPMBUILD/RPMS/$ARCH/"
ls -lh $RPMBUILD/RPMS/$ARCH/

DIST_DIR="$(cd "$(dirname "$0")"/../../../../.. && pwd)/build/dist"

cp $RPMBUILD/RPMS/$ARCH/*.rpm "$DIST_DIR"
echo "RPM packages copied to $DIST_DIR"