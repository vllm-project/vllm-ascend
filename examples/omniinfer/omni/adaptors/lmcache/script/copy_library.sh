#!/bin/bash
set -e

LIB_PATH=$1

SO_NAMES=(
    libetcd_wrapper.so
    libcurl.so.4
    libibverbs.so.1
    libjsoncpp.so.25
    libnuma.so.1
    libglog.so.2
    libgflags.so.2.2
    libnghttp2.so.14
    libidn2.so.0
    libssh.so.4
    libpsl.so.5
    libssl.so.1.1
    libcrypto.so.1.1
    libgssapi_krb5.so.2
    libkrb5.so.3
    libk5crypto.so.3
    libcom_err.so.2
    libldap.so.2
    liblber.so.2
    libbrotlidec.so.1
    libnl-route-3.so.200
    libnl-3.so.200
    libunwind.so.8
    libunistring.so.2
    libkrb5support.so.0
    libkeyutils.so.1
    libsasl2.so.3
    libbrotlicommon.so.1
    libpcre2-8.so.0
)

SEARCH_DIRS=(
    /lib
    /lib64
    /usr/lib
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64
)

echo "Locate and copy the following shared libraries to the directory: $LIB_PATH"
for so in "${SO_NAMES[@]}"; do
    found=0
    for dir in "${SEARCH_DIRS[@]}"; do
        fullpath=$(find "$dir" -name "$so" 2>/dev/null | head -n 1)
        if [[ -n "$fullpath" && -f "$fullpath" ]]; then
          cp -rv "$fullpath" $LIB_PATH
          if [[ -L "$fullpath" ]]; then
            realfile=$(readlink -f "$fullpath")
            cp -rv "$realfile" $LIB_PATH
          fi
          found=1
          break
        fi
    done
    if [[ "$found" -eq 0 ]]; then
      echo "Library not found: $so" >&2
    fi
done

echo "All available libraries have been copied to: $LIB_PATH/"
