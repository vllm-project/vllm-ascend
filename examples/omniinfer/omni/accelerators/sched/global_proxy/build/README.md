# Building the omni-proxy RPM Package

This guide describes how to build the omni-proxy RPM package, which includes custom global proxy modules for Nginx.

## Prerequisites

- Required packages: `gcc`, `make`, `zlib-devel`, `pcre-devel`, `openssl-devel`, `wget`, `tar`, `rpmbuild`

## Steps

1. **Run the build script**:
    ```bash
    bash build.sh
    ```

2. **Find the generated RPMs**:
    - The built RPM files will be located in the `omniinfer/omni/accelerators/dist` directory.
