#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bisheng \
    -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch=dav-2201 \
    -I"${SCRIPT_DIR}" \
    -I"/usr/local/Ascend/cann-8.5.0/aarch64-linux/asc" \
    -I"/usr/local/Ascend/cann-8.5.0/aarch64-linux/asc/include" \
    -I"/usr/local/Ascend/cann-8.5.0/aarch64-linux/asc/include/basic_api" \
    -I"/usr/local/Ascend/cann-8.5.0/aarch64-linux/asc/impl" \
    -I"/usr/local/Ascend/cann-8.5.0/aarch64-linux/asc/impl/basic_api" \
    -I"${ASCEND_TOOLKIT_HOME}/include" \
    -I"${ASCEND_TOOLKIT_HOME}/pkg_inc" \
    -I"${ASCEND_TOOLKIT_HOME}/pkg_inc/runtime" \
    -I"${ASCEND_TOOLKIT_HOME}/pkg_inc/profiling" \
    "${SCRIPT_DIR}/mla_prefill_wrapper.cpp" \
    -o "${SCRIPT_DIR}/mla_prefill_lib.so"
