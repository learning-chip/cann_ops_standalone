#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bisheng \
  -fPIC -shared -xcce -O2 -std=c++17 \
  --npu-arch=dav-2201 \
  -I"${SCRIPT_DIR}" \
  -I"${SCRIPT_DIR}/op_kernel" \
  -I"${ASCEND_TOOLKIT_HOME}/compiler/tikcpp/tikcfw" \
  -I"${ASCEND_TOOLKIT_HOME}/compiler/tikcpp/tikcfw/impl" \
  -I"${ASCEND_TOOLKIT_HOME}/compiler/tikcpp/tikcfw/interface" \
  -I"${ASCEND_TOOLKIT_HOME}/include" \
  -I"${ASCEND_TOOLKIT_HOME}/pkg_inc" \
  -I"${ASCEND_TOOLKIT_HOME}/pkg_inc/runtime" \
  -I"${ASCEND_TOOLKIT_HOME}/pkg_inc/profiling" \
  -I"${ASCEND_TOOLKIT_HOME}/aarch64-linux/asc/include" \
  "${SCRIPT_DIR}/stage3_kernel.cpp" \
  -o "${SCRIPT_DIR}/stage3_lib.so"
