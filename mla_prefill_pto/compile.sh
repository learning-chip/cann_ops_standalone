#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PTO_ISA_ROOT="${PTO_ISA_ROOT:-/workdir/pto-isa-master}"
PARENT_CCE="${SCRIPT_DIR}/../mla_prefill_cce"

bisheng \
    -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch=dav-2201 \
    -I"${SCRIPT_DIR}" \
    -I"${SCRIPT_DIR}/include" \
    -I"${PTO_ISA_ROOT}/include" \
    -I"${PARENT_CCE}" \
    -I${ASCEND_TOOLKIT_HOME}/compiler/tikcpp/tikcfw \
    -I${ASCEND_TOOLKIT_HOME}/compiler/tikcpp/tikcfw/impl \
    -I${ASCEND_TOOLKIT_HOME}/compiler/tikcpp/tikcfw/interface \
    -I"${ASCEND_TOOLKIT_HOME}/include" \
    -I"${ASCEND_TOOLKIT_HOME}/pkg_inc" \
    -I"${ASCEND_TOOLKIT_HOME}/pkg_inc/runtime" \
    -I"${ASCEND_TOOLKIT_HOME}/pkg_inc/profiling" \
    "${SCRIPT_DIR}/mla_prefill.cpp" \
    -o "${SCRIPT_DIR}/mla_prefill_lib.so"
