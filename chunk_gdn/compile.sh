#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/compile_stage1.sh"
bash "${SCRIPT_DIR}/compile_stage2.sh"
bash "${SCRIPT_DIR}/compile_stage3.sh"

