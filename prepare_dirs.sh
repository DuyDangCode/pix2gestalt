#!/usr/bin/env bash
# Tạo sẵn cấu trúc thư mục chuẩn cho eval_bsds.py và run_eval_bsds.sh
# Cách 1 (chạy trên Server): bash prepare_dirs.sh
# Cách 2 (chạy từ máy Local qua SSH): bash prepare_dirs.sh root@116.109.104.17 10522 [/workspace/pix2gestalt]

set -eo pipefail

if [ $# -ge 1 ]; then
    REMOTE_HOST="$1"
    SSH_PORT="${2:-22}"
    REMOTE_ROOT="${3:-/workspace/pix2gestalt}"
    echo "[INFO] Đang tạo thư mục trên server ${REMOTE_HOST} (port ${SSH_PORT}) tại ${REMOTE_ROOT}..."
    ssh -p "${SSH_PORT}" "${REMOTE_HOST}" \
        "mkdir -p '${REMOTE_ROOT}/data/bsds/annotations' '${REMOTE_ROOT}/data/bsds/images' '${REMOTE_ROOT}/pix2gestalt/ckpt' '${REMOTE_ROOT}/results'"
    echo "[SUCCESS] Đã tạo xong: ${REMOTE_ROOT}/data/bsds/annotations"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    mkdir -p \
        "${PROJECT_ROOT}/data/bsds/annotations" \
        "${PROJECT_ROOT}/data/bsds/images" \
        "${PROJECT_ROOT}/pix2gestalt/ckpt" \
        "${PROJECT_ROOT}/results"
    echo "[SUCCESS] Đã tạo sẵn các thư mục tại: ${PROJECT_ROOT}/data/bsds/annotations"
fi

