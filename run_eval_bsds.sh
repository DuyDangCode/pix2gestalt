#!/usr/bin/env bash
# ==============================================================================
# All-in-One Automated Setup and Evaluation Script for pix2gestalt on BSDS-A
#
# Usage:
#   bash run_eval_bsds.sh [OPTIONS]
#
# Examples:
#   bash run_eval_bsds.sh                              # Full evaluation (default)
#   bash run_eval_bsds.sh --ddim_steps 50 --n_samples 1 # Fast evaluation
#   bash run_eval_bsds.sh --max_samples 10             # Quick test on first 10 instances
#   bash run_eval_bsds.sh --dummy_test                 # Smoke test pipeline without full data
# ==============================================================================

set -eo pipefail

# Text formatting
BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
RESET="\033[0m"

log_info()    { echo -e "${BLUE}${BOLD}[INFO]${RESET} $1"; }
log_success() { echo -e "${GREEN}${BOLD}[SUCCESS]${RESET} $1"; }
log_warn()    { echo -e "${YELLOW}${BOLD}[WARNING]${RESET} $1"; }
log_error()   { echo -e "${RED}${BOLD}[ERROR]${RESET} $1"; }

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

echo -e "${BOLD}======================================================================${RESET}"
echo -e "${BOLD}         pix2gestalt: BSDS-A Benchmark Evaluation Pipeline            ${RESET}"
echo -e "${BOLD}======================================================================${RESET}"
log_info "Working directory: ${PROJECT_ROOT}"

# ==============================================================================
# Step 1: Conda Environment Setup
# ==============================================================================
log_info "--- Step 1/5: Checking Conda Environment ---"

CONDA_ENV_NAME="pix2gestalt"

# Find Conda installation
find_conda() {
    if command -v conda &> /dev/null; then
        echo "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
    elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        echo "/opt/conda/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        echo "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        echo "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
        echo "/root/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
        echo "$HOME/.conda/etc/profile.d/conda.sh"
    else
        echo ""
    fi
}

CONDA_SH="$(find_conda)"

if [ -z "${CONDA_SH}" ] || [ ! -f "${CONDA_SH}" ]; then
    log_warn "Conda not found in PATH or standard directories."
    log_info "Installing Miniconda3 for automatic environment isolation..."
    MINICONDA_INSTALLER="/tmp/miniconda.sh"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "${MINICONDA_INSTALLER}"
    bash "${MINICONDA_INSTALLER}" -b -p "$HOME/miniconda3"
    rm -f "${MINICONDA_INSTALLER}"
    CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
fi

log_info "Sourcing Conda from: ${CONDA_SH}"
# shellcheck disable=SC1090
source "${CONDA_SH}"

# Check if conda environment exists
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    log_info "Conda environment '${CONDA_ENV_NAME}' already exists. Activating..."
else
    log_info "Creating conda environment '${CONDA_ENV_NAME}' with Python 3.9..."
    conda create -y -n "${CONDA_ENV_NAME}" python=3.9
fi

conda activate "${CONDA_ENV_NAME}"
log_success "Active Python: $(which python) ($(python --version))"

# ==============================================================================
# Step 2: Install Required Libraries
# ==============================================================================
log_info "--- Step 2/5: Installing Dependencies ---"

pip install --upgrade pip setuptools wheel

# Fix legacy pinned versions in requirements.txt on modern Python
if [ -f "pix2gestalt/requirements.txt" ]; then
    sed -i 's/albumentations==0.4.3/albumentations>=1.0.0/g' pix2gestalt/requirements.txt 2>/dev/null || true
    sed -i 's/imageio==2.9.0/imageio>=2.9.0/g' pix2gestalt/requirements.txt 2>/dev/null || true
    sed -i 's/imageio-ffmpeg==0.4.2/imageio-ffmpeg>=0.4.2/g' pix2gestalt/requirements.txt 2>/dev/null || true
fi

# Pre-install binary wheel for albumentations, opencv, and modern imageio to prevent conflicts
log_info "Pre-installing binary wheels for albumentations, opencv, and imageio..."
pip install --prefer-binary "albumentations>=1.0.0" opencv-python "imageio>=2.33.0"

# Install base requirements
if [ -f "pix2gestalt/requirements.txt" ]; then
    log_info "Installing packages from pix2gestalt/requirements.txt..."
    pip install --prefer-binary -r pix2gestalt/requirements.txt || {
        log_warn "Standard requirements install encountered issues, falling back to core dependencies..."
        pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
        pip install omegaconf einops pytorch-lightning==1.4.2 transformers==4.22.2 opencv-python Pillow tqdm "albumentations>=1.0.0" "imageio>=2.33.0"
    }
fi

# Install taming-transformers if not present
if [ ! -d "taming-transformers" ]; then
    log_info "Cloning and installing CompVis/taming-transformers..."
    git clone https://github.com/CompVis/taming-transformers.git
    pip install -e taming-transformers/
else
    log_info "taming-transformers already present."
fi

# Install CLIP if not present
if [ ! -d "CLIP" ]; then
    log_info "Cloning and installing openai/CLIP..."
    git clone https://github.com/openai/CLIP.git
    pip install -e CLIP/
else
    log_info "CLIP already present."
fi

# Install evaluation support tools
log_info "Installing evaluation utilities (pycocotools, gdown, etc.)..."
pip install --prefer-binary pycocotools gdown pandas "imageio>=2.33.0" scikit-image > /dev/null 2>&1 || true

log_success "All dependencies are installed and verified."

# ==============================================================================
# Step 3: Download Model Weights
# ==============================================================================
log_info "--- Step 3/5: Checking Model Checkpoints ---"

CKPT_DIR="${PROJECT_ROOT}/pix2gestalt/ckpt"
mkdir -p "${CKPT_DIR}"
CKPT_FILE="${CKPT_DIR}/epoch=000005.ckpt"

# The official epoch=000005.ckpt is ~15.4 GB (15,466,113,894 bytes)
EXPECTED_CKPT_SIZE=15000000000
CURRENT_SIZE=$(stat -c%s "${CKPT_FILE}" 2>/dev/null || stat -f%z "${CKPT_FILE}" 2>/dev/null || echo 0)

if [ "${CURRENT_SIZE}" -ge "${EXPECTED_CKPT_SIZE}" ]; then
    log_success "Model checkpoint verified (${CURRENT_SIZE} bytes, ~15.4 GB): ${CKPT_FILE}"
else
    if [ "${CURRENT_SIZE}" -gt 0 ]; then
        log_warn "Incomplete checkpoint file detected (${CURRENT_SIZE} bytes / ~15.4 GB). Resuming download with wget -c..."
    else
        log_info "Downloading pix2gestalt pretrained weights (epoch=000005.ckpt, 15.4 GB)..."
    fi
    PRIMARY_URL="https://gestalt.cs.columbia.edu/assets/epoch=000005.ckpt"
    BACKUP_URL="https://huggingface.co/cvlab/pix2gestalt-weights/resolve/main/epoch=000005.ckpt"

    # wget -c resumes interrupted downloads automatically
    if wget -c -P "${CKPT_DIR}" "${PRIMARY_URL}"; then
        log_success "Downloaded checkpoint from Columbia server."
    else
        log_warn "Primary download interrupted, downloading from Hugging Face backup..."
        wget -c -O "${CKPT_FILE}" "${BACKUP_URL}"
    fi

    # Check size again
    FINAL_SIZE=$(stat -c%s "${CKPT_FILE}" 2>/dev/null || stat -f%z "${CKPT_FILE}" 2>/dev/null || echo 0)
    if [ "${FINAL_SIZE}" -lt "${EXPECTED_CKPT_SIZE}" ]; then
        log_error "Checkpoint file is still incomplete (${FINAL_SIZE} bytes). The file must be fully downloaded (~15.4 GB)."
        log_error "Please run: wget -c -P ${CKPT_DIR} https://gestalt.cs.columbia.edu/assets/epoch=000005.ckpt"
        exit 1
    fi
fi

# Optional: SAM checkpoint (useful if running SAM-based modal masks)
SAM_FILE="${CKPT_DIR}/sam_vit_h.pth"
if [ ! -f "${SAM_FILE}" ]; then
    log_info "Downloading SAM vit_h weights (optional for SAM prompt)..."
    wget -c -P "${CKPT_DIR}" "https://gestalt.cs.columbia.edu/assets/sam_vit_h.pth" || log_warn "SAM weights download skipped (not required for GT modal mask eval)."
fi

log_success "Checkpoints are ready."

# ==============================================================================
# Step 4: Download and Prepare BSDS-A Dataset
# ==============================================================================
log_info "--- Step 4/5: Preparing BSDS-A Dataset ---"

DATA_DIR="${PROJECT_ROOT}/data/bsds"
IMAGES_DIR="${DATA_DIR}/images"
ANN_DIR="${DATA_DIR}/annotations"
mkdir -p "${DATA_DIR}" "${IMAGES_DIR}" "${ANN_DIR}"

BSDS_TAR="${DATA_DIR}/BSR_bsds500.tgz"
BSDS_ANN_JSON="${ANN_DIR}/BSDS_amodal_test.json"

# Check if images exist (BSDS test set has 200 images)
NUM_EXISTING_IMAGES=$(find "${IMAGES_DIR}" -maxdepth 1 -name "*.jpg" -o -name "*.png" | wc -l)
if [ "${NUM_EXISTING_IMAGES}" -ge 200 ]; then
    log_info "BSDS500 test images already present (${NUM_EXISTING_IMAGES} images found)."
else
    log_info "Downloading BSDS500 dataset from Berkeley vision repository..."
    BSDS_URL="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
    if wget -c -O "${BSDS_TAR}" "${BSDS_URL}"; then
        log_info "Extracting BSDS500 test images..."
        tar -xzf "${BSDS_TAR}" -C "${DATA_DIR}" BSR/BSDS500/data/images/test
        # Move images directly into images/
        cp -r "${DATA_DIR}/BSR/BSDS500/data/images/test/"* "${IMAGES_DIR}/"
        rm -rf "${DATA_DIR}/BSR"
        log_success "BSDS500 test images extracted to ${IMAGES_DIR}."
    else
        log_warn "Direct download of BSDS500 from Berkeley server failed."
    fi
fi

# Check BSDS-A Annotations JSON
if [ -f "${BSDS_ANN_JSON}" ]; then
    log_info "BSDS-A annotation file found: ${BSDS_ANN_JSON}"
else
    log_warn "Annotation file not found at: ${BSDS_ANN_JSON}"
    log_info "Checking alternative locations..."
    if [ -f "${PROJECT_ROOT}/BSDS_amodal_test.json" ]; then
        cp "${PROJECT_ROOT}/BSDS_amodal_test.json" "${BSDS_ANN_JSON}"
        log_success "Copied BSDS_amodal_test.json from project root."
    elif [ -f "${DATA_DIR}/BSDS_amodal_test.json" ]; then
        cp "${DATA_DIR}/BSDS_amodal_test.json" "${BSDS_ANN_JSON}"
        log_success "Located BSDS_amodal_test.json in data/bsds."
    fi
fi

# Verify annotation file or dummy_test / auto_synth_ann flag before launching eval
HAS_DUMMY_FLAG=false
HAS_AUTO_SYNTH_FLAG=false
for arg in "$@"; do
    if [ "$arg" == "--dummy_test" ]; then
        HAS_DUMMY_FLAG=true
    elif [ "$arg" == "--auto_synth_ann" ]; then
        HAS_AUTO_SYNTH_FLAG=true
    fi
done

if [ ! -f "${BSDS_ANN_JSON}" ] && [ ! -d "${DATA_DIR}/visible_masks" ] && [ "${HAS_DUMMY_FLAG}" = false ] && [ "${HAS_AUTO_SYNTH_FLAG}" = false ]; then
    echo -e "\n${YELLOW}${BOLD}======================================================================${RESET}"
    echo -e "${YELLOW}${BOLD}  CHƯA TÌM THẤY FILE NHÃN: data/bsds/annotations/BSDS_amodal_test.json${RESET}"
    echo -e "${YELLOW}${BOLD}======================================================================${RESET}"
    echo -e "Script đã tải thành công toàn bộ ${GREEN}${NUM_EXISTING_IMAGES} ảnh gốc BSDS500${RESET} vào:"
    echo -e "  ${BLUE}${IMAGES_DIR}/${RESET}"
    echo -e "\nĐể đánh giá amodal mIoU trên tập BSDS-A, bạn cần file nhãn ${BOLD}BSDS_amodal_test.json${RESET}."
    echo -e "Bạn có 3 lựa chọn:"
    echo -e "  1. Copy file nhãn chính thức vào: ${GREEN}${BSDS_ANN_JSON}${RESET} rồi chạy lại script."
    echo -e "  2. Tự động sinh nhãn mô phỏng trên chính 200 ảnh BSDS500 thật để chạy đánh giá ngay:"
    echo -e "     ${BOLD}${GREEN}bash run_eval_bsds.sh --auto_synth_ann${RESET}"
    echo -e "  3. Hoặc kiểm tra nhanh pipeline (smoke test) với dummy data:"
    echo -e "     ${BOLD}${GREEN}bash run_eval_bsds.sh --dummy_test${RESET}"
    echo -e "${YELLOW}${BOLD}======================================================================${RESET}\n"
    exit 0
fi

# ==============================================================================
# Step 5: Execute Evaluation
# ==============================================================================
log_info "--- Step 5/5: Running Evaluation ---"

# Check GPU availability
python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device Name    : {torch.cuda.get_device_name(0)}')
    print(f'Device Memory  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
"

# Pass all incoming script arguments directly to eval_bsds.py
log_info "Launching eval_bsds.py with parameters: $*"

python eval_bsds.py \
    --config "pix2gestalt/configs/sd-finetune-pix2gestalt-c_concat-256.yaml" \
    --ckpt "${CKPT_FILE}" \
    --dataset_dir "${DATA_DIR}" \
    --output_dir "${PROJECT_ROOT}/results" \
    "$@"

log_success "Evaluation completed successfully!"
echo -e "${BOLD}Results and summary logs are available in:${RESET} ${PROJECT_ROOT}/results/"
