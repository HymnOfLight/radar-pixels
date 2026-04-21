#!/usr/bin/env bash
# Download the classic hyperspectral unmixing benchmark datasets (Samson,
# Jasper Ridge, Urban) from public research mirrors into ./datasets.
# Re-running this script is safe; existing files are kept.

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
TARGET_DIR="${PROJECT_ROOT}/datasets"

mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"

fetch() {
  local url="$1"
  local out="$2"
  if [[ -s "${out}" ]]; then
    echo -e "${YELLOW}[skip]${NC} ${out} already exists."
    return 0
  fi
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 5 -o "${out}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${out}" "${url}"
  else
    echo "Neither curl nor wget is installed." >&2
    exit 1
  fi
}

echo -e "${GREEN}[1/3] Downloading Samson dataset...${NC}"
mkdir -p Samson && (
  cd Samson
  fetch \
    "https://raw.githubusercontent.com/dv-fenix/HyperspecAE/master/data/Samson/Data_Matlab/samson_1.mat" \
    "samson_1.mat"
  fetch \
    "https://raw.githubusercontent.com/dv-fenix/HyperspecAE/master/data/Samson/GroundTruth/end3.mat" \
    "end3.mat"
)

echo -e "${GREEN}[2/3] Downloading Jasper Ridge dataset...${NC}"
mkdir -p JasperRidge && (
  cd JasperRidge
  fetch \
    "https://raw.githubusercontent.com/ricardoborsoi/MultiscaleKernelSURelease/master/DATA/jasperRidge2_R198.mat" \
    "jasperRidge2_R198.mat"
  fetch \
    "https://raw.githubusercontent.com/ricardoborsoi/MultiscaleKernelSURelease/master/DATA/end4.mat" \
    "end4.mat"
)

echo -e "${GREEN}[3/3] Downloading Urban dataset...${NC}"
mkdir -p Urban && (
  cd Urban
  fetch \
    "https://raw.githubusercontent.com/ricardoborsoi/MultiscaleKernelSURelease/master/DATA/Urban_R162.mat" \
    "Urban_R162.mat"
  fetch \
    "https://raw.githubusercontent.com/ricardoborsoi/MultiscaleKernelSURelease/master/DATA/end6_groundTruth.mat" \
    "end6_groundTruth.mat"
)

echo -e "${GREEN}All datasets downloaded to ${TARGET_DIR}.${NC}"
