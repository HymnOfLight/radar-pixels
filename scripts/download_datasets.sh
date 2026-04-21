#!/usr/bin/env bash
# Download the classic hyperspectral unmixing benchmark datasets (Samson,
# Jasper Ridge, Urban) from public research mirrors into ./datasets.
# Re-running this script is safe; existing files are kept.

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
TARGET_DIR="${PROJECT_ROOT}/datasets"

mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"

# Download a file, retrying through a list of mirror URLs and skipping when the
# file already exists on disk.
fetch_with_mirrors() {
  local out="$1"
  shift
fetch() {
  local url="$1"
  local out="$2"
  if [[ -s "${out}" ]]; then
    echo -e "${YELLOW}[skip]${NC} ${out} already exists."
    return 0
  fi
  local url
  for url in "$@"; do
    echo "  -> trying ${url}"
    if command -v curl >/dev/null 2>&1; then
      if curl -fL --retry 3 --retry-delay 5 -o "${out}.part" "${url}"; then
        mv "${out}.part" "${out}"
        echo -e "  ${GREEN}[ok]${NC} saved ${out}"
        return 0
      fi
    elif command -v wget >/dev/null 2>&1; then
      if wget -q -O "${out}.part" "${url}"; then
        mv "${out}.part" "${out}"
        echo -e "  ${GREEN}[ok]${NC} saved ${out}"
        return 0
      fi
    else
      echo "Neither curl nor wget is installed." >&2
      exit 1
    fi
    rm -f "${out}.part"
  done
  echo -e "${RED}[fail]${NC} unable to download ${out} from any mirror." >&2
  return 1
}

echo -e "${GREEN}[1/3] Downloading Samson dataset (95x95x156, 3 endmembers)...${NC}"
mkdir -p Samson && (
  cd Samson
  fetch_with_mirrors "Samson.mat" \
    "https://raw.githubusercontent.com/gaetanosettembre/data_unmixing/main/Datasets/Samson/Samson.mat"
  fetch_with_mirrors "Samson_GT.mat" \
    "https://raw.githubusercontent.com/gaetanosettembre/data_unmixing/main/Datasets/Samson/Samson_GT.mat"
)

echo -e "${GREEN}[2/3] Downloading Jasper Ridge dataset (100x100x198, 4 endmembers)...${NC}"
mkdir -p JasperRidge && (
  cd JasperRidge
  fetch_with_mirrors "jasperRidge2_R198.mat" \
    "https://raw.githubusercontent.com/gaetanosettembre/data_unmixing/main/Datasets/Jasper/jasperRidge2_R198.mat" \
    "https://raw.githubusercontent.com/ricardoborsoi/MultiscaleKernelSURelease/master/DATA/jasperRidge2_R198.mat"
  fetch_with_mirrors "Jasper_GT.mat" \
    "https://raw.githubusercontent.com/gaetanosettembre/data_unmixing/main/Datasets/Jasper/Jasper_GT.mat"
)

echo -e "${GREEN}[3/3] Downloading Urban dataset (307x307x162, 4-6 endmembers)...${NC}"
mkdir -p Urban && (
  cd Urban
  fetch_with_mirrors "Urban.mat" \
    "https://raw.githubusercontent.com/gaetanosettembre/data_unmixing/main/Datasets/Urban/Urban.mat"
  fetch_with_mirrors "end4_groundTruth.mat" \
    "https://raw.githubusercontent.com/gaetanosettembre/data_unmixing/main/Datasets/Urban/end4_groundTruth.mat"
  fetch_with_mirrors "end5_groundTruth.mat" \
    "https://raw.githubusercontent.com/gaetanosettembre/data_unmixing/main/Datasets/Urban/end5_groundTruth.mat"
  fetch_with_mirrors "end6_groundTruth.mat" \
    "https://raw.githubusercontent.com/gaetanosettembre/data_unmixing/main/Datasets/Urban/end6_groundTruth.mat"
)

echo -e "${GREEN}All datasets downloaded to ${TARGET_DIR}.${NC}"
echo "Files layout:"
(cd "${TARGET_DIR}" && find . -maxdepth 3 -type f | sort)
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
