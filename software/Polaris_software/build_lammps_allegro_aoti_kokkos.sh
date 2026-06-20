#!/bin/bash
# ==============================================================================
# LAMMPS Build Script with Allegro/NequIP, AOTI, and Kokkos Support
# ==============================================================================
# 
# This script builds LAMMPS with:
#   - Allegro/NequIP pair style (pair_allegro)
#   - AOTI (Ahead-Of-Time Inductor) compilation support (requires PyTorch >= 2.6)
#   - Kokkos package (REQUIRED for pair_allegro)
#   - GPU support via CUDA
#
# Based on recommendations from: https://github.com/mir-group/pair_nequip_allegro
#
# REQUIREMENTS:
#   - LAMMPS source code (10 September 2025 release or newer)
#   - pair_nequip_allegro repository (cloned and patched)
#   - Conda environment with PyTorch >= 2.6.0 (for AOTI support)
#   - CUDA toolkit (full installation, not just conda CUDA)
#   - MPI (OpenMPI or similar)
#   - GCC compiler (version 12+ recommended)
#   - CMake
#
# USAGE:
#   ./build_lammps_allegro_aoti_kokkos.sh
#
# ENVIRONMENT VARIABLES (optional, with defaults):
#   LAMMPS_DIR          - Path to LAMMPS source directory (default: ./lammps)
#   PAIR_NEQUIP_DIR     - Path to pair_nequip_allegro repository (default: ./pair_nequip_allegro)
#   CONDA_ENV_NAME      - Conda environment name (default: nequip_allegro)
#   CONDA_BASE          - Conda base directory (default: $HOME/miniforge3)
#   BUILD_DIR           - Build directory (default: $LAMMPS_DIR/build-gpu-allegro-kokkos)
#   INSTALL_DIR         - Install directory (default: $BUILD_DIR/install)
#   CLEAN_BUILD         - Set to "yes" to clean build directory before building
#
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}==============================================================================${NC}"
echo -e "${BLUE}LAMMPS Build Script with Allegro/NequIP, AOTI, and Kokkos Support${NC}"
echo -e "${BLUE}==============================================================================${NC}"
echo ""

# ==============================================================================
# Configuration: Set defaults and read environment variables
# ==============================================================================

# Set default paths (can be overridden by environment variables)
LAMMPS_DIR="${LAMMPS_DIR:-${HOME}/lammps}"
PAIR_NEQUIP_DIR="${PAIR_NEQUIP_DIR:-${HOME}/pair_nequip_allegro}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-nequip_allegro}"
CONDA_BASE="${CONDA_BASE:-${HOME}/miniforge3}"
BUILD_DIR="${BUILD_DIR:-${LAMMPS_DIR}/build-gpu-allegro-kokkos}"
INSTALL_DIR="${INSTALL_DIR:-${BUILD_DIR}/install}"

echo -e "${GREEN}Configuration:${NC}"
echo "  LAMMPS_DIR:      ${LAMMPS_DIR}"
echo "  PAIR_NEQUIP_DIR: ${PAIR_NEQUIP_DIR}"
echo "  CONDA_ENV_NAME:  ${CONDA_ENV_NAME}"
echo "  CONDA_BASE:      ${CONDA_BASE}"
echo "  BUILD_DIR:       ${BUILD_DIR}"
echo "  INSTALL_DIR:     ${INSTALL_DIR}"
echo ""

# ==============================================================================
# Step 1: Load system modules (if available)
# ==============================================================================

if command -v module &> /dev/null; then
    echo -e "${GREEN}[Step 1/10] Loading required modules...${NC}"
    module load mpi/openmpi-x86_64 2>/dev/null || echo -e "${YELLOW}  Warning: Could not load mpi/openmpi-x86_64${NC}"
    module load gcc-toolset/12 2>/dev/null || echo -e "${YELLOW}  Warning: Could not load gcc-toolset/12${NC}"
    
    # Try to load CUDA module (try multiple versions)
    CUDA_MODULE_LOADED=false
    for cuda_module in "cuda" "cuda/12.6" "cuda/12.2" "cuda/12.0" "cuda/11.8"; do
        if module load "${cuda_module}" 2>/dev/null; then
            echo -e "${GREEN}  Successfully loaded CUDA module: ${cuda_module}${NC}"
            CUDA_MODULE_LOADED=true
            break
        fi
    done
    if [ "$CUDA_MODULE_LOADED" = false ]; then
        echo -e "${YELLOW}  Warning: Could not load any CUDA module${NC}"
    fi
else
    echo -e "${YELLOW}[Step 1/10] Module system not available, skipping module loading${NC}"
fi

# ==============================================================================
# Step 2: Check GPU availability
# ==============================================================================

echo -e "${GREEN}[Step 2/10] Checking GPU availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}  GPU node detected:${NC}"
    nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader | head -1 | sed 's/^/    /'
else
    echo -e "${YELLOW}  Warning: nvidia-smi not found. Build will proceed but GPU support may not work.${NC}"
fi

# ==============================================================================
# Step 3: Detect and configure CUDA
# ==============================================================================

echo -e "${GREEN}[Step 3/10] Detecting CUDA installation...${NC}"

# Set up CUDA environment - prioritize CUDA_HOME from module if set
if [ -n "${CUDA_HOME:-}" ] && [ -d "$CUDA_HOME" ]; then
    echo -e "${GREEN}  Using CUDA_HOME from environment/module: ${CUDA_HOME}${NC}"
elif command -v nvcc &> /dev/null; then
    # If nvcc is in PATH, try to find CUDA_HOME from it
    NVCC_PATH=$(which nvcc)
    if [ -n "$NVCC_PATH" ]; then
        CUDA_HOME_CANDIDATE=$(dirname "$(dirname "$NVCC_PATH")")
        if [ -d "$CUDA_HOME_CANDIDATE" ] && [ -f "$CUDA_HOME_CANDIDATE/bin/nvcc" ]; then
            export CUDA_HOME="$CUDA_HOME_CANDIDATE"
            echo -e "${GREEN}  Detected CUDA_HOME from nvcc path: ${CUDA_HOME}${NC}"
        fi
    fi
fi

# If CUDA_HOME still not set, check common installation paths
if [ -z "${CUDA_HOME:-}" ] || [ ! -d "$CUDA_HOME" ]; then
    echo -e "${YELLOW}  CUDA_HOME not set or invalid, checking common paths...${NC}"
    for cuda_path in "/usr/local/cuda-12.6" "/usr/local/cuda-12.2" "/usr/local/cuda-12.0" \
                     "/usr/local/cuda-11.8" "/usr/local/cuda" "/opt/cuda"; do
        if [ -d "$cuda_path" ] && [ -f "$cuda_path/bin/nvcc" ]; then
            export CUDA_HOME="$cuda_path"
            echo -e "${GREEN}  Found CUDA at: ${CUDA_HOME}${NC}"
            break
        fi
    done
fi

# Verify CUDA installation
if [ -z "${CUDA_HOME:-}" ] || [ ! -d "$CUDA_HOME" ]; then
    echo -e "${RED}  Error: CUDA_HOME not found or invalid!${NC}"
    echo -e "${RED}  Please ensure CUDA is installed or a CUDA module is loaded.${NC}"
    echo -e "${YELLOW}  You can set CUDA_HOME manually: export CUDA_HOME=/path/to/cuda${NC}"
    exit 1
fi

# Set CUDA paths
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

# Verify CUDA is accessible
if command -v nvcc &> /dev/null; then
    echo -e "${GREEN}  CUDA compiler found:${NC}"
    nvcc --version | head -3 | sed 's/^/    /'
else
    echo -e "${RED}  Error: nvcc not found in PATH after setting CUDA_HOME!${NC}"
    echo -e "${RED}  CUDA_HOME: ${CUDA_HOME}${NC}"
    exit 1
fi

# ==============================================================================
# Step 4: Verify LAMMPS source directory
# ==============================================================================

echo -e "${GREEN}[Step 4/10] Verifying LAMMPS source directory...${NC}"
if [ ! -d "${LAMMPS_DIR}" ]; then
    echo -e "${RED}  Error: LAMMPS directory not found at ${LAMMPS_DIR}${NC}"
    echo -e "${YELLOW}  Please set LAMMPS_DIR environment variable or clone LAMMPS:${NC}"
    echo -e "${YELLOW}    git clone https://github.com/lammps/lammps.git ${LAMMPS_DIR}${NC}"
    exit 1
fi

if [ ! -f "${LAMMPS_DIR}/CMakeLists.txt" ]; then
    echo -e "${RED}  Error: ${LAMMPS_DIR} does not appear to be a LAMMPS source directory${NC}"
    exit 1
fi

echo -e "${GREEN}  ✓ LAMMPS source directory found${NC}"

# ==============================================================================
# Step 5: Setup conda environment
# ==============================================================================

echo -e "${GREEN}[Step 5/10] Setting up conda environment...${NC}"
NEQUIP_ENV="${CONDA_BASE}/envs/${CONDA_ENV_NAME}"

if [ ! -d "${NEQUIP_ENV}" ] || [ ! -f "${NEQUIP_ENV}/bin/python" ]; then
    echo -e "${RED}  Error: Conda environment '${CONDA_ENV_NAME}' not found at ${NEQUIP_ENV}${NC}"
    echo -e "${YELLOW}  Please create the environment first:${NC}"
    echo -e "${YELLOW}    conda create -n ${CONDA_ENV_NAME} python=3.10 -y${NC}"
    echo -e "${YELLOW}    conda activate ${CONDA_ENV_NAME}${NC}"
    echo -e "${YELLOW}    pip install torch>=2.6.0 nequip${NC}"
    exit 1
fi

# Ensure environment is in PATH
export PATH="${NEQUIP_ENV}/bin:${PATH}"

# Try to activate via conda if available
if [ -f "${NEQUIP_ENV}/bin/activate" ]; then
    # Initialize conda if not already initialized
    if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
        # Try to source conda initialization
        if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
            source "${CONDA_BASE}/etc/profile.d/conda.sh" 2>/dev/null || true
        elif [ -f "${HOME}/.bashrc" ] && grep -q "conda initialize" "${HOME}/.bashrc"; then
            source "${HOME}/.bashrc" 2>/dev/null || true
        fi
    fi
    # Activate the environment
    if command -v conda &> /dev/null; then
        conda activate "${CONDA_ENV_NAME}" 2>/dev/null || {
            echo -e "${YELLOW}  Warning: Could not activate conda environment via conda command${NC}"
            echo -e "${YELLOW}  Using environment directly via PATH${NC}"
        }
    fi
fi

echo -e "${GREEN}  ✓ Conda environment configured${NC}"

# ==============================================================================
# Step 6: Check PyTorch version and AOTI support
# ==============================================================================

echo -e "${GREEN}[Step 6/10] Checking PyTorch version and AOTI support...${NC}"
PYTHON_CMD=$(which python 2>/dev/null || echo "${NEQUIP_ENV}/bin/python")

if [ ! -f "${PYTHON_CMD}" ]; then
    echo -e "${RED}  Error: Python not found in conda environment${NC}"
    exit 1
fi

PYTORCH_VERSION=$("${PYTHON_CMD}" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
if [ -z "${PYTORCH_VERSION}" ]; then
    echo -e "${RED}  Error: PyTorch not found in conda environment${NC}"
    echo -e "${YELLOW}  Please install PyTorch: pip install torch>=2.6.0${NC}"
    exit 1
fi

echo -e "${GREEN}  PyTorch version: ${PYTORCH_VERSION}${NC}"

# Check if PyTorch version supports AOTI (requires >= 2.6)
PYTORCH_MAJOR=$("${PYTHON_CMD}" -c "import torch; print(torch.__version__.split('.')[0])" 2>/dev/null)
PYTORCH_MINOR=$("${PYTHON_CMD}" -c "import torch; print(torch.__version__.split('.')[1])" 2>/dev/null)

ENABLE_AOTI=false
if [ "${PYTORCH_MAJOR}" -gt 2 ] || ([ "${PYTORCH_MAJOR}" -eq 2 ] && [ "${PYTORCH_MINOR}" -ge 6 ]); then
    # Check if AOTI headers exist
    PYTHON_VERSION=$("${PYTHON_CMD}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.10")
    AOTI_HEADER_PATH="${NEQUIP_ENV}/lib/python${PYTHON_VERSION}/site-packages/torch/include/torch/csrc/inductor/aoti_package/model_package_loader.h"
    if [ -f "${AOTI_HEADER_PATH}" ]; then
        ENABLE_AOTI=true
        echo -e "${GREEN}  ✓ PyTorch version supports AOTI and headers found${NC}"
        echo -e "${GREEN}    AOTI header: ${AOTI_HEADER_PATH}${NC}"
    else
        echo -e "${YELLOW}  Warning: PyTorch >= 2.6 but AOTI headers not found${NC}"
        echo -e "${YELLOW}    Expected: ${AOTI_HEADER_PATH}${NC}"
        echo -e "${YELLOW}    Will use TorchScript models (.nequip.pth) instead${NC}"
    fi
else
    echo -e "${YELLOW}  Warning: PyTorch version ${PYTORCH_VERSION} does not support AOTI (requires >= 2.6)${NC}"
    echo -e "${YELLOW}    Will use TorchScript models (.nequip.pth) instead${NC}"
fi

# Get libtorch path
LIBTORCH_PATH=$("${PYTHON_CMD}" -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null)
if [ -z "${LIBTORCH_PATH}" ]; then
    echo -e "${RED}  Error: Could not get libtorch path from PyTorch${NC}"
    exit 1
fi
CMAKE_PREFIX_PATH="${LIBTORCH_PATH}"
echo -e "${GREEN}  libtorch path: ${CMAKE_PREFIX_PATH}${NC}"

# Check CXX11 ABI compatibility
CXX11_ABI=$("${PYTHON_CMD}" -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)" 2>/dev/null)
if [ "${CXX11_ABI}" = "True" ]; then
    echo -e "${GREEN}  ✓ PyTorch uses CXX11 ABI - compatible with Kokkos${NC}"
else
    echo -e "${YELLOW}  Warning: PyTorch CXX11 ABI is False${NC}"
    echo -e "${YELLOW}    For Kokkos builds, you may need cxx11 abi libtorch${NC}"
fi

# ==============================================================================
# Step 7: Apply pair_nequip_allegro patch
# ==============================================================================

echo -e "${GREEN}[Step 7/10] Checking pair_nequip_allegro patch...${NC}"
if [ ! -d "${LAMMPS_DIR}/src/EXTRA-PAIR" ] || [ ! -f "${LAMMPS_DIR}/src/EXTRA-PAIR/pair_allegro.h" ]; then
    echo -e "${YELLOW}  Patch not applied, attempting to apply...${NC}"
    if [ ! -d "${PAIR_NEQUIP_DIR}" ]; then
        echo -e "${RED}  Error: pair_nequip_allegro directory not found at ${PAIR_NEQUIP_DIR}${NC}"
        echo -e "${YELLOW}  Please clone the repository:${NC}"
        echo -e "${YELLOW}    git clone https://github.com/mir-group/pair_nequip_allegro.git ${PAIR_NEQUIP_DIR}${NC}"
        exit 1
    fi
    if [ ! -f "${PAIR_NEQUIP_DIR}/patch_lammps.sh" ]; then
        echo -e "${RED}  Error: patch_lammps.sh not found at ${PAIR_NEQUIP_DIR}/patch_lammps.sh${NC}"
        exit 1
    fi
    
    # Save current directory
    ORIGINAL_DIR=$(pwd)
    # Change to pair_nequip_allegro directory (required by patch script)
    cd "${PAIR_NEQUIP_DIR}" || {
        echo -e "${RED}  Error: Cannot change to ${PAIR_NEQUIP_DIR}${NC}"
        exit 1
    }
    echo -e "${GREEN}  Running patch_lammps.sh from $(pwd)${NC}"
    bash ./patch_lammps.sh "${LAMMPS_DIR}" || {
        echo -e "${RED}  Error: patch_lammps.sh failed${NC}"
        cd "${ORIGINAL_DIR}"
        exit 1
    }
    cd "${ORIGINAL_DIR}" || exit 1
    echo -e "${GREEN}  ✓ Patch applied successfully${NC}"
else
    echo -e "${GREEN}  ✓ Patch already applied${NC}"
fi

# ==============================================================================
# Step 8: Detect GPU architecture
# ==============================================================================

echo -e "${GREEN}[Step 8/10] Detecting GPU architecture...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
    echo -e "${GREEN}  Detected GPU compute capability: ${GPU_COMPUTE_CAP}${NC}"
    
    # Map compute capability to Kokkos architecture
    case "${GPU_COMPUTE_CAP}" in
        8.0|8.6)
            KOKKOS_GPU_ARCH="AMPERE80"
            GPU_ARCH="sm_80"
            ;;
        7.5)
            KOKKOS_GPU_ARCH="TURING75"
            GPU_ARCH="sm_75"
            ;;
        7.0|7.2)
            KOKKOS_GPU_ARCH="VOLTA70"
            GPU_ARCH="sm_70"
            ;;
        6.0|6.1)
            KOKKOS_GPU_ARCH="PASCAL60"
            GPU_ARCH="sm_60"
            ;;
        *)
            echo -e "${YELLOW}  Warning: Unknown GPU compute capability ${GPU_COMPUTE_CAP}, defaulting to AMPERE80${NC}"
            KOKKOS_GPU_ARCH="AMPERE80"
            GPU_ARCH="sm_80"
            ;;
    esac
else
    echo -e "${YELLOW}  Warning: nvidia-smi not found, defaulting to AMPERE80 (sm_80)${NC}"
    KOKKOS_GPU_ARCH="AMPERE80"
    GPU_ARCH="sm_80"
fi

echo -e "${GREEN}  Using GPU architecture: ${GPU_ARCH} (Kokkos: ${KOKKOS_GPU_ARCH})${NC}"

# Extract numeric architecture for CMAKE_CUDA_ARCHITECTURES
CUDA_ARCH_NUM=$(echo "${GPU_ARCH}" | sed 's/sm_//')
echo -e "${GREEN}  CMAKE_CUDA_ARCHITECTURES: ${CUDA_ARCH_NUM}${NC}"

# ==============================================================================
# Step 9: Setup compilers and build directory
# ==============================================================================

echo -e "${GREEN}[Step 9/10] Setting up compilers and build directory...${NC}"

# Find compilers
MPICXX=$(which mpicxx || which mpic++ || echo "mpicxx")
MPICC=$(which mpicc || echo "mpicc")
MPIFORT=$(which mpifort || which mpif90 || echo "mpifort")

echo -e "${GREEN}  Found compilers:${NC}"
echo "    CXX: ${MPICXX}"
echo "    CC:  ${MPICC}"
echo "    FC:  ${MPIFORT}"

# For KOKKOS with CUDA, we need to use nvcc_wrapper
NVCC_WRAPPER="${LAMMPS_DIR}/lib/kokkos/bin/nvcc_wrapper"
if [ -f "${NVCC_WRAPPER}" ]; then
    chmod +x "${NVCC_WRAPPER}" 2>/dev/null || true
    echo -e "${GREEN}  Using KOKKOS nvcc_wrapper: ${NVCC_WRAPPER}${NC}"
    export OMPI_CXX="${MPICXX}" 2>/dev/null || true
    KOKKOS_CXX="${NVCC_WRAPPER}"
else
    echo -e "${YELLOW}  Warning: nvcc_wrapper not found at ${NVCC_WRAPPER}${NC}"
    echo -e "${YELLOW}    KOKKOS CUDA builds require nvcc_wrapper. Using mpicxx instead.${NC}"
    KOKKOS_CXX="${MPICXX}"
fi

# Clean build directory if requested
if [ "${CLEAN_BUILD:-}" = "yes" ]; then
    echo -e "${YELLOW}  Cleaning build directory...${NC}"
    rm -rf "${BUILD_DIR}"
fi

# Create build directory
mkdir -p "${BUILD_DIR}"
mkdir -p "${INSTALL_DIR}"
cd "${BUILD_DIR}"

echo -e "${GREEN}  ✓ Build directory ready: ${BUILD_DIR}${NC}"

# ==============================================================================
# Step 10: Configure and build LAMMPS
# ==============================================================================

echo -e "${GREEN}[Step 10/10] Configuring and building LAMMPS...${NC}"
echo ""

# Build CMake command
# Note: KOKKOS is REQUIRED for pair_allegro (as of 10 September 2025 LAMMPS release)
CMAKE_CMD=(
    cmake "${LAMMPS_DIR}/cmake"
    -D CMAKE_BUILD_TYPE=RelWithDebInfo
    -D BUILD_MPI=yes
    -D BUILD_OMP=yes
    -D PKG_EXTRA-PAIR=yes
    -D PKG_KOKKOS=ON
    -D CMAKE_INSTALL_PREFIX="${INSTALL_DIR}"
    -D CMAKE_CXX_COMPILER="${KOKKOS_CXX}"
    -D CMAKE_C_COMPILER="${MPICC}"
    -D CMAKE_Fortran_COMPILER="${MPIFORT}"
    -D CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}"
    -D PKG_GPU=yes
    -D GPU_API=cuda
    -D GPU_PREC=double
    -D GPU_ARCH="${GPU_ARCH}"
    -D CMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH_NUM}"
    -D CUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}"
    -D Kokkos_ENABLE_SERIAL=ON
    -D Kokkos_ENABLE_CUDA=ON
    -D Kokkos_ENABLE_OPENMP=ON
    -D Kokkos_ENABLE_CUDA_LAMBDA=ON
    -D "Kokkos_ARCH_${KOKKOS_GPU_ARCH}=ON"
    -D CMAKE_CXX_FLAGS="-O3 -g"
    -D CMAKE_C_FLAGS="-O3 -g"
)

# Add AOTI flag if supported
if [ "$ENABLE_AOTI" = true ]; then
    CMAKE_CMD+=(-D NEQUIP_AOT_COMPILE=ON)
    echo -e "${GREEN}  Enabling AOTI compilation (recommended for performance)${NC}"
else
    CMAKE_CMD+=(-D NEQUIP_AOT_COMPILE=OFF)
    echo -e "${YELLOW}  AOTI compilation disabled - will use TorchScript models (.nequip.pth)${NC}"
fi

# MKL workaround (PyTorch's CMake looks for MKL automatically)
CMAKE_CMD+=(-D MKL_INCLUDE_DIR=/tmp)

echo -e "${GREEN}Running CMake configuration...${NC}"
echo ""
"${CMAKE_CMD[@]}" 2>&1 | tee cmake_config.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}  CMake configuration failed!${NC}"
    echo -e "${YELLOW}  Check cmake_config.log for details${NC}"
    exit 1
fi

# Verify configuration
if [ "$ENABLE_AOTI" = true ]; then
    if grep -q "NEQUIP_AOT_COMPILE is enabled" cmake_config.log; then
        echo -e "${GREEN}  ✓ AOTI compilation enabled${NC}"
    else
        echo -e "${YELLOW}  Warning: AOTI compilation flag not confirmed in CMake output${NC}"
    fi
fi

if grep -qi "KOKKOS" cmake_config.log; then
    echo -e "${GREEN}  ✓ KOKKOS package enabled${NC}"
else
    echo -e "${RED}  Error: KOKKOS package not confirmed in CMake output${NC}"
    exit 1
fi

# Build
echo ""
echo -e "${GREEN}Building LAMMPS (this may take 30-60 minutes)...${NC}"
NPROC=$(nproc 2>/dev/null || echo 8)
make -j${NPROC} 2>&1 | tee build.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}  Build failed!${NC}"
    echo -e "${YELLOW}  Check build.log for details${NC}"
    exit 1
fi

# Install
echo ""
echo -e "${GREEN}Installing LAMMPS...${NC}"
make install 2>&1 | tee install.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}  Installation failed!${NC}"
    echo -e "${YELLOW}  Check install.log for details${NC}"
    exit 1
fi

# ==============================================================================
# Verification
# ==============================================================================

echo ""
echo -e "${BLUE}==============================================================================${NC}"
echo -e "${GREEN}Build Complete!${NC}"
echo -e "${BLUE}==============================================================================${NC}"
echo ""

if [ -f "${INSTALL_DIR}/bin/lmp" ]; then
    echo -e "${GREEN}LAMMPS binary: ${INSTALL_DIR}/bin/lmp${NC}"
    echo ""
    echo -e "${GREEN}Verifying installation...${NC}"
    "${INSTALL_DIR}/bin/lmp" -help | head -20
    echo ""
    
    # Check if pair_allegro is available
    if "${INSTALL_DIR}/bin/lmp" -help 2>&1 | grep -q "pair_style allegro"; then
        echo -e "${GREEN}✓ pair_style allegro is available!${NC}"
    else
        echo -e "${YELLOW}Warning: pair_style allegro not found in help output${NC}"
    fi
    
    # Check KOKKOS package info
    if "${INSTALL_DIR}/bin/lmp" -help 2>&1 | grep -qi "kokkos"; then
        echo -e "${GREEN}✓ KOKKOS package is available!${NC}"
    fi
    
    echo ""
    echo -e "${GREEN}To use LAMMPS:${NC}"
    echo -e "${YELLOW}  export LAMMPS_BIN=\"${INSTALL_DIR}/bin/lmp\"${NC}"
    echo ""
    
    if [ "$ENABLE_AOTI" = true ]; then
        echo -e "${GREEN}Note: AOTI support is enabled. To use AOT-compiled models:${NC}"
        echo -e "${YELLOW}  1. Compile your model: nequip-compile --aot-deploy model.pth model.nequip.pt2${NC}"
        echo -e "${YELLOW}  2. Use .nequip.pt2 file in LAMMPS pair_coeff command${NC}"
    else
        echo -e "${YELLOW}Note: AOTI support is disabled. Use TorchScript models (.nequip.pth)${NC}"
    fi
else
    echo -e "${RED}Error: LAMMPS binary not found!${NC}"
    exit 1
fi
