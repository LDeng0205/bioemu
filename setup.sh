#!/bin/bash

# This script sets up the environment for BioCGM
set -euo pipefail

# Parse command line arguments
DEV=false
MD=false
COLABFOLD=false
MYPY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            DEV=true
            shift
            ;;
        --md)
            MD=true
            shift
            ;;
        --colabfold)
            COLABFOLD=true
            shift
            ;;
        --mypy)
            MYPY=true
            shift
            ;;
        -h | --help)
            echo "Usage: $0 [--dev] [--md] [--colabfold] [--mypy]"
            echo ""
            echo "Options:"
            echo "  --dev        Install development dependencies"
            echo "  --md         Install molecular dynamics dependencies"
            echo "  --colabfold  Install ColabFold dependencies"
            echo "  --mypy       Install mypy dependencies"
            echo "  -h, --help   Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                # Install basic dependencies only"
            echo "  $0 --dev                          # Install with development dependencies"
            echo "  $0 --colabfold                    # Install with ColabFold dependencies"
            echo "  $0 --dev --mypy                   # Install with development and mypy dependencies"
            echo "  $0 --dev --md --colabfold --mypy  # Install all optional dependencies"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Use --help for usage information" >&2
            exit 1
            ;;
    esac
done

# Enable ColabFold for WSL2
export TF_FORCE_UNIFIED_MEMORY="1"
export XLA_PYTHON_CLIENT_MEM_FRACTION="4.0"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
export TF_FORCE_GPU_ALLOW_GROWTH="true"

# Change venv directory to the current conda environment
export BIOEMU_COLABFOLD_DIR=$CONDA_PREFIX

# Create a conda environment for BioCGM
# DO NOT create biocgm environment here if you are calling this script with bash
# Instead, activate biocgm first and then call this script again
# conda create -n biocgm python=3.12 --yes
# conda activate biocgm

# Check the current conda env is biocgm
# if [[ "$CONDA_DEFAULT_ENV" != "biocgm" ]]; then
#     echo "Current conda env is not biocgm. Please create and activate biocgm first."
#     exit 1
# fi

# Install libstdcxx-ng and libgcc-ng
conda install -c conda-forge "libstdcxx-ng>=13" "libgcc-ng>=13" --yes

# Install uv
pip install uv

# Build extras string based on flags
EXTRAS=()
[[ "$DEV" == "true" ]] && EXTRAS+=("dev")
[[ "$MD" == "true" ]] && EXTRAS+=("md")
[[ "$COLABFOLD" == "true" ]] && EXTRAS+=("colabfold")
[[ "$MYPY" == "true" ]] && EXTRAS+=("mypy")

# Join extras with commas
if [[ ${#EXTRAS[@]} -gt 0 ]]; then
    EXTRAS_STRING=$(
        IFS=,
        echo "${EXTRAS[*]}"
    )
else
    EXTRAS_STRING=""
fi

# Install required packages with correct versions
if [[ -n "$EXTRAS_STRING" ]]; then
    uv pip install -e .["$EXTRAS_STRING"]
else
    uv pip install -e .
fi

# Update required packages
uv pip install dm-haiku --upgrade
uv pip install numpy --upgrade
uv pip install pandas --upgrade

# Patch ColabFold if enabled
if python -c "import importlib.util; exit(0) if importlib.util.find_spec('colabfold') else exit(1)"; then
    SITE_PACKAGES_DIR=$(python -c "import site; print(next(p for p in site.getsitepackages() if 'site-packages' in p))")
    SCRIPT_DIR=$(pwd)/src/bioemu/colabfold_setup
    patch "${SITE_PACKAGES_DIR}/alphafold/model/modules.py" "${SCRIPT_DIR}/modules.patch"
    patch "${SITE_PACKAGES_DIR}/colabfold/batch.py" "${SCRIPT_DIR}/batch.patch"
fi

touch "${CONDA_PREFIX}"/.COLABFOLD_PATCHED # mark ColabFold as patched

if [[ $DEV == "true" ]]; then
    pre-commit install
    pre-commit install --hook-type pre-push
    pre-commit install --hook-type pre-commit
fi

echo "Setup complete!"
