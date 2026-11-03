#!/bin/bash -l
# One-time setup of the conda environment for Arctic-XBeach on the Deltares H7 cluster. Run it again to update the environment.
#   bash -l hpc/h7/setup_env.sh     (from the repo checkout, on a login node; -l so that the module command is defined)
set -euo pipefail

REPO_DIR=${AXB_REPO_DIR:-/u/nederhof/git/arctic-xbeach}
ENV_DIR=${AXB_ENV_DIR:-/u/nederhof/envs/arctic-xbeach}

# module and conda scripts use variables that are not set, so they run with 'set -u' off
set +u
module load miniconda/py312_24.7.1-0
set -u

cd "$REPO_DIR"
command -v git > /dev/null || { echo "git is needed to install xbTools from GitHub"; exit 1; }

if [ -d "$ENV_DIR" ]; then
    echo "Updating $ENV_DIR"
    conda env update -p "$ENV_DIR" -f environment.yml --prune
else
    echo "Creating $ENV_DIR"
    conda env create -p "$ENV_DIR" -f environment.yml
fi

set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_DIR"
set -u

pip install -e ".[calibration]"
python -m pytest -q
