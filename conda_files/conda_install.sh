#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 <conda_path> [-y]"
    exit 1
}

skip_install=false
yes_flag=false
ngen_conda_path=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -y)
            yes_flag=true
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            ;;
        *)
            if [ -n "$ngen_conda_path" ]; then
                echo "Error: multiple conda paths specified."
                usage
            fi
            ngen_conda_path="$1"
            shift
            ;;
    esac
done

if [ -z "$ngen_conda_path" ]; then
    echo "Error: conda path is required."
    usage
fi

if [[ "$skip_install" == "false" ]]; then
    if [ -e "$ngen_conda_path" ]; then
        if [ "$yes_flag" = true ]; then
            echo "Removing existing environment: $ngen_conda_path"
            rm -rf "$ngen_conda_path"
        else
            read -p "The conda install path exists, remove? [Y/n] " ans

            case "$ans" in
                [Yy]|[Yy][Ee][Ss]|"")
                    rm -rf "$ngen_conda_path"
                    ;;
                [Nn]|[Nn][Oo])
                    echo "Aborting."
                    exit 1
                    ;;
                *)
                    echo "Invalid response. Aborting."
                    exit 1
                    ;;
            esac
        fi
    fi


    mkdir -p "$ngen_conda_path"
    conda_envs_dir="$(dirname "$ngen_conda_path")"
    conda config --add envs_dirs "$conda_envs_dir"

    echo "Create conda installation"

    conda create -y \
        -p "$ngen_conda_path" \
        -c conda-forge \
        python=3.11 \
        pip \
        setuptools \
        wheel \
        cython \
            > /dev/null

    echo "activate conda env"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ngen

    echo "build mpi from wendian module mpi/openmpi/gcc/4.1.6"
    module load compilers/gcc/12.2.1
    module load mpi/openmpi/gcc/4.1.6

    echo "MPI compiler:"
    which mpicc
    mpicc --version
    mpicc --showme

    MPICC=$(which mpicc) 
    python -m pip install --no-cache-dir --no-binary=mpi4py --no-build-isolation mpi4py 
    MPI4PY_SO="$(python -c 'import mpi4py.MPI; print(mpi4py.MPI.__file__)')"
    echo "mpi4py extension:"
    echo "$MPI4PY_SO"
    echo "MPI libraries linked by mpi4py:"
    ldd "$MPI4PY_SO" | grep -i mpi  
    conda list | grep -Ei 'mpi|mpich|openmpi'

    echo "install spotpy"
    python -m pip install spotpy > /dev/null

    echo "Install Other Python packages"
    conda install -y \
        -c conda-forge \
        numpy \
        scipy \
        pandas \
        xarray \
        netcdf4 \
        matplotlib \
        pyyaml \
        requests \
        uv \
            > /dev/null
fi

echo "activate conda env"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngen

echo "test conda install"
"$ngen_conda_path"/bin/python - <<'PY'
import numpy
import scipy
import pandas
import xarray
import netCDF4
import matplotlib
import yaml
import requests
import spotpy
from mpi4py import MPI

print("imports ok")
PY

MPI4PY_SO="$(python -c 'import mpi4py.MPI; print(mpi4py.MPI.__file__)')"
echo "mpi4py extension:"
echo "$MPI4PY_SO"
echo "MPI libraries linked by mpi4py:"
ldd "$MPI4PY_SO" | grep -i mpi