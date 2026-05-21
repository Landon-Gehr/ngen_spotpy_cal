#!/bin/bash
#SBATCH --job-name="single-ngen-run"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --output="script_outputs/run_ngen-%j.out"                  
#SBATCH --time=00:09:00


module purge
module load compilers/gcc/12.2.1
module load mpi/openmpi/gcc/4.1.6

. ~/.conda/.conda_init
conda activate ngen_mpi

run_config_path="/home/lsgehr/scratch/NextGen/spotpy_cal/sites/03488000/run_ngen_conf.yaml"

python run_ngen.py --run-config-path=$run_config_path --clear-run-dir=True --save-sim=True