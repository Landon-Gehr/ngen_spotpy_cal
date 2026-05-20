#!/bin/bash
#SBATCH --job-name="ngen-data-prep"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output="script_outputs/ngen_data_prep-%j.out"                  
#SBATCH --time=00:10:00


module purge
module load compilers/gcc/12.2.1
module load mpi/openmpi/gcc/4.1.6

. ~/.conda/.conda_init
conda activate ngen_mpi

gage_id="11264500"
start_dt="2008-10-01 00:00:00"
end_dt="2012-09-30 00:00:00"

python data_prep.py "$gage_id" "$start_dt" "$end_dt"