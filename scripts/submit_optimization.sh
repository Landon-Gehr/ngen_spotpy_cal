#!/bin/bash
#SBATCH --job-name="scuea-optimization"
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --output="script_outputs/scuea_optimization-%j.out"                  
#SBATCH --time=24:00:00 
#SBATCH --mail-user="lsgehr@mines.edu"
#SBATCH --mail-type=FAIL,START,END  
#SBATCH --signal=SIGTERM@300

module purge
module load compilers/gcc/12.2.1
module load mpi/openmpi/gcc/4.1.6

. ~/.conda/.conda_init
conda activate ngen_mpi

config_file="/home/lsgehr/scratch/NextGen/spotpy_cal/data/11264500/cat-3313417/config/ngen_cal_conf.yaml"
singularity_image_path="/home/lsgehr/scratch/NextGen/NGIAB-HPCInfra/singularity_ngen/ngen.sif"
scratch_bind="/scratch/lsgehr:/scratch/lsgehr"

export PYTHONUNBUFFERED=1
RESTARTS=${RESTARTS:-0}
MAX_RESTARTS=${MAX_RESTARTS:-5}

srun --nodes=1 \
     --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
     --cpu-bind=NONE \
     --exclusive \
     python spotpy_calibration.py $config_file $singularity_image_path --ngs=20 \
        --bind=$scratch_bind \
        --verbosity=info

EXIT_CODE=$?

if [ $EXIT_CODE -eq 42 ]; then
    if [ $RESTARTS -lt $MAX_RESTARTS ]; then
        NEW_RESTARTS=$((RESTARTS + 1))
        echo "Restarting ($NEW_RESTARTS/$MAX_RESTARTS)..."
        sbatch --export=ALL,RESTARTS=$NEW_RESTARTS $0
    else
        echo "Max restarts ($MAX_RESTARTS) reached, not resubmitting."
    fi
else
    echo "Job finished normally (exit code $EXIT_CODE)."
fi