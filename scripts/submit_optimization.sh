#!/bin/bash
#SBATCH --job-name="scuea-optimization"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --output="script_outputs/scuea_optimization-%j.out"                  
#SBATCH --time=00:10:00 
#SBATCH --mail-user="lsgehr@mines.edu"
###SBATCH --mail-type=FAIL,START,END  
#SBATCH --signal=SIGTERM@30

module purge
module load compilers/gcc/12.2.1
module load mpi/openmpi/gcc/4.1.6

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngen

export OMPI_MCA_btl_vader_single_copy_mechanism=none

# config_file="/home/lsgehr/scratch/NextGen/ngen_spotpy_cal/data/gage-11264500/config/ngen_cal_conf_template.yaml"
python_script="/home/lsgehr/scratch/NextGen/ngen_spotpy_cal/scripts/home/lsgehr/scratch/NextGen/ngen_spotpy_cal/scripts/spotpy_calibration.py"
config_file="/home/lsgehr/scratch/NextGen/ngen_spotpy_cal/data/MERCED_R_A_HAPPY_ISLES_BRIDGE_NR_YOSEMITE_CA/config/ngen_cal_conf.yaml"
scratch_bind="/scratch/lsgehr:/scratch/lsgehr"

export PYTHONUNBUFFERED=1
RESTARTS=${RESTARTS:-0}
MAX_RESTARTS=${MAX_RESTARTS:-0}

srun --nodes=1 \
     --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
     --cpu-bind=NONE \
     --exclusive \
     python $python_script $config_file $singularity_image_path --ngs=20 \
        --bind=$scratch_bind \
        --verbosity=debug \
        --sampling-reps=3

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
    echo "Job finished with exit code $EXIT_CODE."
fi