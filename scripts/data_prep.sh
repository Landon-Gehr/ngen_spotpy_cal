#!/bin/bash
#SBATCH --job-name="ngen-data-prep"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output="script_outputs/ngen_data_prep-%j.out"                  
#SBATCH --time=00:30:00


module purge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngen

gage_id="01078000"
cat_id="13105"
start_dt="2008-10-01T00:00:00" 
end_dt="2012-09-30T00:00:00"
save_dir="/home/lsgehr/scratch/NextGen/ngen_spotpy_cal/data/"
image="/home/lsgehr/scratch/NextGen/ngen_spotpy_cal/singularity/templates/ngen.sif"
realization_template="/home/lsgehr/scratch/NextGen/ngen_spotpy_cal/template_data/realization_template.json"
troute_template="/home/lsgehr/scratch/NextGen/ngen_spotpy_cal/template_data/troute_template.yaml"

python data_prep.py "$gage_id" "$start_dt" "$end_dt" \
    --save-dir="$save_dir" \
    --cat-id="$cat_id" \
    --image="$image" \
    --template-realization="$realization_template" \
    --template-troute="$troute_template"