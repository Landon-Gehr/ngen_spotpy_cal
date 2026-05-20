#!/bin/bash

singularity_gpkg_file=$1
singularity_realization_file=$2
parallel=$3
run_dir=$4
model_binary=$5
verbosity=$6
rank=$7

cd $run_dir

num_partitions=$(python /dmod/utils/partitioning/local_only_partitions.py "$singularity_gpkg_file" "$parallel" "."|tail -n 1)
run_command="$model_binary $singularity_gpkg_file all $singularity_gpkg_file all $singularity_realization_file $(pwd)/partitions_$num_partitions.json"


# echo "run_command here"
if [[ "$verbosity" == "debug" ]]; then
    echo "rank $rank running with workers $SLURM_NTASKS, $parallel, $num_partitions ngen sim: $run_command"
    time $run_command
elif [[ "$verbosity" == "info" ]]; then
    echo "rank $rank running with workers $SLURM_NTASKS, $parallel, $num_partitions ngen sim: $run_command"
    time $run_command > /dev/null 2>&1
else
    $run_command > /dev/null 2>&1
fi