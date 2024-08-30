#!/bin/bash

ENV_PATH=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
    --env_path)
        ENV_PATH="$2"
        shift 2
        ;;
    *)
        echo "Unknown parameter passed: $1"
        exit 1
        ;;
    esac
done

# if [ -z "$(pwd | grep 'finetune')" ]; then
#     echo "Please run this script from the scripts directory"
#     exit 1
# fi

if [ -z "$ENV_PATH" ]; then
    echo "ENV_PATH is not set, please set it with --env_path"
    exit 1
fi

module restore
module load Mamba
module load Apptainer
module load PrgEnv-gnu
module load cpe-cuda/23.03
module load cudatoolkit/23.3_11.8

conda deactivate
conda activate $ENV_PATH

poetry install