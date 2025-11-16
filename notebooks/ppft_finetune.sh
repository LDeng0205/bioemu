#!/bin/bash


set -euo pipefail

# Enable ColabFold for WSL2
export TF_FORCE_UNIFIED_MEMORY="1"
export XLA_PYTHON_CLIENT_MEM_FRACTION="4.0"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
export TF_FORCE_GPU_ALLOW_GROWTH="true"

# Change venv directory to the current conda environment
export BIOEMU_COLABFOLD_DIR=$CONDA_PREFIX

PROJECT_ROOT_DIR=$(git rev-parse --show-toplevel)

pdb_list=(
    train_1.pdb
    train_2.pdb
    train_3.pdb
    train_4.pdb
    train_5.pdb
)

pdb_list=(
    "6OBK.pdb"
    "2L2D.pdb"
    "XX:run1_0254_0003.pdb"
    "XX:run10_1081_0004.pdb"
    "r4_412_TrROS_Hall.pdb"
)

p_fold_target_list=(
    0.307
    0.697
    0.941
    0.998
    0.992
)

pdb_dir='/data/megascale/AlphaFold_model_PDBs'

for i in "${!pdb_list[@]}"; do
    pdb="${pdb_list[$i]}"
    p_fold_target="${p_fold_target_list[$i]}"
    python ppft_finetune.py --pdb_path $pdb_dir/"$pdb" --output_dir ./ppft_out/"$pdb" --p_fold_target "$p_fold_target" --rollout_config_path ./rollout.yaml #--use_checkpointing
done
