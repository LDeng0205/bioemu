#!/usr/bin/bash

#SBATCH --time=50:00:00
#SBATCH -p btrippe,stat,hns
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH --mem=180GB
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

set -euo pipefail

ml reset

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/groups/btrippe/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/groups/btrippe/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/groups/btrippe/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/groups/btrippe/miniconda3/bin:$PATH"
    fi  
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate /scratch/groups/btrippe/arthur/conda_envs/bioemu

cd /scratch/groups/btrippe/arthur/projects/bioemu/adjoint_method

# Enable ColabFold for WSL2
export TF_FORCE_UNIFIED_MEMORY="1"
export XLA_PYTHON_CLIENT_MEM_FRACTION="4.0"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
export TF_FORCE_GPU_ALLOW_GROWTH="true"

# Change venv directory to the current conda environment
export BIOEMU_COLABFOLD_DIR=$CONDA_PREFIX

PROJECT_ROOT_DIR=$(git rev-parse --show-toplevel)

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

pdb_dir='/home/groups/btrippe/datasets/megascale/AlphaFold_model_PDBs'

PROJECT_ROOT_DIR=$(git rev-parse --show-toplevel)
for seed in 0; do
    for i in "${!pdb_list[@]}"; do
        pdb="${pdb_list[$i]}"
        p_fold_target="${p_fold_target_list[$i]}"
        python ppft_finetune.py --pdb_path $pdb_dir/"$pdb" --output_dir ./adjoint_sweep/"$pdb"_N${N_rollout}_t${mid_t} --p_fold_target "$p_fold_target" --rollout_config_path ./adjoint_N50_t0.001.yaml --seed $seed --n_epochs 200 --batch_size 80 --use_checkpointing --cache_embeds_dir "$PROJECT_ROOT_DIR"/.cache/bioemu/embeds --cache_so3_dir "$PROJECT_ROOT_DIR"/.cache/bioemu/so3 --learning_rate 1e-5
    done
done