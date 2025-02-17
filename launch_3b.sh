#!/bin/bash
#SBATCH --job-name=zero-3b-4gpu
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=23:55:00
#SBATCH --mail-user=sl2998@princeton.edu
#SBATCH --partition=pli-c
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err

module purge
module load anaconda3/2024.6
source $(conda info --base)/etc/profile.d/conda.sh
conda activate zero

wandb offline

export N_GPUS=4
export BASE_MODEL=/scratch/gpfs/sl2998/cache/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b
export MODEL_SIZE=3b
# export TASK=gsm8k
# export TASK=arth
# export TASK=math
export TASK=countdown
export DATA_DIR=/scratch/gpfs/sl2998/data/tinyzero/$TASK
export ROLLOUT_TP_SIZE=4
# export RANDOM_SEED=42
# export RANDOM_SEED=123
# export RANDOM_SEED=1234
# export RANDOM_SEED=12345
# export RANDOM_SEED=0
# export RANDOM_SEED=24
# export RANDOM_SEED=6648
# export RANDOM_SEED=4923
# export RANDOM_SEED=257821
export RANDOM_SEED=97363
export EXPERIMENT_NAME=$TASK-qwen2.5-$MODEL_SIZE-$RANDOM_SEED
export VLLM_ATTENTION_BACKEND=XFORMERS

# bash ./scripts/train_tiny_zero.sh

python3 -m verl.trainer.main_ppo \
data.train_files=$DATA_DIR/train.parquet \
data.val_files=$DATA_DIR/test.parquet \
data.train_batch_size=256 \
data.val_batch_size=1312 \
data.max_prompt_length=256 \
data.max_response_length=1024 \
data.random_seed=$RANDOM_SEED \
actor_rollout_ref.model.path=$BASE_MODEL \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.ppo_mini_batch_size=128 \
actor_rollout_ref.actor.ppo_micro_batch_size=8 \
actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
critic.optim.lr=1e-5 \
critic.model.path=$BASE_MODEL \
critic.ppo_micro_batch_size=8 \
algorithm.kl_ctrl.kl_coef=0.001 \
trainer.logger=['wandb'] \
+trainer.val_before_train=False \
trainer.default_hdfs_dir=null \
trainer.n_gpus_per_node=$N_GPUS \
trainer.nnodes=1 \
trainer.save_freq=1000 \
trainer.test_freq=100 \
trainer.project_name=TinyZero \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.total_epochs=15 2>&1 | tee ${MODEL_SIZE}-seed-${RANDOM_SEED}.log