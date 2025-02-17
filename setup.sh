#!/bin/bash

conda create -n zero python=3.9
# install torch [or you can skip this step and let vllm to install the correct version for you]
conda activate zero
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install tensordict packaging
# install vllm
pip install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip install ray

# verl
pip install -e .

# flash attention 2
pip install flash-attn --no-build-isolation
# quality of life
pip install wandb IPython matplotlib
pip install numpy

module purge
module load anaconda3/2024.6
source $(conda info --base)/etc/profile.d/conda.sh
conda activate zero

mkdir /scratch/gpfs/sl2998/data/tinyzero
python ./examples/data_preprocess/countdown.py --local_dir /scratch/gpfs/sl2998/data/tinyzero