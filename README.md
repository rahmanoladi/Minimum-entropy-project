# Introduction
This Project explores two novel loss functions for classification in deep learning. The two loss functions extend cross entropy by regularizing with minimum entropy terms. The first new loss function is termed **mixed entropy** 

# Requirements
The programs in this repo were written using Python 3.9.18, Pytorch 2.2.1, DGL 2.4.0 and DGLLife 0.3.2. You should install the above packages in your environment before running the programs in this repo.

# Sample Runs
**Example 1: Single run of DeeperGCN model on FreeSolv dataset without Path info, and no noise:** 

python train_deeper_gcn.py  --use_gpu 1 --dataset FreeSolv --repitition 1 --epochs 200 \\ \
--use_path_info 0  --add_noise 0 --noise_factor 0.0  \\ \
--num-layers 1 --hidden_dim 1140  \\ \
--dropout 0.35  --weight_decay 7.2362e-13  --batch-size 6  --lr 0.0283 \\ \
--dir_to_save_model path/to/deeper_gcn_models
