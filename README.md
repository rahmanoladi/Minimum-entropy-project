# Introduction
This project explores two novel loss functions for classification in deep learning. The two loss functions extend standard cross entropy loss by regularizing it with minimum entropy and Kullback-Leibler (K-L) divergence terms. The first of the two novel loss functions is formally termed **mixed entropy loss** (MIX-ENT for short),  while the second is formally called **minimum entropy regularized cross-entropy loss** (MIN-ENT for short). The MIX-ENT function introduces a term that can be shown to be equivalent to the sum of a minimum entropy regularizer and a second K-L divergence term. It should be noted that the second K-L divergence term is different from that in the standard cross-entropy loss function, in the same that it swaps the roles of the target probability, q(x), and hypothesis probability p(x). The MIN-ENT simply adds a minimum entropy regularizer to the standard cross entropy loss function. In both MIX-ENT and MIN-ENT, the minimum entropy regularizer minimizes the entropy of the hypothesis probability distribution output by the neural network. My experiments on the EMNIST-Letters datasets shows that my implementation of MIX-ENT and MIN-ENT can let the VGG model climb from its previous 3rd position on the paperswithcode Leaderboard to reach the 2nd position, outperforming the Spinal-VGG model in so doing. Specifically, using standard cross-entropy VGG achieves 95.86 and Spinal-VGG achieves 95.88 classification accuracies, whereas using VGG (without Spinal-VGG) our MIN-ENT achieved 95.933 while our MIX-ENT achieved 95.927 accuracies. The pre-trained models for both MIX-ENT and MIN-ENT are in the models_directory of this repo. Feel free to reproduce the results.   

# Requirements
The programs in this repo were written using Python 3.9.21, Pytorch 2.3.0, and torchvision 0.18.0. You should install the above packages in your environment before running the programs in this repo.

# Sample Runs
**Example 1: Running our pre-trained Mix-Ent VGG model ** 

python train_deeper_gcn.py  --use_gpu 1 --dataset FreeSolv --repitition 1 --epochs 200 \\ \
--use_path_info 0  --add_noise 0 --noise_factor 0.0  \\ \
--num-layers 1 --hidden_dim 1140  \\ \
--dropout 0.35  --weight_decay 7.2362e-13  --batch-size 6  --lr 0.0283 \\ \
--dir_to_save_model path/to/deeper_gcn_models

python run_pretrained.py  --model vgg  --criterion min_ent   --input_check_name  95.933_min_ent_vgg_emnist_letters.pt  \\ \
   --base_1  2.02   --base_2  10.73 \\ \
--loss_1_weight 0.839  --loss_2_weight 0.308   --criterion min_ent   --input_check_name  95.933_min_ent_vgg_emnist_letters.pt


