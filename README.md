# Introduction
This repository introduces two novel loss functions for classification in deep learning. The two loss functions extend standard cross entropy loss by regularizing it with minimum entropy and Kullback-Leibler (K-L) divergence terms. The first of the two novel loss functions is termed **mixed entropy loss** (MIX-ENT for short),  while the second is called **minimum entropy regularized cross-entropy loss** (MIN-ENT for short). The MIX-ENT function introduces a term that can be shown to be equivalent to the sum of a minimum entropy regularizer and a second K-L divergence term. It should be noted that the second K-L divergence term is different from that in the standard cross-entropy loss function, in the sense that it swaps the roles of the target probability, q(x), and the hypothesis probability, p(x). The MIN-ENT simply adds a minimum entropy regularizer to the standard cross entropy loss function. In both MIX-ENT and MIN-ENT, the minimum entropy regularizer minimizes the entropy of the hypothesis probability distribution which is output by the neural network. My experiments on the EMNIST-Letters datasets shows that my implementation of MIX-ENT and MIN-ENT lets the VGG model climb from its previous 3rd position on the paperswithcode Leaderboard to reach the 2nd position on the Leaderboard, outperforming the Spinal-VGG model in so doing. Specifically, using standard cross-entropy, VGG achieves 95.86% while Spinal-VGG achieves 95.88% classification accuracies, whereas using VGG (without Spinal-VGG) our MIN-ENT achieved 95.933%, while our MIX-ENT achieved 95.927% accuracies. The pre-trained models for both MIX-ENT and MIN-ENT are in the models_directory of this repo. Feel free to reproduce the results by running them via the instructions below.   

# Requirements
The programs in this repo were written using Python 3.9.21, Pytorch 2.3.0, and torchvision 0.18.0. You should install the above packages in your environment before running the programs in this repo.

# Sample Runs
**Example 1: Running our pre-trained Min-Ent VGG model** 

python run_pretrained.py  --model vgg &nbsp;&nbsp; --criterion &nbsp; min_ent &nbsp;&nbsp; --input_check_name   &nbsp; 95.933_min_ent_vgg_emnist_letters.pt \\ \
--models_directory &nbsp; path/to/your/models_directory \\ \
--base_1 &nbsp; 2.02 &nbsp;&nbsp; --base_2 &nbsp; 10.73  &nbsp;&nbsp; --loss_1_weight &nbsp; 0.839  &nbsp;&nbsp; --loss_2_weight &nbsp; 0.308

**Example 2: Running our pre-trained Mix-Ent VGG model** 

python run_pretrained.py  --model vgg &nbsp;&nbsp;  --criterion &nbsp; mix_ent_2 &nbsp;&nbsp;  --input_check_name   95.927_mix_ent_vgg_emnist_letters.pt \\ \
--models_directory &nbsp; path/to/your/models_directory \\ \
--base_1 &nbsp; 12.58 &nbsp;&nbsp;  --base_2 &nbsp; 12.45  &nbsp;&nbsp; --loss_1_weight &nbsp; 0.717 &nbsp;&nbsp; --loss_2_weight &nbsp; 0.943 

