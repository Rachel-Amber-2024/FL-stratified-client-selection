#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys
import os
from utils import *
from fedprox_func import *

"""PARSES THE DEFINED ARGUMENTS FROM THE SYS"""

parser = argparse.ArgumentParser(description='FedProx on CIFAR10')

parser.add_argument('--dataset', type=str, default='CIFAR10', help="The dataset used.")
parser.add_argument('--partition', type=str, default='dir_0.001', help="The data partition method used. partition ∈ { dir_{alpha}, shard }")
parser.add_argument('--sampling', type=str, help="The sampling scheme used. sampling ∈ { random, importance, ours }")
parser.add_argument("--sample_ratio", type=float, default=0.1, help="The percentage of clients sampled sample_ratio. We consider 100 clients in all our datasets and use thus sample_ratio=0.1.")
parser.add_argument("--lr", type=float, default=0.05, help="The learning rate lr used.")
parser.add_argument("--batch_size", type=int, default=50, help="The batch size used.")
parser.add_argument("--n_SGD", type=int, default=80, help="The number of SGD run locally n_SGD used.")
parser.add_argument("--n_iter", type=int, default=800, help="The number of rounds of training.")
parser.add_argument("--strata_num", type=int, default=10, help="The number of strata used in ours sampling.")
parser.add_argument("--decay", type=float, default=1.0, help="The learning rate decay used after each SGD. We consider no decay in our experiments, decay=1.")
parser.add_argument("--mu", type=float, default=0.0, help="The local loss function regularization parameter mu. FedProx with µ = 0 and without systems heterogeneity (no stragglers) corresponds to FedAvg.")
parser.add_argument("--seed", type=int, default=0, help="The seed used to initialize the training model. We use 0 in all our experiments.")
parser.add_argument("--force", type=bool, default=False, help="Force a boolean equal to True when a simulation has already been run but needs to be rerun.")

args = parser.parse_args()

print(args)


"""NAME UNDER WHICH THE EXPERIMENT'S VARIABLES WILL BE SAVED"""
file_name = (
    f"CIFAR10_{args.partition}_{args.sampling}_p{args.sample_ratio}_lr{args.lr}_b{args.batch_size}_n{args.n_SGD}_i{args.n_iter}_s{args.strata_num}_d{args.decay}_m{args.mu}_s{args.seed}"
)
print(file_name)


"""GET THE DATASETS USED FOR THE FL TRAINING"""
from dataset.CIFAR10_partition import get_CIFAR10_dataloaders
list_dls_train, list_dls_test = get_CIFAR10_dataloaders(args.dataset, args.partition, args.batch_size)

get_num_cnt(args, list_dls_train)


"""STRATIFY THE CLIENTS"""
stratify_result = stratify_clients(args)


"""NUMBER OF SAMPLED CLIENTS"""
n_sampled = int(args.sample_ratio * len(list_dls_train))
print("number fo sampled clients", n_sampled)


"""LOAD THE INTIAL GLOBAL MODEL"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(args.seed)

class CNN_CIFAR10_dropout(torch.nn.Module):
    """Model Used by the paper introducing FedAvg"""

    def __init__(self):
        super(CNN_CIFAR10_dropout, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=(3, 3)
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3)
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3)
        )

        self.fc1 = nn.Linear(4 * 4 * 64, 64)
        self.fc2 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.dropout(x)
        x = x.view(-1, 4 * 4 * 64)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        return x

model_cifar10 = CNN_CIFAR10_dropout()
if config.USE_GPU:
    model_cifar10 = model_cifar10.cuda()
print("model_cifar10: ", model_cifar10)


"""START TRAINING"""
run(args, model_cifar10, n_sampled, list_dls_train, list_dls_test, file_name)


print("EXPERIMENT IS FINISHED")
