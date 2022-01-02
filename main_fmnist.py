#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys
import os
from utils import *
from fedprox_func import *

"""PARSES THE DEFINED ARGUMENTS FROM THE SYS"""

parser = argparse.ArgumentParser(description='FedProx on FMNIST')

parser.add_argument('--dataset', type=str, default='FMNIST', help="The dataset used.")
parser.add_argument('--partition', type=str, default='iid', help="The data partition method used. partition ∈ { iid, dir_{alpha}, shard }")
parser.add_argument('--sampling', type=str, help="The sampling scheme used. sampling ∈ { random, importance, ours }")
parser.add_argument("--sample_ratio", type=float, default=0.1, help="The percentage of clients sampled sample_ratio. We consider 100 clients in all our datasets and use thus sample_ratio=0.1.")
parser.add_argument("--lr", type=float, default=0.01, help="The learning rate lr used.")
parser.add_argument("--batch_size", type=int, default=50, help="The batch size used.")
parser.add_argument("--n_SGD", type=int, default=50, help="The number of SGD run locally n_SGD used.")
parser.add_argument("--n_iter", type=int, default=200, help="The number of rounds of training.")
parser.add_argument("--strata_num", type=int, default=10, help="The number of strata used in ours sampling.")
parser.add_argument("--decay", type=float, default=1.0, help="The learning rate decay used after each SGD. We consider no decay in our experiments, decay=1.")
parser.add_argument("--mu", type=float, default=0.0, help="The local loss function regularization parameter mu. FedProx with µ = 0 and without systems heterogeneity (no stragglers) corresponds to FedAvg.")
parser.add_argument("--seed", type=int, default=0, help="The seed used to initialize the training model. We use 0 in all our experiments.")
parser.add_argument("--force", type=bool, default=False, help="Force a boolean equal to True when a simulation has already been run but needs to be rerun.")

args = parser.parse_args()

print(args)


"""NAME UNDER WHICH THE EXPERIMENT'S VARIABLES WILL BE SAVED"""
file_name = (
    f"FMNIST_{args.partition}_{args.sampling}_p{args.sample_ratio}_lr{args.lr}_b{args.batch_size}_n{args.n_SGD}_i{args.n_iter}_s{args.strata_num}_d{args.decay}_m{args.mu}_s{args.seed}"
)
print(file_name)


"""GET THE DATASETS USED FOR THE FL TRAINING"""
from dataset.FMNIST_partition import get_FMNIST_dataloaders
list_dls_train, list_dls_test = get_FMNIST_dataloaders(args.dataset, args.partition, args.batch_size)

get_num_cnt(args, list_dls_train)


"""STRATIFY THE CLIENTS"""
stratify_result = stratify_clients(args)


"""NUMBER OF SAMPLED CLIENTS"""
n_sampled = int(args.sample_ratio * len(list_dls_train))


"""LOAD THE INTIAL GLOBAL MODEL"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(args.seed)

class CNN_FMNIST_dropout(nn.Module):
    def __init__(self):
        super(CNN_FMNIST_dropout, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU())  # 16, 28, 28
        self.pool1 = nn.MaxPool2d(2)  # 16, 14, 14
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU())  # 32, 12, 12
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())  # 64, 10, 10
        self.pool2 = nn.MaxPool2d(2)  # 64, 5, 5
        self.fc = nn.Linear(5 * 5 * 64, 10)

    def forward(self, x):
        out = self.layer1(x)
        # print(out.shape)
        out = self.pool1(out)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out = self.pool2(out)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.fc(out)
        return out

model_fmnist = CNN_FMNIST_dropout()
if config.USE_GPU:
    model_fmnist = model_fmnist.cuda()
print("model_fmnist: ", model_fmnist)


"""START TRAINING"""
run(args, model_fmnist, n_sampled, list_dls_train, list_dls_test, file_name)


print("EXPERIMENT IS FINISHED")
