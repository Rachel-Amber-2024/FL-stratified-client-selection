#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

"""
-------
CIFAR 10
-------
"""

DATASET_FOLDER = "dataset/CIFAR10/"


class CIFARDataset(Dataset):
    """Convert the CIFAR CIFAR10 file into a Pytorch Dataset"""

    def __init__(self, file_path: str, k: int):
        dataset = pickle.load(open(file_path, "rb"))

        self.X = dataset[0][k]
        self.y = np.array(dataset[1][k])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        # 3D input 32x32x3
        x = torch.Tensor(self.X[idx]).permute(2, 0, 1) / 255
        x = (x - 0.5) / 0.5
        y = self.y[idx]

        return x, y


class CIFARShardDataset(Dataset):
    """Convert the MNIST pkl file into a Pytorch Dataset"""

    def __init__(self, file_path, k):

        with open(file_path, "rb") as pickle_file:
            dataset = pickle.load(pickle_file)
            self.features = np.vstack(dataset[0][k])

            vector_labels = list()
            for idx, digit in enumerate(dataset[1][k]):
                vector_labels += [digit] * len(dataset[0][k][idx])

            self.labels = np.array(vector_labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):

        # 3D input 32x32x3
        x = torch.Tensor(self.features[idx]).permute(2, 0, 1) / 255
        x = (x - 0.5) / 0.5
        # y = self.labels[idx]
        y = torch.LongTensor([self.labels[idx]])[0]

        return x, y

def get_1shard(ds, row_0: int, digit: int, samples: int):
    """return an array from `ds` of `digit` starting of `row_0` in the indices of `ds`"""

    row = row_0

    shard = list()

    while len(shard) < samples:
        if ds.targets[row] == digit:
            shard.append(ds.data[row])
        row += 1

    return row, shard


def clients_set_CIFAR10_shard(file_name, n_clients, batch_size=100, shuffle=True):
    """Download for all the clients their respective dataset"""

    print(file_name)

    list_dl = list()
    for k in range(n_clients):
        dataset_object = CIFARShardDataset(file_name, k)
        dataset_dl = DataLoader(
            dataset_object, batch_size=batch_size, shuffle=shuffle
        )
        list_dl.append(dataset_dl)

    return list_dl


def create_CIFAR10_ds_1shard_per_client(n_clients, samples_train, samples_test):

    CIFAR10_train = datasets.CIFAR10(
        root=DATASET_FOLDER,
        train=True,
        download=True,
        # transform=transforms.ToTensor(),
    )

    CIFAR10_test = datasets.CIFAR10(
        root=DATASET_FOLDER,
        train=False,
        download=True,
        # transform=transforms.ToTensor(),
    )
    shards_train, shards_test = [], []
    labels = []

    for i in range(10):
        row_train, row_test = 0, 0
        for j in range(10):
            row_train, shard_train = get_1shard(
                CIFAR10_train, row_train, i, samples_train
            )
            row_test, shard_test = get_1shard(
                CIFAR10_test, row_test, i, samples_test
            )

            shards_train.append([shard_train])
            shards_test.append([shard_test])

            labels += [[i]]

    X_train = np.array(shards_train)
    X_test = np.array(shards_test)

    y_train = labels
    y_test = y_train

    folder = DATASET_FOLDER
    train_path = f"CIFAR10_shard_train_{n_clients}_{samples_train}.pkl"
    with open(folder + train_path, "wb") as output:
        pickle.dump((X_train, y_train), output)

    test_path = f"CIFAR10_shard_test_{n_clients}_{samples_test}.pkl"
    with open(folder + test_path, "wb") as output:
        pickle.dump((X_test, y_test), output)


def partition_CIFAR10_dataset(
        dataset,
        file_name: str,
        matrix,
        n_clients: int,
        n_classes: int,
        train: bool,
):
    """Partition dataset into `n_clients`.
    Each client i has matrix[k, i] of data of class k"""

    list_clients_X = [[] for i in range(n_clients)]
    list_clients_y = [[] for i in range(n_clients)]

    client_sample_nums = [500] * n_clients

    list_idx = []
    # 按label分类为10*5000，存入list_idx
    for k in range(n_classes):
        idx_k = np.where(np.array(dataset.targets) == k)[0]
        list_idx += [idx_k]

    for idx_client, n_sample in enumerate(client_sample_nums):

        clients_idx_i = []
        client_samples = 0

        for k in range(n_classes):

            if k < 9:
                samples_digit = int(matrix[idx_client, k] * n_sample)
            if k == 9:
                samples_digit = n_sample - client_samples
            client_samples += samples_digit

            clients_idx_i = np.concatenate(
                (clients_idx_i, np.random.choice(list_idx[k], samples_digit))
            )

        clients_idx_i = clients_idx_i.astype(int)

        for idx_sample in clients_idx_i:
            list_clients_X[idx_client] += [dataset.data[idx_sample]]
            list_clients_y[idx_client] += [dataset.targets[idx_sample]]

        list_clients_X[idx_client] = np.array(list_clients_X[idx_client])

    folder = DATASET_FOLDER
    with open(folder + file_name, "wb") as output:
        pickle.dump((list_clients_X, list_clients_y), output)


# 1
def create_CIFAR10_dirichlet(
        dataset_name: str,
        patition: str,
        alpha: float,
        n_clients: int,
        n_classes: int,
):
    """Create a CIFAR dataset partitioned according to a dirichilet distribution Dir(alpha)"""

    from numpy.random import dirichlet

    # shape ``(size, k)``
    matrix = dirichlet([alpha] * n_classes, size=n_clients)
    # if matrix.isnull().any():
    #     matrix.replace(np.nan, 0)  # delete nan

    CIFAR10_train = datasets.CIFAR10(
        root=DATASET_FOLDER,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    CIFAR10_test = datasets.CIFAR10(
        root=DATASET_FOLDER,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    file_name_train = f"{dataset_name}_{patition}_train_{n_clients}.pkl"
    partition_CIFAR10_dataset(
        CIFAR10_train,
        file_name_train,
        matrix,
        n_clients,
        n_classes,
        True,
    )

    file_name_test = f"{dataset_name}_{patition}_test_{n_clients}.pkl"
    partition_CIFAR10_dataset(
        CIFAR10_test,
        file_name_test,
        matrix,
        n_clients,
        n_classes,
        False,
    )


# 3
def clients_set_CIFAR(
        file_name: str, n_clients: int, batch_size: int, shuffle=True
):
    """Download for all the clients their respective dataset"""
    print(file_name)

    list_dl = list()

    for k in range(n_clients):
        dataset_object = CIFARDataset(file_name, k)

        dataset_dl = DataLoader(
            dataset_object, batch_size=batch_size, shuffle=shuffle
        )

        list_dl.append(dataset_dl)

    return list_dl


# 0
def get_CIFAR10_dataloaders(dataset_name, patition, batch_size: int, shuffle=True, reset_data=False):
    """
    """
    folder = DATASET_FOLDER

    if patition == "shard":
        n_clients = 100
        samples_train, samples_test = 500, 80

        file_name_train = f"CIFAR10_shard_train_{n_clients}_{samples_train}.pkl"
        path_train = DATASET_FOLDER + file_name_train

        file_name_test = f"CIFAR10_shard_test_{n_clients}_{samples_test}.pkl"
        path_test = DATASET_FOLDER + file_name_test

        if not os.path.isfile(path_train):
            create_CIFAR10_ds_1shard_per_client(
                n_clients, samples_train, samples_test
            )

        list_dls_train = clients_set_CIFAR10_shard(
            path_train, n_clients, batch_size=batch_size, shuffle=shuffle
        )

        list_dls_test = clients_set_CIFAR10_shard(
            path_test, n_clients, batch_size=batch_size, shuffle=shuffle
        )

    # dirichlet
    elif patition[0:4] == "dir_":
        n_classes = 10
        n_clients = 100
        alpha = patition[4:]

        file_name_train = f"{dataset_name}_{patition}_train_{n_clients}.pkl"
        path_train = folder + file_name_train

        file_name_test = f"{dataset_name}_{patition}_test_{n_clients}.pkl"
        path_test = folder + file_name_test

        # fix the dataset
        if not os.path.isfile(path_train) or reset_data:
            print("⚠⚠⚠ creating new dataset alpha:", alpha, " ⚠⚠⚠")
            create_CIFAR10_dirichlet(
                dataset_name, patition, alpha, n_clients, n_classes
            )

        list_dls_train = clients_set_CIFAR(
            path_train, n_clients, batch_size, True
        )

        list_dls_test = clients_set_CIFAR(
            path_test, n_clients, batch_size, True
        )

    return list_dls_train, list_dls_test
