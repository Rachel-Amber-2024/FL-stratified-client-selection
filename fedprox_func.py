#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import config
from utils import *
from copy import deepcopy
from torch.autograd import Variable

if config.USE_GPU:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# turn data into Variable, do .cuda() when USE_GPU is True
def get_variable(x):
    x = Variable(x)
    return x.cuda() if config.USE_GPU else x


def loss_classifier(predictions, labels):

    criterion = nn.CrossEntropyLoss()
    return criterion(predictions, labels)


def accuracy_dataset(model, dataset):
    """Compute the accuracy of `model` on `test_data`"""

    correct = 0

    for features, labels in dataset:

        features = get_variable(features)
        labels = get_variable(labels)

        predictions = model(features)
        _, predicted = predictions.max(1, keepdim=True)

        correct += torch.sum(predicted.view(-1, 1) == labels.view(-1, 1)).item()

    accuracy = 100 * correct / len(dataset.dataset)

    return accuracy


def loss_dataset(model, train_data, loss_classifier):
    """Compute the loss of `model` on `test_data`"""
    loss = 0
    for idx, (features, labels) in enumerate(train_data):

        features = get_variable(features)
        labels = get_variable(labels)

        predictions = model(features)
        loss += loss_classifier(predictions, labels)

    loss /= idx + 1
    return loss


def local_learning(model, mu: float, optimizer, train_data, n_SGD: int, loss_classifier):
    model_0 = deepcopy(model)

    for _ in range(n_SGD):

        features, labels = next(iter(train_data))

        features = get_variable(features)
        labels = get_variable(labels)

        optimizer.zero_grad()

        predictions = model(features)

        batch_loss = loss_classifier(predictions, labels)
        
        tensor_1 = list(model.parameters())
        tensor_2 = list(model_0.parameters())
        norm = sum(
            [
                torch.sum((tensor_1[i] - tensor_2[i]) ** 2)
                for i in range(len(tensor_1))
            ]
        )
        batch_loss += mu / 2 * norm
        
        batch_loss.backward()
        optimizer.step()


def FedProx_random_sampling(
    model,
    n_sampled,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr,
    file_name: str,
    decay,
    mu,
):
    K = len(training_sets)  # number of clients
    n_samples = np.array([len(db.dataset) for db in training_sets])
    weights = n_samples / np.sum(n_samples)
    print("Clients' weights:", weights)

    loss_hist = np.zeros((n_iter + 1, K))
    acc_hist = np.zeros((n_iter + 1, K))

    for k, dl in enumerate(training_sets):

        loss_hist[0, k] = float(loss_dataset(model, dl, loss_classifier).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)

    # LOSS AND ACCURACY OF THE INITIAL MODEL
    server_loss = np.dot(weights, loss_hist[0])
    server_acc = np.dot(weights, acc_hist[0])
    print(f"====> i: 0 Loss: {server_loss} Test Accuracy: {server_acc}")

    sampled_clients_hist = np.zeros((n_iter, K)).astype(int)

    for i in range(n_iter):

        clients_params = []

        np.random.seed(i)
        sampled_clients = random.sample([x for x in range(K)], n_sampled)

        for k in sampled_clients:

            local_model = deepcopy(model)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

            local_learning(
                local_model,
                mu,
                local_optimizer,
                training_sets[k],
                n_SGD,
                loss_classifier,
            )

            # GET THE PARAMETER TENSORS OF THE MODEL
            list_params = list(local_model.parameters())
            list_params = [tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)

            sampled_clients_hist[i, k] = 1

        # CREATE THE NEW GLOBAL MODEL
        new_model = deepcopy(model)
        weights_ = [weights[client] for client in sampled_clients]

        for layer_weigths in new_model.parameters():
            layer_weigths.data.sub_(sum(weights_) * layer_weigths.data)

        for k, client_hist in enumerate(clients_params):
            for idx, layer_weights in enumerate(new_model.parameters()):
                contribution = client_hist[idx].data * weights_[k]
                layer_weights.data.add_(contribution)

        model = new_model

        # COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
        for k, dl in enumerate(training_sets):
            loss_hist[i + 1, k] = float(
                loss_dataset(model, dl, loss_classifier).detach()
            )

        for k, dl in enumerate(testing_sets):
            acc_hist[i + 1, k] = accuracy_dataset(model, dl)

        server_loss = np.dot(weights, loss_hist[i + 1])
        server_acc = np.dot(weights, acc_hist[i + 1])

        print(
            f"====> i: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}"
        )

        # DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
        lr *= decay

    # SAVE THE DIFFERENT TRAINING HISTORY
    #    save_pkl(models_hist, "local_model_history", file_name)
    #    save_pkl(server_hist, "server_history", file_name)
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist


def FedProx_importance_sampling(
    model,
    n_sampled,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr,
    file_name: str,
    decay,
    mu,
):
    K = len(training_sets)  # number of clients
    n_samples = np.array([len(db.dataset) for db in training_sets])
    weights = n_samples / np.sum(n_samples)
    print("Clients' weights:", weights)

    loss_hist = np.zeros((n_iter + 1, K))
    acc_hist = np.zeros((n_iter + 1, K))

    for k, dl in enumerate(training_sets):

        loss_hist[0, k] = float(loss_dataset(model, dl, loss_classifier).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)

    # LOSS AND ACCURACY OF THE INITIAL MODEL
    server_loss = np.dot(weights, loss_hist[0])
    server_acc = np.dot(weights, acc_hist[0])
    print(f"====> i: 0 Loss: {server_loss} Test Accuracy: {server_acc}")

    sampled_clients_hist = np.zeros((n_iter, K)).astype(int)

    for i in range(n_iter):

        clients_params = []

        np.random.seed(i)
        sampled_clients = np.random.choice(
            K, size=n_sampled, replace=True, p=weights
        )

        for k in sampled_clients:

            local_model = deepcopy(model)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

            local_learning(
                local_model,
                mu,
                local_optimizer,
                training_sets[k],
                n_SGD,
                loss_classifier,
            )

            # GET THE PARAMETER TENSORS OF THE MODEL
            list_params = list(local_model.parameters())
            list_params = [tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)

            sampled_clients_hist[i, k] = 1

        # CREATE THE NEW GLOBAL MODEL
        new_model = deepcopy(model)
        weights_ = [1 / n_sampled] * n_sampled
        for layer_weigths in new_model.parameters():
            layer_weigths.data.sub_(layer_weigths.data)

        for k, client_hist in enumerate(clients_params):
            for idx, layer_weights in enumerate(new_model.parameters()):
                contribution = client_hist[idx].data * weights_[k]
                layer_weights.data.add_(contribution)

        model = new_model

        # COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
        for k, dl in enumerate(training_sets):
            loss_hist[i + 1, k] = float(
                loss_dataset(model, dl, loss_classifier).detach()
            )

        for k, dl in enumerate(testing_sets):
            acc_hist[i + 1, k] = accuracy_dataset(model, dl)

        server_loss = np.dot(weights, loss_hist[i + 1])
        server_acc = np.dot(weights, acc_hist[i + 1])

        print(
            f"====> i: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}"
        )

        # DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
        lr *= decay

    # SAVE THE DIFFERENT TRAINING HISTORY
    #    save_pkl(models_hist, "local_model_history", file_name)
    #    save_pkl(server_hist, "server_history", file_name)
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist


def FedProx_stratified_sampling(
    args,
    model,
    n_sampled: int,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr: float,
    file_name: str,
    decay,
    mu,
):
    # Variables initialization
    K = len(training_sets)  # number of clients
    n_samples = np.array([len(db.dataset) for db in training_sets])
    weights = n_samples / np.sum(n_samples)
    print("Clients' weights:", weights)

    # STRATIFY THE CLIENTS
    stratify_result = stratify_clients(args)

    allocation_number = []
    if config.WITH_ALLOCATION and not args.partition == 'shard':
        partition_result = pickle.load(open(f"dataset/data_partition_result/{args.dataset}_{args.partition}.pkl", "rb"))
        allocation_number = cal_allocation_number(partition_result, stratify_result, args.sample_ratio)
    print(allocation_number)

    N_STRATA = len(stratify_result)
    SIZE_STRATA = [len(cls) for cls in stratify_result]
    N_CLIENTS = sum(len(c) for c in stratify_result)  # number of clients

    loss_hist = np.zeros((n_iter + 1, K))
    acc_hist = np.zeros((n_iter + 1, K))

    for k, dl in enumerate(training_sets):
        loss_hist[0, k] = float(loss_dataset(model, dl, loss_classifier).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)

    # LOSS AND ACCURACY OF THE INITIAL MODEL
    server_loss = np.dot(weights, loss_hist[0])
    server_acc = np.dot(weights, acc_hist[0])
    print(f"====> i: 0 Loss: {server_loss} Test Accuracy: {server_acc}")

    sampled_clients_hist = np.zeros((n_iter, K)).astype(int)


    for i in range(n_iter):

        clients_params = []
        clients_models = []
        sampled_clients_for_grad = []

        # GET THE CLIENTS' CHOSEN PROBABILITY
        chosen_p = np.zeros((N_STRATA, N_CLIENTS)).astype(float)
        for j, cls in enumerate(stratify_result):
            for k in range(N_CLIENTS):
                if k in cls:
                    chosen_p[j][k] = round(1/SIZE_STRATA[j], 12)

        selected = []

        if config.WITH_ALLOCATION and not args.partition == 'shard':
            selects = sample_clients_with_allocation(chosen_p, allocation_number)
        else:
            choice_num = int(100 * args.sample_ratio / args.strata_num)
            selects = sample_clients_without_allocation(chosen_p, choice_num)

        if args.partition == 'iid':
            selects = choice(100, int(100 * args.sample_ratio), replace=False,
                             p=[0.01 for _ in range(100)])

        for _ in selects:
            selected.append(_)
        print("Chosen clients: ", selected)

        for k in selected:
            local_model = deepcopy(model)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

            local_learning(
                local_model,
                mu,
                local_optimizer,
                training_sets[k],
                n_SGD,
                loss_classifier,
            )

            # SAVE THE LOCAL MODEL TRAINED
            list_params = list(local_model.parameters())
            list_params = [
                tens_param.detach() for tens_param in list_params
            ]
            clients_params.append(list_params)
            clients_models.append(deepcopy(local_model))

            sampled_clients_for_grad.append(k)
            sampled_clients_hist[i, k] = 1

        # CREATE THE NEW GLOBAL MODEL AND SAVE IT
        new_model = deepcopy(model)
        weights_ = [1 / n_sampled] * n_sampled
        for layer_weigths in new_model.parameters():
            layer_weigths.data.sub_(layer_weigths.data)

        for k, client_hist in enumerate(clients_params):
            for idx, layer_weights in enumerate(new_model.parameters()):
                contribution = client_hist[idx].data * weights_[k]
                layer_weights.data.add_(contribution)

        model = new_model

        # COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
        for k, dl in enumerate(training_sets):
            loss_hist[i + 1, k] = float(
                loss_dataset(model, dl, loss_classifier).detach()
            )

        for k, dl in enumerate(testing_sets):
            acc_hist[i + 1, k] = accuracy_dataset(model, dl)

        server_loss = np.dot(weights, loss_hist[i + 1])
        server_acc = np.dot(weights, acc_hist[i + 1])

        print(
            f"====> i: {i + 1} Loss: {server_loss} Server Test Accuracy: {server_acc}"
        )

        lr *= decay

    # SAVE THE DIFFERENT TRAINING HISTORY
    # save_pkl(models_hist, "local_model_history", file_name)
    # save_pkl(server_hist, "server_history", file_name)
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist


def run(args, model_mnist, n_sampled, list_dls_train, list_dls_test, file_name):
    """RUN FEDAVG WITH RANDOM SAMPLING"""
    if args.sampling == "random" and (
            not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or args.force
    ):
        FedProx_random_sampling(
            model_mnist,
            n_sampled,
            list_dls_train,
            list_dls_test,
            args.n_iter,
            args.n_SGD,
            args.lr,
            file_name,
            args.decay,
            args.mu,
        )

    """RUN FEDAVG WITH IMPORTANCE SAMPLING"""
    if args.sampling == "importance" and (
            not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or args.force
    ):
        FedProx_importance_sampling(
            model_mnist,
            n_sampled,
            list_dls_train,
            list_dls_test,
            args.n_iter,
            args.n_SGD,
            args.lr,
            file_name,
            args.decay,
            args.mu,
        )

    """RUN FEDAVG WITH OURS SAMPLING"""
    if (args.sampling == "ours") and (
            not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or args.force
    ):
        FedProx_stratified_sampling(
            args,
            model_mnist,
            n_sampled,
            list_dls_train,
            list_dls_test,
            args.n_iter,
            args.n_SGD,
            args.lr,
            file_name,
            args.decay,
            args.mu,
        )
