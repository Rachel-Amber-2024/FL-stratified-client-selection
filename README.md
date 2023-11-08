# FedSTS: A Stratified Client Selection Framework for Consistently Fast Federated Learning

A PyTorch implementation of our paper FedSTS: A Stratified Client Selection Framework for Consistently Fast Federated Learning.

## Dependencies
+ Python (>=3.6)
+ PyTorch (>=1.7.1)
+ NumPy (>=1.19.2)
+ Scikit-Learn (>=0.24.1)
+ Scipy (>=1.6.1)

To install all dependencies:
```
pip install -r requirements.txt
```

## Running an experiment

Here we provide the implementation of Stratified Client Selection Scheme along with MNIST, FMNIST and CIFAR-10 dataset. This code takes as input:

- The `dataset` used.
- The data `partition` method used. partition ∈ { iid, dir_{alpha}, shard }
- The `sampling` scheme used. sampling ∈ { random, importance, ours }
- The percentage of clients sampled `sample_ratio`. We consider 100 clients in all our datasets and use thus sample_ratio=0.1.
- The learning rate `lr` used.
- The batch size `batch_size` used.
- The number of SGD run locally `n_SGD` used.
- The number of rounds of training `n_iter`.
- The number of strata `strata_num` used in ours sampling.
- The learning rate `decay` used after each SGD. We consider no decay in our experiments, decay=1.
- The local loss function regularization parameter `mu`. FedProx with µ = 0 and without systems heterogeneity (no stragglers) corresponds to FedAvg.
- The `seed` used to initialize the training model. We use 0 in all our experiments.
- Force a boolean equal `force` to True when a simulation has already been run but needs to be rerun.


+ To train and evaluate on MNIST:
```
python main_mnist.py --dataset=MNIST \
    --partition=iid \
    --sampling=random \
    --sample_ratio=0.1 \
    --lr=0.01 \
    --batch_size=50 \
    --n_SGD=50 \
    --n_iter=200 \
    --strata_num=10 \
    --decay=1.0 \
    --mu=0.0 \
    --seed=0 \
    --force=False
```

+ To train and evaluate on FMNIST:
```
python main_fmnist.py --dataset=FMNIST \
    --partition=shard \
    --sampling=importance \
    --sample_ratio=0.1 \
    --lr=0.01 \
    --batch_size=50 \
    --n_SGD=50 \
    --n_iter=200 \
    --strata_num=10 \
    --decay=1.0 \
    --mu=0.0 \
    --seed=0 \
    --force=False
```

+ To train and evaluate on CIFAR-10:
```
python main_cifar10.py --dataset=CIFAR10 \
    --partition=dir_0.001 \
    --sampling=ours \
    --sample_ratio=0.1 \
    --lr=0.05 \
    --batch_size=50 \
    --n_SGD=80 \
    --n_iter=800 \
    --strata_num=10 \
    --decay=1.0 \
    --mu=0.0 \
    --seed=0 \
    --force=False
```


Every experiment saves by default the training loss, the testing accuracy, and the sampled clients at every iteration in the folder `saved_exp_info`. The global model and local models histories can also be saved.

## Citation
If you use our code in your research, please cite the following article:
```

```
