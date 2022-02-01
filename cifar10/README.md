# CIFAR-10 Baselines

## Introduction

CIFAR-10 is one of the most popular datasets in the ML/DL society, and ResNet is also one of the most common baselines for the DL papers regardless of sub-fields. 

As a junior researcher, I found out that there is no *de-facto* standard procedure for training ResNet-18 on the CIFAR-10 dataset, which made many researchers and practitioners (including me) confused and time-consuming since it is quite hard to choose the "nice" starting point of each experiment.

While I looked for the baseline for my experiment, I found a nice baseline with detailed explanations on the PyTorch Lightning documentation. Please refer to the [link](https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/cifar10-baseline.html). This baseline shows up to `92%` of accuracy without fine-tuning, and up to `94%` of accuracy with fine-tuning using SWA(Stochastic Weight Averaging). I believe that these scores above are firm and sufficient for most researchers to further investigate their intersets of research. 

For your information, I summarized the details of the baseline created by PyTorch Lightning team below.

```python
# Use torchvision's implementation and modify it to accomodate the input size of (3, 32, 32)
model = torchvision.models.resnet18(pretrained=False, num_classes=10)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()

# Use normalization, random crop, and random horizontal flip for data augmentation
train_transform = T.Compose([
    T.ToTensor(), 
    T.Normalize(mean=[0.4915, 0.4823, 0.4468], std=[0.2470, 0.2435, 0.2616]),
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
])
test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.4915, 0.4823, 0.4468], std=[0.2470, 0.2435, 0.2616]),
])

# important hyperparameters
num_epochs = 30
eval_ratio = 0.1 # used for train/eval/test split
batch_size = 256
optimizer  = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
scheduler  = optim.lr_scheduler.OneCycleLR(
    optimizer=optimizer, 
    max_lr=0.1,
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
)
```

Here, I implemented a grid search on the most common hyperparameter settings using PyTorch and WanDB. It uses WanDB's sweep feature and automatically reports the train/eval/test performance, the model's weights and gradients, and hyperparameters. 

## How to run



## Experiment Setting

### Random Seed

- Didn't set any random seed. Reproducibility not guaranteed!!!

- As a personal researcher with limited resources, it is impossible to run each setting multiple times and average its results.

- I also didn't want to tune the model for random seeds, which creates another dimensionality.

- However, you will notice that the results below are consistent in general across different `eval_ratio`s. 

### Model

- ResNet-18 implemented in `torchvision` used.

- The most top `Conv2d` layer is replaced with `nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)`.

- And max pooling layers are replaced with `nn.Identity()`.

### Optimizer

- Common:

    - learning_rate: `[0.001, 0.005, 0.01]`

    - weight_decay: `0.001`

- SGD w/ momentum & RMSProp:

    - momentum: `0.9`

- Adam & AdamW:

    - betas: `[0.9, 0.999]`

    - epsilon: `1e-8`

### Learning Rate Scheduler

- OneCycleLR:

    - max_lr: `learning_rate * 10`

- CosineAnnealingLR:

    - T_max: `50`

- ExponentialLR:

    - gamma: `0.995`

- ReduceLROnPlateau:

    - metric: `eval/accuracy` (mode: max)

    - patience (tolerance): `1`

## Summary

| run_id | eval_ratio | test_acc | test_loss | epoch_at_max_test_acc | learning_rate | optimizer | lr_scheduler       | train_acc | train_loss | more_finetunig |
|--------|------------|----------|-----------|-----------------------|---------------|-----------|--------------------|-----------|------------|----------------|
|     48 |        0.1 |   0.9402 |    0.2169 |                    50 |         0.01  |     sgd   | OneCycleLR         |    0.998  |   0.01108  |            yes |
|     28 |        0.1 |   0.9384 |    0.2273 |                    50 |         0.005 |     sgd   | OneCycleLR         |    0.999  |   0.007206 |            no  |
|     26 |        0.1 |   0.9359 |    0.3535 |                    45 |         0.005 |     adamw | OneCycleLR         |    0.9991 |   0.003272 |            no  |
|      6 |        0.1 |   0.935  |    0.3303 |                    48 |         0.001 |     adamw | OneCycleLR         |    0.9994 |   0.002192 |            no  |
|     46 |        0.1 |   0.9348 |    0.3101 |                    50 |         0.01  |     adamw | OneCycleLR         |    0.998  |   0.006341 |            no  |
|     38 |        0.1 |   0.927  |    0.365  |                    46 |         0.005 |     adamw | ReduceLROnPlateau  |    0.9982 |   0.006607 |            no  |
|     58 |        0.1 |   0.9189 |    0.3752 |                    45 |         0.01  |     adamw | ReduceLROnPlateau  |    0.9973 |   0.009073 |            no  |
|     18 |        0.1 |   0.9133 |    0.3564 |                    50 |         0.001 |     adamw | ReduceLROnPlateau  |    0.9849 |   0.04224  |            no  |
|     30 |        0.1 |   0.9117 |    0.3175 |                    42 |         0.005 |     adamw | CosineAnnealingLR  |    0.9764 |   0.06818  |            no  |
|     10 |        0.1 |   0.9109 |    0.3331 |                    35 |         0.001 |     adamw | CosineAnnealingLR  |    0.9782 |   0.06269  |            no  |
|    101 |        0.3 |   0.9309 |    0.243  |                    50 |         0.01  |     sgd   | OneCycleLR         |    0.9983 |   0.009917 |            yes |
|     79 |        0.3 |   0.9279 |    0.3727 |                    50 |         0.005 |     adamw | OneCycleLR         |    0.9987 |   0.003954 |            yes |
|     99 |        0.3 |   0.927  |    0.3794 |                    50 |         0.01  |     adamw | OneCycleLR         |    0.9978 |   0.006801 |            yes |
|     66 |        0.3 |   0.924  |    0.3742 |                    49 |         0.001 |     adamw | OneCycleLR         |    0.9995 |   0.002283 |            no  |
|     81 |        0.3 |   0.9232 |    0.2227 |                    50 |         0.005 |     sgd   | OneCycleLR         |    0.9993 |   0.007638 |            yes |
|     83 |        0.3 |   0.9042 |    0.35   |                    45 |         0.005 |     adamw | CosineAnnealingLR  |    0.9729 |   0.07661  |            no  |
|     71 |        0.3 |   0.8976 |    0.396  |                    47 |         0.001 |     adamw | ReduceLROnPlateau  |    0.9812 |   0.05181  |            no  |
|     91 |        0.3 |   0.8959 |    0.4018 |                    39 |         0.005 |     adamw | ReduceLROnPlateau  |    0.9842 |   0.04484  |            no  |
|    103 |        0.3 |   0.8956 |    0.3654 |                    45 |         0.01  |     adamw | CosineAnnealingLR  |    0.9691 |   0.09034  |            no  |
|     70 |        0.3 |   0.8951 |    0.357  |                    45 |         0.001 |     adamw | CosineAnnealingLR  |    0.975  |   0.07186  |            no  |
|    161 |        0.5 |   0.9177 |    0.2947 |                    50 |         0.01  |     sgd   | OneCycleLR         |    0.9989 |   0.009164 |            no  |
|    139 |        0.5 |   0.9116 |    0.4356 |                    48 |         0.005 |     adamw | OneCycleLR         |    0.9987 |   0.004942 |            no  |
|    119 |        0.5 |   0.9098 |    0.4708 |                    50 |         0.001 |     adamw | OneCycleLR         |    0.9993 |   0.0027   |            yes |
|    159 |        0.5 |   0.9066 |    0.4504 |                    50 |         0.01  |     adamw | OneCycleLR         |    0.998  |   0.007488 |            no  |
|    141 |        0.5 |   0.9062 |    0.3372 |                    49 |         0.005 |     sgd   | OneCycleLR         |    0.999  |   0.009976 |            yes |
|    130 |        0.5 |   0.8964 |    0.3314 |                    49 |         0.001 |     adam  | ReduceLROnPlateau  |    0.9727 |   0.0881   |            yes |
|    171 |        0.5 |   0.8873 |    0.5191 |                    49 |         0.01  |     adamw | ReduceLROnPlateau  |    0.9933 |   0.02225  |            yes |
|    131 |        0.5 |   0.8757 |    0.5121 |                    42 |         0.001 |     adamw | ReduceLROnPlateau  |    0.9797 |   0.05641  |            no  |
|    151 |        0.5 |   0.8743 |    0.5023 |                    50 |         0.005 |     adamw | ReduceLROnPlateau  |    0.98   |   0.05736  |            no  |
|    123 |        0.5 |   0.8658 |    0.4321 |                    28 |         0.001 |     adamw | CosineAnnealingLR  |    0.9679 |   0.0933   |            no  |
