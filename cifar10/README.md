# CIFAR-10 Baselines

## Introduction

CIFAR-10 is one of the most popular datasets in the ML/DL society, and ResNet is also one of the most common baselines for the DL papers regardless of sub-fields. 

As a junior researcher, I found out that there is no *de-facto* standard procedure for training ResNet-18 on the CIFAR-10 dataset, which made many researchers and practitioners (including me) confused and time-consuming since it is quite hard to choose the "nice" starting point of each experiment.

While I looked for the baseline for my experiment, I found a nice baseline with detailed explanations on the PyTorch Lightning documentation. Please refer to the [link](https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/cifar10-baseline.html). 

This baseline shows up to `92%` of accuracy without fine-tuning, and up to `94%` of accuracy with fine-tuning using SWA(Stochastic Weight Averaging). I believe that these scores above are firm, reproducible and sufficient enough for most researchers to further investigate their intersets of research. 

For your information, I summarized the details of the baseline created by PyTorch Lightning team below.

```python
# Use torchvision's implementation and modify it to accomodate the input size of (3, 32, 32)
model = torchvision.models.resnet18(pretrained=False, num_classes=10)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()

# Use normalization, random crop, and random horizontal flip for data augmentation
train_transform = T.Compose([
    T.ToTensor(), 
    T.normalize(mean=[0.4915, 0.4823, 0.4468], std=[0.2470, 0.2435, 0.2616]),
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
])
test_transform = T.Compose([
    T.ToTensor(),
    T.normalize(mean=[0.4915, 0.4823, 0.4468], std=[0.2470, 0.2435, 0.2616]),
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

### Best Strategy

| rank | optimizer | learning_rate | lr_scheduler      | 90%_acc | 70%_acc | 50%_acc | 30%_acc | 10%_acc |
|------|-----------|---------------|-------------------|---------|---------|---------|---------|---------|
|   #1 |       sgd |         0.01  | OneCycleLR        |  0.9402 |  0.9309 |  0.9177 |  0.8834 |  0.7642 |
|   #2 |       sgd |         0.005 | OneCycleLR        |  0.9384 |  0.9232 |  0.9062 |  0.8659 |  0.752  |
|   #3 |     adamw |         0.005 | OneCycleLR        |  0.9359 |  0.9279 |  0.9116 |  0.8822 |  0.7598 |
|   #4 |     adamw |         0.001 | OneCycleLR        |  0.935  |  0.924  |  0.9098 |  0.8837 |  0.7863 |
|   #5 |     adamw |         0.01  | OneCycleLR        |  0.9348 |  0.927  |  0.9066 |  0.8734 |  0.7229 |
|   #6 |     adamw |         0.001 | ReduceLROnPlateau |  0.9133 |  0.8976 |  0.8757 |  0.8679 |  0.725  |
|   #7 |      adam |         0.001 | CosineAnnealingLR |  0.9052 |  0.8942 |  0.8562 |  0.8492 |  0.7529 |
|   #8 |       sgd |         0.01  | CosineAnnealingLR |  0.8857 |  0.8692 |  0.821  |  0.7953 |  0.6502 |
|   #9 |      adam |         0.01  | None              |  0.8575 |  0.8441 |  0.8046 |  0.7595 |  0.6177 |
|  #10 |       sgd |         0.005 | CosineAnnealingLR |  0.8595 |  0.8426 |  0.7692 |  0.7481 |  0.5935 |

## Details

### Train with 90% of train set

**[Top 10]**

| run_id | test_acc | test_loss | achieved_at | learning_rate | optimizer | lr_scheduler       | train_acc | train_loss | is_saturated |
|--------|----------|-----------|-------------|---------------|-----------|--------------------|-----------|------------|--------------|
|     48 |   0.9402 |    0.2169 |          50 |         0.01  |     sgd   | OneCycleLR         |    0.998  |   0.01108  |          no  |
|     28 |   0.9384 |    0.2273 |          50 |         0.005 |     sgd   | OneCycleLR         |    0.999  |   0.007206 |          yes |
|     26 |   0.9359 |    0.3535 |          45 |         0.005 |     adamw | OneCycleLR         |    0.9991 |   0.003272 |          yes |
|      6 |   0.935  |    0.3303 |          48 |         0.001 |     adamw | OneCycleLR         |    0.9994 |   0.002192 |          yes |
|     46 |   0.9348 |    0.3101 |          50 |         0.01  |     adamw | OneCycleLR         |    0.998  |   0.006341 |          yes |
|     38 |   0.927  |    0.365  |          46 |         0.005 |     adamw | ReduceLROnPlateau  |    0.9982 |   0.006607 |          yes |
|     58 |   0.9189 |    0.3752 |          45 |         0.01  |     adamw | ReduceLROnPlateau  |    0.9973 |   0.009073 |          yes |
|     18 |   0.9133 |    0.3564 |          50 |         0.001 |     adamw | ReduceLROnPlateau  |    0.9849 |   0.04224  |          yes |
|     30 |   0.9117 |    0.3175 |          42 |         0.005 |     adamw | CosineAnnealingLR  |    0.9764 |   0.06818  |          yes |
|     10 |   0.9109 |    0.3331 |          35 |         0.001 |     adamw | CosineAnnealingLR  |    0.9782 |   0.06269  |          yes |

**[By Optimizer and Learning Rate Scheduler]**

| run_id | test_acc | test_loss | learning_rate | optimizer | lr_scheduler       |
|--------|----------|-----------|---------------|-----------|--------------------|
|     48 |   0.9402 |    0.2169 |         0.01  |     sgd   | OneCycleLR         |
|     60 |   0.8892 |    0.4006 |         0.01  |     sgd   | ReduceLROnPlateau  |
|     52 |   0.8857 |    0.3674 |         0.01  |     sgd   | CosineAnnealingLR  |
|     36 |   0.5059 |    1.345  |         0.005 |     sgd   | ExponentialLR      |
|     44 |   0.4628 |    1.434  |         0.01  |     sgd   | None               |
|      9 |   0.9052 |    0.2984 |         0.001 |     adam  | CosineAnnealingLR  |
|     17 |   0.8871 |    0.3406 |         0.001 |     adam  | ReduceLROnPlateau  |
|     41 |   0.8575 |    0.4682 |         0.01  |     adam  | None               |
|     13 |   0.8097 |    0.5509 |         0.001 |     adam  | ExponentialLR      |
|      5 |   0.6975 |    0.9179 |         0.001 |     adam  | OneCycleLR         |
|     26 |   0.9359 |    0.3535 |         0.005 |     adamw | OneCycleLR         |
|     38 |   0.927  |    0.365  |         0.005 |     adamw | ReduceLROnPlateau  |
|     30 |   0.9117 |    0.3175 |         0.005 |     adamw | CosineAnnealingLR  |
|     42 |   0.862  |    0.5221 |         0.01  |     adamw | None               |
|     14 |   0.795  |    0.5885 |         0.001 |     adamw | ExponentialLR      |
|     43 |   0.8825 |    0.3713 |         0.01  |   rmsprop | None               |
|     19 |   0.8228 |    0.522  |         0.001 |   rmsprop | ReduceLROnPlateau  |
|     11 |   0.7827 |    0.6223 |         0.001 |   rmsprop | CosineAnnealingLR  |
|     15 |   0.7375 |    0.7552 |         0.001 |   rmsprop | ExponentialLR      |
|      7 |   0.4774 |    1.533  |         0.001 |   rmsprop | OneCycleLR         |

**[By Learnig Rate]**

| run_id | test_acc | test_loss | learning_rate | optimizer | lr_scheduler       |
|--------|----------|-----------|---------------|-----------|--------------------|
|      6 |   0.935  |    0.3303 |         0.001 |     adamw | OneCycleLR         |
|      9 |   0.9052 |    0.2984 |         0.001 |     adam  | CosineAnnealingLR  |
|      8 |   0.8918 |    0.3869 |         0.001 |     sgd   | OneCycleLR         |
|     19 |   0.8228 |    0.522  |         0.001 |   rmsprop | ReduceLROnPlateau  |
|     28 |   0.9384 |    0.2273 |         0.005 |     sgd   | OneCycleLR         |
|     26 |   0.9359 |    0.3535 |         0.005 |     adamw | OneCycleLR         |
|     23 |   0.8817 |    0.3945 |         0.005 |   rmsprop | None               |
|     37 |   0.8534 |    0.4374 |         0.005 |     adam  | ReduceLROnPlateau  |
|     48 |   0.9402 |    0.2169 |         0.01  |     sgd   | OneCycleLR         |
|     46 |   0.9348 |    0.3101 |         0.01  |     adamw | OneCycleLR         |
|     43 |   0.8825 |    0.3713 |         0.01  |   rmsprop | None               |
|     41 |   0.8575 |    0.4682 |         0.01  |     adam  | None


### Train with 70% of train set

**[Top 10]**

| run_id | test_acc | test_loss | achieved_at | learning_rate | optimizer | lr_scheduler       | train_acc | train_loss | is_saturated |
|--------|----------|-----------|-------------|---------------|-----------|--------------------|-----------|------------|--------------|
|    101 |   0.9309 |    0.243  |          50 |         0.01  |     sgd   | OneCycleLR         |    0.9983 |   0.009917 |          no  |
|     79 |   0.9279 |    0.3727 |          50 |         0.005 |     adamw | OneCycleLR         |    0.9987 |   0.003954 |          no  |
|     99 |   0.927  |    0.3794 |          50 |         0.01  |     adamw | OneCycleLR         |    0.9978 |   0.006801 |          no  |
|     66 |   0.924  |    0.3742 |          49 |         0.001 |     adamw | OneCycleLR         |    0.9995 |   0.002283 |          yes |
|     81 |   0.9232 |    0.2227 |          50 |         0.005 |     sgd   | OneCycleLR         |    0.9993 |   0.007638 |          no  |
|     83 |   0.9042 |    0.35   |          45 |         0.005 |     adamw | CosineAnnealingLR  |    0.9729 |   0.07661  |          yes |
|     71 |   0.8976 |    0.396  |          47 |         0.001 |     adamw | ReduceLROnPlateau  |    0.9812 |   0.05181  |          yes |
|     91 |   0.8959 |    0.4018 |          39 |         0.005 |     adamw | ReduceLROnPlateau  |    0.9842 |   0.04484  |          yes |
|    103 |   0.8956 |    0.3654 |          45 |         0.01  |     adamw | CosineAnnealingLR  |    0.9691 |   0.09034  |          yes |
|     70 |   0.8951 |    0.357  |          45 |         0.001 |     adamw | CosineAnnealingLR  |    0.975  |   0.07186  |          yes |

### Train with 50% of train set

**[Top 10]**

| run_id | test_acc | test_loss | achieved_at | learning_rate | optimizer | lr_scheduler       | train_acc | train_loss | is_saturated |
|--------|----------|-----------|-------------|---------------|-----------|--------------------|-----------|------------|--------------|
|    161 |   0.9177 |    0.2947 |          50 |         0.01  |     sgd   | OneCycleLR         |    0.9989 |   0.009164 |          yes |
|    139 |   0.9116 |    0.4356 |          48 |         0.005 |     adamw | OneCycleLR         |    0.9987 |   0.004942 |          yes |
|    119 |   0.9098 |    0.4708 |          50 |         0.001 |     adamw | OneCycleLR         |    0.9993 |   0.0027   |          no  |
|    159 |   0.9066 |    0.4504 |          50 |         0.01  |     adamw | OneCycleLR         |    0.998  |   0.007488 |          yes |
|    141 |   0.9062 |    0.3372 |          49 |         0.005 |     sgd   | OneCycleLR         |    0.999  |   0.009976 |          no  |
|    130 |   0.8964 |    0.3314 |          49 |         0.001 |     adam  | ReduceLROnPlateau  |    0.9727 |   0.0881   |          no  |
|    171 |   0.8873 |    0.5191 |          49 |         0.01  |     adamw | ReduceLROnPlateau  |    0.9933 |   0.02225  |          no  |
|    131 |   0.8757 |    0.5121 |          42 |         0.001 |     adamw | ReduceLROnPlateau  |    0.9797 |   0.05641  |          yes |
|    151 |   0.8743 |    0.5023 |          50 |         0.005 |     adamw | ReduceLROnPlateau  |    0.98   |   0.05736  |          yes |
|    123 |   0.8658 |    0.4321 |          28 |         0.001 |     adamw | CosineAnnealingLR  |    0.9679 |   0.0933   |          yes |

### Train with 30% of train set

**[Top 10]**

| run_id | test_acc | test_loss | achieved_at | learning_rate | optimizer | lr_scheduler       | train_acc | train_loss | is_saturated |
|--------|----------|-----------|-------------|---------------|-----------|--------------------|-----------|------------|--------------|
|    179 |   0.8837 |    0.5721 |          46 |         0.001 |     adamw | OneCycleLR         |    0.9991 |   0.004164 |          yes |
|    221 |   0.8834 |    0.5727 |          50 |         0.01  |     sgd   | OneCycleLR         |    0.999  |   0.009764 |          no  |
|    199 |   0.8822 |    0.4525 |          49 |         0.005 |     adamw | OneCycleLR         |    0.9985 |   0.006624 |          yes |
|    219 |   0.8734 |    0.5805 |          50 |         0.01  |     adamw | OneCycleLR         |    0.996  |   0.01333  |          yes |
|    191 |   0.8679 |    0.5729 |          50 |         0.001 |     adamw | ReduceLROnPlateau  |    0.9863 |   0.03996  |          no  |
|    201 |   0.8659 |    0.6266 |          47 |         0.005 |     sgd   | OneCycleLR         |    0.998  |   0.01583  |          yes |
|    182 |   0.8492 |    0.5325 |          50 |         0.001 |     adam  | CosineAnnealingLR  |    0.9276 |   0.221    |          yes |
|    183 |   0.8462 |    0.4634 |          40 |         0.001 |     adamw | CosineAnnealingLR  |    0.9661 |   0.09938  |          yes |
|    178 |   0.8438 |    0.5154 |          48 |         0.001 |     adam  | OneCycleLR         |    0.9178 |   0.2544   |          no  |
|    203 |   0.843  |    0.4856 |          45 |         0.005 |     adamw | CosineAnnealingLR  |    0.9512 |   0.1392   |          yes |

### Train with 10% of train set

**[Top 10]**

| run_id | test_acc | test_loss | achieved_at | learning_rate | optimizer | lr_scheduler       | train_acc | train_loss | is_saturated |
|--------|----------|-----------|-------------|---------------|-----------|--------------------|-----------|------------|--------------|
|    239 |   0.7863 |   1.004   |          44 |         0.001 |     adamw | OneCycleLR         |   0.998   |  0.01365   |          yes |
|    238 |   0.7805 |   0.7304  |          49 |         0.001 |     adam  | OneCycleLR         |   0.9504  |  0.1748    |          yes |
|    281 |   0.7642 |   0.989   |          49 |         0.01  |     sgd   | OneCycleLR         |   0.9986  |  0.01892   |          yes |
|    259 |   0.7598 |   1.026   |          50 |         0.005 |     adamw | OneCycleLR         |   0.9944  |  0.02534   |          yes |
|    242 |   0.7529 |   0.7927  |          43 |         0.001 |     adam  | CosineAnnealingLR  |   0.9292  |  0.2232    |          yes |
|    261 |   0.752  |   1.029   |          50 |         0.005 |     sgd   | OneCycleLR         |   0.9952  |  0.03305   |          yes |
|    243 |   0.7378 |   0.8566  |          48 |         0.001 |     adamw | CosineAnnealingLR  |   0.9628  |  0.1283    |          yes |
|    251 |   0.725  |   0.9926  |          48 |         0.001 |     adamw | ReduceLROnPlateau  |   0.9716  |  0.08939   |          yes |
|    279 |   0.7228 |   1.054   |          49 |         0.01  |     adamw | OneCycleLR         |   0.9768  |  0.07811   |          no  |
|    263 |   0.7155 |   0.9136  |          48 |         0.005 |     adamw | CosineAnnealingLR  |   0.8724  |  0.3633    |          yes |

