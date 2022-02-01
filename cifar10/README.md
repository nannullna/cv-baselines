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



## Summary

### Eval ratio = 0.1

| run_id | eval_ratio | test_acc | test_loss | epoch_at_max_test_acc | learning_rate | optimizer | lr_scheduler |
|--------|------------|----------|-----------|-----------------------|---------------|-----------|--------------|
|     48 |        0.1 |   0.9402 |    0.2169 |                    50 |          0.01 |       sgd | OneCycleLR   |



