## ResNet

![ResNet](resnet.png)

## Linear Probing Accuracy

**CIFAR10 on ResNet-18**

| first p% of train set | Layer 1 | Layer 2 | Layer 3 | Layer 4 |
|-----------------------|---------|---------|---------|---------|
|                  100% |   0.699 |   0.807 |   0.887 |   0.908 |
|                   10% |   0.631 |   0.730 |   0.823 |   0.877 |
|                    1% |   0.488 |   0.576 |   0.687 |   0.819 |
|                  0.1% |   0.287 |   0.353 |   0.462 |   0.571 |

\[Details\]

* Adam optimizer with lr=`1e-3`, betas=`[0.9, 0.999]`, and no weight decay

* Fully trained with 50,000 train examples
