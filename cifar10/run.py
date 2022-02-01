import os
import argparse
import requests
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, 
)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import torchvision
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

import wandb
from tqdm.auto import tqdm, trange

def create_optimizer(config, model: nn.Module):
    if config.optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    elif config.optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer_type == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer_type == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    return optimizer

def create_scheduler(config, optimizer):
    # ["none", "OneCycleLR", "CosineAnnealingLR", "ExponentialLR", "ReduceLROnPlateau"]
    if config.lr_scheduler_type == "none":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: config.learning_rate)
    elif config.lr_scheduler_type == "OneCycleLR":
        steps_per_epoch = int(50000 * (1-config.eval_ratio)) // config.batch_size + 1
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=config.learning_rate * 10,
            epochs=config.num_epochs,
            steps_per_epoch=steps_per_epoch,
        )
    elif config.lr_scheduler_type == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    elif config.lr_scheduler_type == "ExponentialLR":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    elif config.lr_scheduler_type == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max")
    return scheduler

def main(config):

    eval_ratio = config.eval_ratio
    weight_decay = config.weight_decay
    momentum = config.momentum

    dataset_path = "/opt/datasets/cifar10"

    mean = [0.4915, 0.4823, 0.4468]
    std  = [0.2470, 0.2435, 0.2616]

    train_transform = T.Compose([
        T.ToTensor(), 
        T.Normalize(mean=mean, std=std),
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    train_set = CIFAR10(root=dataset_path, train=True,  transform=train_transform, download=True)
    test_set  = CIFAR10(root=dataset_path, train=False, transform=test_transform,  download=True)

    print(f"eval ratio {eval_ratio} has started!")

    # -------------------- #
    # Model Initialization #
    # -------------------- #
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    # modify architecture to accomodate (3, 32, 32) 

    # --------------------- #
    # Train/Eval/Test split #
    # --------------------- #
    train_ids = np.random.choice(np.arange(len(train_set)), size=int(len(train_set)*(1-eval_ratio)), replace=False)
    eval_ids  = np.arange(len(train_set))[np.invert(np.isin(np.arange(len(train_set)), train_ids))]

    train_subset = Subset(train_set, train_ids)
    eval_subset  = Subset(train_set, eval_ids)

    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
    eval_loader  = DataLoader(eval_subset,  batch_size=config.batch_size, shuffle=False)
    test_loader  = DataLoader(test_set,     batch_size=config.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(args, model)
    scheduler = create_scheduler(args, optimizer)

    model.cuda()

    wandb.watch(model, log='all')

    for epoch in trange(config.num_epochs):

        # ------------- #
        #     Train     #
        # ------------- #
        preds = []
        targets = []
        losses = []

        model.train()
        for X, y in train_loader:
            optimizer.zero_grad()

            X = X.cuda()
            y = y.cuda()

            out = model(X)
            loss = criterion(out, y)
            loss.backward()

            optimizer.step()
            if config.lr_scheduler_type in ["OneCycleLR", "CosineAnnealingLR", "ExponentialLR"]:
                scheduler.step()

            pred = torch.argmax(out, dim=1)
            preds.extend(pred.tolist())
            targets.extend(y.tolist())
            losses.append(loss.item())

        acc = accuracy_score(targets, preds)
        lss = np.mean(losses)

        wandb.log({"train/accuracy": acc, "train/loss": lss, "epoch": epoch})

        # -------------- #
        #    Evaluate    #
        # -------------- #
        preds = []
        targets = []
        losses = []

        model.eval()
        with torch.no_grad():

            for X, y in eval_loader:
                X = X.cuda()
                y = y.cuda()
                out = model(X)
                loss = criterion(out, y)

                pred = torch.argmax(out, dim=1)
                preds.extend(pred.tolist())
                targets.extend(y.tolist())
                losses.append(loss.item())

        acc = accuracy_score(targets, preds)
        lss = np.mean(losses)

        if config.lr_scheduler_type in ["ReduceLROnPlateau"]:
            scheduler.step(acc)

        wandb.log({"eval/accuracy": acc, "eval/loss": lss, "epoch": epoch})

        # -------------- #
        #      Test      #
        # -------------- #
        preds = []
        targets = []
        losses = []

        with torch.no_grad():
            for X, y in tqdm(test_loader):
                X = X.cuda()
                y = y.cuda()
                out = model(X)
                loss = criterion(out, y)

                pred = torch.argmax(out, dim=1)
                preds.extend(pred.tolist())
                targets.extend(y.tolist())
                losses.append(loss.item())

        test_acc  = accuracy_score(targets, preds)
        test_loss = np.mean(losses)

        wandb.log({"test/accuracy": test_acc, "test/loss": test_loss, "epoch": epoch})

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int)

    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--eval_ratio', type=float, default=0.01)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--optimizer_type', type=str, choices=["sgd", "rmsprop", "adam", "adamw"], default="sgd")
    parser.add_argument('--lr_scheduler_type', type=str, choices=["none", "OneCycleLR", "CosineAnnealingLR", "ExponentialLR", "ReduceLROnPlateau"])

    args = parser.parse_args()

    wandb.init(config=args)
    config = wandb.config

    wandb.define_metric("train/accuracy", summary="max")
    wandb.define_metric("train/loss", summary="min")
    wandb.define_metric("eval/accuracy", summary="max")
    wandb.define_metric("eval/loss", summary="min")
    wandb.define_metric("test/accuracy", summary="max")
    wandb.define_metric("test/loss", summary="min")

    main(config)