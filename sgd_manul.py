import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import wandb
import random
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seed for reproducibility
np.random.seed(100)
torch.manual_seed(100)


D = 40
M = 100000
batch_size = 256
p = np.random.randn(D)
q = np.random.randn(D)

def generate_data(M, D, p, q):
    X = np.random.rand(M, D)
    f_star = np.sin(X @ p) + np.cos(X @ q)
    # Z-score normalization
    X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    # X_normalized = X
    # Add bias column
    X_normalized = np.hstack((X_normalized, np.ones((M, 1))))
    return X_normalized, f_star

features, labels = generate_data(M, D, p, q)
labels = labels.reshape(-1, 1)
# Convert to PyTorch tensors
features = torch.tensor(features, dtype=torch.float32, device=device)
labels = torch.tensor(labels, dtype=torch.float32, device=device)
# 8:2 split
train_size = int(0.8 * M)
train_features = features[:train_size]
train_labels = labels[:train_size]
test_features = features[train_size:]
test_labels = labels[train_size:]


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

m = 1000
# 定义模型参数
w = torch.normal(0, 0.1, size=(D + 1,m), requires_grad=True, device=device)
a = torch.normal(0, 0.1, size=(m, 1), requires_grad=True, device=device)

def simple_net(X, w, a):
    return (X @ w).relu() @ a / m

def squared_loss(y_hat, y):
    y = y.reshape(y_hat.shape)
    return ((y_hat - y) ** 2)

def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


lr = 0.01
num_epochs = 10000
net = simple_net
loss = squared_loss

train_losses = []
test_losses = []

config = {
    "lr": lr,
    "num_epochs": num_epochs,
    "batch_size": batch_size, 
    "M": M,
    "Optimizer": "SGD"
}

# wandb.init(project="Numerical Method", name="SGD_0916_Example1_1e2")

for epoch in range(num_epochs):
    train_l = 0.0
    for X, y in data_iter(batch_size, train_features, train_labels): 
        l = loss(net(X, w, a), y)  # X和y的小批量损失
        l.sum().backward()
        sgd([w, a], lr, batch_size)  # 使用参数的梯度更新参数
        train_l += l.sum()
    train_l = train_l / len(train_labels)
    with torch.no_grad():
        test_l = loss(net(test_features, w, a), test_labels).mean().item()
        print(f'epoch {epoch + 1}, train loss {float(train_l):8f},test loss {float(test_l):8f}')
    # wandb.log({
    #     "epoch": epoch,
    #     "train_loss": train_l.item(),
    #     "test_loss": test_l
    # })



