import numpy as np
import torch

D = 20
M = 100000

# X取值为N(0, 0.2)
X = np.random.randn(M, D) * 0.2
X_z = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
X_bias = np.hstack([X_z, np.ones((M, 1))])

def target_function(x):
    # exp(-10 * ||x||)
    return np.exp(-10 * np.linalg.norm(x))

Y = np.array([])
for x in X:
    Y = np.append(Y, target_function(x))

# Y数据进行 z标准化
Y = (Y - Y.mean()) / Y.std()

size = int(0.8 * M)
X_train = torch.tensor(X_bias[:size], dtype=torch.float32)
Y_train = torch.tensor(Y[:size], dtype=torch.float32)
X_test = torch.tensor(X_bias[size:], dtype=torch.float32)
Y_test = torch.tensor(Y[size:], dtype=torch.float32)

print('X_train:', X_train.shape)
print('Y_train:', Y_train.shape)
print('X_test:', X_test.shape)
print('Y_test:', Y_test.shape)

torch.save((X_train, Y_train, X_test, Y_test), './data/dataset_3_20D.pt')
print('dataset.pt saved')
