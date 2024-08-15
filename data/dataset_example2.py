import numpy as np
import torch

D = 40
M = 10000

c = np.random.randn(D)

# X取值为【0，5】
X = np.random.rand(M, D) * 5
X_z = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
X_bias = np.hstack([X_z, np.ones((M, 1))])

def target_function(x, c):
    # \sum_{i=1}^{D} c_i * x_i ^ 2
    return np.sum(c * x ** 2)

Y = np.array([])
for x in X:
    Y = np.append(Y, target_function(x, c))

# Y数据进行 z标准化
Y = (Y - Y.mean()) / Y.std()

X_train = torch.tensor(X_bias[:8000], dtype=torch.float32)
Y_train = torch.tensor(Y[:8000], dtype=torch.float32)
X_test = torch.tensor(X_bias[8000:], dtype=torch.float32)
Y_test = torch.tensor(Y[8000:], dtype=torch.float32)

print('X_train:', X_train.shape)
print('Y_train:', Y_train.shape)
print('X_test:', X_test.shape)
print('Y_test:', Y_test.shape)

torch.save((X_train, Y_train, X_test, Y_test), './data/dataset_2_40D.pt')
print('dataset.pt saved')
