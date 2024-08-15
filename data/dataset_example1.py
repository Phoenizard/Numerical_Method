import numpy as np
import torch

D = 40
M = 10000

p = np.random.randn(D)
q = np.random.randn(D)

X = np.random.rand(M, D)
X_z = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
X_bias = np.hstack([X_z, np.ones((M, 1))])

def target_function(x, p, q):
    return np.sin(np.dot(p, x)) + np.cos(np.dot(q, x))

Y = np.array([])
for x in X:
    Y = np.append(Y, target_function(x, p, q))

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

torch.save((X_train, Y_train, X_test, Y_test), './data/dataset_1_40D.pt')
print('dataset.pt saved')
