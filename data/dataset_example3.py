import numpy as np
import torch

# 设置维度和样本数
dimension = 20
num_samples = 100000

# 生成数据集
data = np.random.normal(0, 0.2, (num_samples, dimension))

# 添加偏置项（第41维）
bias = np.ones((num_samples, 1))
data_with_bias = np.hstack((data, bias))

# 数据划分为训练集和测试集
indices = np.arange(num_samples)
np.random.shuffle(indices)
train_size = int(0.8 * num_samples)
X_train = data_with_bias[indices[:train_size]]
X_test = data_with_bias[indices[train_size:]]

# 生成随机的目标值
Y_train = np.random.randn(train_size)
Y_test = np.random.randn(num_samples - train_size)

# Z-标准化
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

print('X_train:', X_train_tensor.shape)
print('Y_train:', Y_train_tensor.shape)
print('X_test:', X_test_tensor.shape)
print('Y_test:', Y_test_tensor.shape)

torch.save((X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor), './data/dataset_3_20D.pt')
print('dataset.pt saved')
