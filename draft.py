import torch

# 假设 W 是一个 m x n 的矩阵
m, n = 5, 3  # 例如，m=5, n=3
W = torch.rand(m, n)

# 假设 z_w 是一个长度为 m 的向量
z_w = torch.rand(m)

# 假设 h 是一个标量
h = 2.0

# 计算 z_w'
z_w_prime = z_w * h

# 构造矩阵 Z
Z = z_w_prime.unsqueeze(1).expand(-1, n)

# 计算新的矩阵 W'
W_prime = W + Z

print("Original W:\n", W)
print("z_w:\n", z_w)
print("h:\n", h)
print("z_w_prime:\n", z_w_prime)
print("Z:\n", Z)
print("W_prime:\n", W_prime)