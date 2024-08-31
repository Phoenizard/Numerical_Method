import torch
from modules import Simple_Perceptron
from data import grip_data
import wandb
from tqdm import tqdm
import time
from utils import G_modified, validate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

lr = 1
m = 100
D = 40
model = Simple_Perceptron.Simple_Perceptron(41, m, 1).to(device)
train_loader, X_train, Y_train, X_test, Y_test, D = grip_data.load_data(device=device)
epochs = 10000

train_losses = []
test_losses = []

config = {
    'learning_rate': lr,
    'batch_size': 64,
    'epochs': epochs,
    'hidden_layer': m,
    'input': D + 1,
    'output': 1,
    'optimizer': 'IEQ'
}

import datetime

date = datetime.datetime.now().strftime("%m%d%H%M")
wandb.init(project='Numerical Method', name=f"PM_IEQ_ManulGrad_{date}", config=config, notes="IEQ手动参数加速+新差分")

for epoch in tqdm(range(epochs)):
    flag = True
    for X, Y in train_loader:
        if flag:
            U = (model(X) - Y.reshape(-1, 1))
            flag = False
        # 增广模型中的参数
        theta_0 = torch.cat([model.W.flatten(), model.a.flatten()]).reshape(-1, 1)
        #=====Jacobian 矩阵=========================
        J = G_modified(X, model)
        #===========================================
        # 转置矩阵 J_T
        with torch.no_grad():
        # 计算 A = I + 2Δt * J_n * J_n^T，确保 A 在 CUDA 上
            A = torch.eye(J.shape[0], device=device) + 2 * lr * torch.mm(J, J.T)
            
            # # 使用 Cholesky 分解计算 A 的逆矩阵，确保操作在 CUDA 上
            # L = torch.linalg.cholesky(A)
            # A_inv = torch.cholesky_inverse(L)
            
            A_inv = torch.inverse(A).to(device)

            # 更新 U^{n+1}
            U_1 = torch.mm(A_inv, U)
            
            # 更新 theta^{n+1}
            theta_1 = theta_0 - 2 * lr * torch.mm(J.T, U_1)
            
            # 更新模型参数，确保更新后的参数在 GPU 上
            model.W.data = theta_1[:model.W.numel()].reshape(model.W.shape)
            model.a.data = theta_1[model.W.numel():].reshape(model.a.shape)
            
            # 更新 U_n 和 theta_n
            U = U_1
            wandb.log({'J_norm': torch.norm(J).item()})
    validate(model, X_train, Y_train, X_test, Y_test, epoch, is_recoard=True)