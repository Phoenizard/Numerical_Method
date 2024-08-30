import torch
from modules import Simple_Perceptron
from data import grip_data
import wandb
from tqdm import tqdm
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

lr = 0.001
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
wandb.init(project='Numerical Method', name=f"PM_IEQ_Example_2_{date}", config=config, notes="IEQ 参数分开更新")

for epoch in tqdm(range(epochs)):
    flag = True
    for X, Y in train_loader:
        if flag:
            flag = False
            U = (model.forward(X) - Y.reshape(-1, 1))
        theta_0 = torch.cat([model.W.flatten(), model.a.flatten()]).reshape(-1, 1)
        J = torch.zeros(U.shape[0], theta_0.numel(), device=device)
        #=========Remain to be Update=============
        for i in range(U.shape[0]):
            U[i].backward(retain_graph=True)
            J[i] = torch.cat([model.W.grad.flatten(), model.a.grad.flatten()])
            model.W.grad.zero_()
            model.a.grad.zero_()
        #=========================================
        with torch.no_grad():
            J_T = J.T.to(device)
            # 计算量A，A=I + 2(J^T)J

            A = torch.eye(theta_0.numel(), device=device) + 2 * torch.mm(J_T, J)
            
            # Cholesky 分解和矩阵求逆
            L = torch.linalg.cholesky(A)
            L_inv = torch.inverse(L)
            A_inv = torch.mm(L_inv.T, L_inv)
            # 计算theta_1
            theta_1 = theta_0 - 2 * lr * torch.mm(torch.mm(A_inv, J_T), U)
            # 更新参数
            model.W.data = theta_1[:model.W.numel()].reshape(model.W.shape)
            model.a.data = theta_1[model.W.numel():].reshape(model.a.shape)
            # U = (I - 2 * lr * J * A^-1 * J^T) * U
            U = U - 2 * lr * torch.mm(J, torch.mm(A_inv, torch.mm(J_T, U)))

            model.W.grad.zero_()
            model.a.grad.zero_()