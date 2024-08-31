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
wandb.init(project='Numerical Method', name=f"PM_IEQ_ManulGrad_{date}", config=config, notes="IEQ手动参数加速")

for epoch in tqdm(range(epochs)):
    flag = True
    for X, Y in train_loader:
        if flag:
            X = X.to(device)
            Y = Y.to(device)
            flag = False
            U = (model(X) - Y.reshape(-1, 1))
        theta_0 = torch.cat([model.W.flatten(), model.a.flatten()]).reshape(-1, 1)
        #=====Jacobian 矩阵=========================
        J = G_modified(X, model)
        #===========================================
        # 转置矩阵 J_T
        with torch.no_grad():
            J_T = J.T.to(device)
            A = torch.eye(theta_0.numel(), device=device) + 2 * lr * torch.mm(J_T, J)
            L = torch.linalg.cholesky(A)
            A_inv = torch.cholesky_inverse(L)

            theta_1 = theta_0 - 2 * lr * torch.mm(torch.mm(A_inv, J_T), U)
            model.W.data = theta_1[:model.W.numel()].reshape(model.W.shape)
            model.a.data = theta_1[model.W.numel():].reshape(model.a.shape)
            
            U = U - 2 * lr * torch.mm(J, torch.mm(A_inv, torch.mm(J_T, U)))
            # 更新参数

    validate(model, X_train, Y_train, X_test, Y_test, epoch, is_recoard=True)