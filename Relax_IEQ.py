import torch
from modules import Simple_Perceptron
from data import grip_data
import wandb
from tqdm import tqdm
import time
import warnings
from utils import G_modified, validate, G_modified_CUDA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

lr = 1
m = 100
D = 40
l = 64
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
wandb.init(project='Numerical Method', name=f"PM_IEQ_J_Trans_{date}", config=config, notes="IEQ 转化Jacobian, CUDA并行计算")

for epoch in tqdm(range(epochs)):
    flag = True
    for X, Y in train_loader:
        if flag:
            U = (model.forward(X) - Y.reshape(-1, 1))
            flag = False
        J = G_modified_CUDA(X, model)
        with torch.no_grad():
            theta_0 = torch.cat([model.W.flatten(), model.a.flatten()]).reshape(-1, 1)
            J_T = J.T
            # 计算量A，A=I + 2lr(J^T)J
            A = torch.eye(theta_0.numel(), device=device) + 2 * lr * torch.mm(J_T, J)
            L = torch.linalg.cholesky(A)
            A_inv = torch.cholesky_inverse(L)
            # A_inv = torch.inverse(A)
            U_wave = (torch.eye(U.numel(), device=device) - 2 * lr * torch.mm(J, torch.mm(A_inv, J_T))) @ U 
            theta_1 = theta_0 - 2 * lr * torch.mm(torch.mm(A_inv, J_T), U_wave)
            model.W.data = theta_1[:(D + 1) * m].reshape(D + 1, m)
            model.a.data = theta_1[(D + 1) * m:].reshape(m, 1)
            # wandb.log({'U_norm': torch.norm(U).item(),
            #           'J_norm': torch.norm(J).item()})
            #===============================Relaxation===============================
            U_hat = (model.forward(X) - Y.reshape(-1, 1))
            a = torch.norm(U_wave - U_hat) ** 2
            b = 2 * torch.dot(U_hat.flatten(), (U_wave - U_hat).flatten())
            c = torch.norm(U_hat) ** 2 - torch.norm(U_wave) ** 2 - 0.99 * torch.norm(theta_1 - theta_0) ** 2 / lr
            if a == 0:
                warnings.warn("a == 0")
                ellipsis_0 = 0
            elif (b ** 2 - 4 * a * c) < 0:
                warnings.warn("b^2 - 4ac < 0")
                ellipsis_0 = 0
            else:
                ellipsis_0 = max((-b - torch.sqrt(b ** 2 - 4 * a * c)) / (2 * a), 0)
            if ellipsis_0 > 1:
                warnings.warn("ellipsis_0 > 1")
                ellipsis_0 = 1
            U = ellipsis_0 * U_wave + (1 - ellipsis_0) * U_hat
            #=======================================================================
            wandb.log({'ellipsis': ellipsis_0})
    with torch.no_grad():
        train_loss = model.loss(model(X_train), Y_train).mean()
        test_loss = model.loss(model(X_test), Y_test).mean()
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        norm = model.get_norm()
        wandb.log({'epoch': epoch + 1,
                   'train_loss': train_loss, 
                   'test_loss': test_loss,
                   'norm_W': norm[0],
                   'norm_a': norm[1]})
        # print(f'epoch {epoch + 1}, loss {train_loss:.8f}, test loss {test_loss:.8f}')