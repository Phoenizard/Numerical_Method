import torch
from modules import Simple_Perceptron
from data import grip_data
import wandb
from tqdm import tqdm
import time
from utils import G_modified, validate

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
wandb.init(project='Numerical Method', name=f"PM_IEQ_ManulGrad_{date}", config=config, notes="IEQ 转化Jacobian")

for epoch in tqdm(range(epochs)):
    flag = True
    for X, Y in train_loader:
        if flag:
            U = (model.forward(X) - Y.reshape(-1, 1))
            flag = False
        J = G_modified(X, model)
        with torch.no_grad():
            theta_0 = torch.cat([model.W.flatten(), model.a.flatten()]).reshape(-1, 1)
            J_T = J.T
            # 计算量A，A=I + 2lr(J^T)J
            A = torch.eye(theta_0.numel(), device=device) + 2 * (lr / l) * torch.mm(J_T, J)
            L = torch.linalg.cholesky(A)
            A_inv = torch.cholesky_inverse(L)
            theta_1 = theta_0 - 2 * (lr / l) * torch.mm(torch.mm(A_inv, J_T), U)
            model.W.data = theta_1[:(D + 1) * m].reshape(D + 1, m)
            model.a.data = theta_1[(D + 1) * m:].reshape(m, 1)
            U = (torch.eye(U.numel(), device=device) - 2 * (lr / l) * torch.mm(J, torch.mm(A_inv, J_T))) @ U 
            # wandb.log({'U_norm': torch.norm(U).item(),
            #           'J_norm': torch.norm(J).item()})
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