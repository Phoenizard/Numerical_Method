import torch
from modules import Simple_Perceptron
from data import grip_data
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Simple_Perceptron.Simple_Perceptron(41, 100, 1)
train_loader, X_train, Y_train, X_test, Y_test, D = grip_data.load_data(device='cuda')
epochs = 10000
lr = 1
m = 100
D = 40
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
wandb.init(project='Numerical Method', name=f"PM_IEQ_Example_2_{date}", config=config)

for epoch in range(epochs):
    flag = True
    for X, Y in train_loader:
        U = (model.forward(X) - Y.reshape(-1, 1))
        theta_0 = torch.cat([model.W.flatten(), model.a.flatten()]).reshape(-1, 1)
        J = torch.zeros(U.shape[0], theta_0.numel())
        for i in range(U.shape[0]):
            U[i].backward(retain_graph=True)
            J[i] = torch.cat([model.W.grad.flatten(), model.a.grad.flatten()])
            model.W.grad.zero_()
            model.a.grad.zero_()
        with torch.no_grad():
            J_T = J.T
            # 计算量A，A=I + 2(J^T)J
            A = torch.eye(theta_0.numel(), device=device) + 2 * torch.mm(J_T, J)
            A_inv = torch.inverse(A)
            theta_1 = theta_0 - 2 * lr * torch.mm(torch.mm(A_inv, J_T), U)
            # 更新参数
            model.W.data = theta_1[:model.W.numel()].reshape(model.W.shape)
            model.a.data = theta_1[model.W.numel():].reshape(model.a.shape)
            model.W.grad.zero_()
            model.a.grad.zero_()

    with torch.no_grad():
        train_loss = model.loss(model(X_train), Y_train).mean()
        test_loss = model.loss(model(X_test), Y_test).mean()
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        norm = model.get_norm(model)
        wandb.log({'epoch': epoch + 1,
                   'train_loss': train_loss, 
                   'test_loss': test_loss,
                   'norm_W': norm[0],
                   'norm_a': norm[1],
                   'accuracy': 1 - test_loss})
        print(f'epoch {epoch + 1}, loss {train_loss:.4f}, test loss {test_loss:.4f}')