from modules import Simple_Perceptron
import torch
import warnings
from optimizers.space_discretization import PM

def Euler(model: Simple_Perceptron, N_a, N_w, lr):
    with torch.no_grad():
        model.W -= lr * N_w
        model.a -= lr * N_a
        model.W.grad.zero_()
        model.a.grad.zero_()

def SAV(model: Simple_Perceptron, N_a, N_w, lr, loss, C=1, _lambda=0):
    D, m = (model.W.shape[0] - 1), model.W.shape[1]
    theta = torch.cat([model.W.flatten(), model.a.flatten()]).reshape(-1, 1)
    N_theta = torch.cat([N_w.flatten(), N_a.flatten()]).reshape(-1, 1)
    theta_1_2 = -lr * N_theta / (torch.sqrt(loss + C) * (1 + lr * _lambda))
    # theta_a_2 = -lr * N_a / (torch.sqrt(loss + C) * (1 + lr * _lambda))
    # theta_w_2 = -lr * N_w / (torch.sqrt(loss + C) * (1 + lr * _lambda))
    model.r = model.r / (1 + lr * torch.sum(N_theta * (N_theta / (1 + lr * _lambda))) / (2 * (loss + C)))
    with torch.no_grad():
        theta += model.r.item() * theta_1_2
        model.W.data = theta[:(D + 1) * m].reshape(D + 1, m)
        model.a.data = theta[(D + 1) * m:].reshape(m, 1)
        model.a.grad.zero_()
        model.W.grad.zero_()


def ESAV(model: Simple_Perceptron, N_a, N_w, lr, loss, _lambda=0):
    linear_N_a = N_a / (1 + _lambda * lr)
    linear_N_w = N_w / (1 + _lambda * lr)
    theta_a_2 = - linear_N_a * lr / (torch.exp(loss))
    theta_w_2 = - linear_N_w * lr / (torch.exp(loss))
    #=========Update SAV R================
    model.r = model.r / (1 + lr * (torch.sum(N_a * linear_N_a) + torch.sum(N_w * linear_N_w)))
    with torch.no_grad():
        #=========Update Params================
        model.a += theta_a_2 * model.r.item()
        model.W += theta_w_2 * model.r.item()
        model.a.grad.zero_()
        model.W.grad.zero_()

def MESAV(model: Simple_Perceptron, N_a, N_w, lr, loss, E_0=0, _lambda=0):
    linear_N_a = N_a / (1 + _lambda * lr)
    linear_N_w = N_w / (1 + _lambda * lr)
    theta_a_2 = - linear_N_a * E_0 * lr / (torch.exp(loss))
    theta_w_2 = - linear_N_w * E_0 * lr / (torch.exp(loss))
    #=========Update SAV R================
    model.r = model.r / (1 + lr * (torch.sum(N_a * linear_N_a) + torch.sum(N_w * linear_N_w)))
    with torch.no_grad():
        #=========Update Params================
        model.a += theta_a_2 * model.r.item()
        model.W += theta_w_2 * model.r.item()
        model.a.grad.zero_()
        model.W.grad.zero_()

def ReSAV(model: Simple_Perceptron, N_a, N_w, lr, loss, C=1, _lambda=0):
    model.r = torch.sqrt(loss + C)
    theta_a_2 = -lr * N_a / (torch.sqrt(loss + C) * (1 + lr * _lambda))
    theta_w_2 = -lr * N_w / (torch.sqrt(loss + C) * (1 + lr * _lambda))
    model.r = model.r / (1 + lr * (torch.sum(N_a * (N_a / (1 + lr * _lambda))) + torch.sum(N_w * (N_w / (1 + lr * _lambda)))) / (2 * (loss + C)))
    with torch.no_grad():
        model.a += model.r.item() * theta_a_2
        model.W += model.r.item() * theta_w_2
        model.a.grad.zero_()
        model.W.grad.zero_()

def RelSAV(model: Simple_Perceptron, N_a, N_w, lr, loss, X, Y, ratio_n = 0.99, C=1, _lambda=4):
    #===============Update the parameters in SAV================
    theta_a_1 = model.a.clone()
    theta_w_1 = model.W.clone()
    theta_a_2 = -lr * N_a / (torch.sqrt(loss + C) * (1 + lr * _lambda))
    theta_w_2 = -lr * N_w / (torch.sqrt(loss + C) * (1 + lr * _lambda))
    r_wave = model.r / (1 + lr * (torch.sum(N_a * (N_a / (1 + lr * _lambda))) + torch.sum(N_w * (N_w) / (1 + lr * _lambda))) / (2 * (loss + C)))
    with torch.no_grad():
        model.a += r_wave.item() * theta_a_2
        model.W += r_wave.item() * theta_w_2
        model.a.grad.zero_()
        model.W.grad.zero_()
        tmp_loss = model.loss(model(X), Y).mean()
    #===============Update r in SAV================
    r_hat = torch.sqrt(tmp_loss + C)
    a = (r_wave - r_hat) ** 2
    b = 2 * r_hat * (r_wave - r_hat)
    c = r_hat ** 2 - r_wave ** 2 -  ratio_n * (torch.norm(model.a - theta_a_1) ** 2 + torch.norm(model.W - theta_w_1) ** 2) / lr
    if a == 0:
        warnings.warn("a出现为0的情况", UserWarning)
        ellipsis_0 = 0
    elif (b ** 2 - 4 * a * c) < 0:
        warnings.warn("b^2 - 4ac < 0", UserWarning)
        ellipsis_0 = 0
    else: 
        ellipsis_0 = max((-b - torch.sqrt(b ** 2 - 4 * a * c)) / (2 * a), 0)
    model.r = ellipsis_0 * r_wave + (1 - ellipsis_0) * r_hat
    return ellipsis_0


def RelESAV(model: Simple_Perceptron, N_a, N_w, lr, loss, X, Y, ratio_n = 0.99, _lambda=4):
    with torch.no_grad():
        #=========Update Linear N================
        linear_N_a = N_a / (1 + _lambda * lr)
        linear_N_w = N_w / (1 + _lambda * lr)
        #=========Update Theta================
        theta_a_1 = model.a.data
        theta_w_1 = model.W.data
        theta_a_2 = - linear_N_a * lr / (torch.exp(loss))
        theta_w_2 = - linear_N_w * lr / (torch.exp(loss))
        #=========Update ESAV R================
        r_wave = model.r / (1 + lr * (torch.sum(N_a * linear_N_a) + torch.sum(N_w * linear_N_w)))
        #=========Update Params================
        model.a += model.r.item() * theta_a_2
        model.W += model.r.item() * theta_w_2
        model.a.grad.zero_()
        model.W.grad.zero_()

    new_loss = PM(model, X, Y)

    with torch.no_grad():
        #=========Update Relaxation================
        r_hat = torch.exp(new_loss)
        a = (r_wave - r_hat) ** 2
        b = 2 * r_hat * (r_wave - r_hat)
        c = r_hat ** 2 - r_wave ** 2 - ratio_n * ((torch.norm(model.a.data - theta_a_1) ** 2 + torch.norm(model.W.data - theta_w_1) ** 2)) / lr
        if a == 0:
            ellipsis_0 = 0
            warnings.warn("a出现为0的情况", UserWarning)
        elif (b ** 2 - 4 * a * c) < 0:
            ellipsis_0 = 0
            warnings.warn("b^2 - 4ac < 0", UserWarning)
        else: 
            ellipsis_0 = max((-b - torch.sqrt(b ** 2 - 4 * a * c)) / (2 * a), 0)
        model.r = ellipsis_0 * r_wave + (1 - ellipsis_0) * r_hat
    return ellipsis_0  