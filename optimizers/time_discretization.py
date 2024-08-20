from modules import Simple_Perceptron
import torch

def Euler(model: Simple_Perceptron, N_a, N_w, lr):
    with torch.no_grad():
        model.W -= lr * N_w
        model.a -= lr * N_a
        model.W.grad.zero_()
        model.a.grad.zero_()

def SAV(model: Simple_Perceptron, N_a, N_w, lr, loss, C=1, _lambda=0):
    theta_a_2 = -lr * N_a / (torch.sqrt(loss + C) * (1 + lr * _lambda))
    theta_w_2 = -lr * N_w / (torch.sqrt(loss + C) * (1 + lr * _lambda))
    model.r = model.r / (1 + lr * (torch.sum(N_a * (N_a / (1 + lr * _lambda))) + torch.sum(N_w * (N_w) / (1 + lr * _lambda))) / (2 * (loss + C)))
    with torch.no_grad():
        model.a += model.r.item() * theta_a_2
        model.W += model.r.item() * theta_w_2
        model.a.grad.zero_()
        model.W.grad.zero_()


def ESAV(model: Simple_Perceptron, N_a, N_w, lr, loss, _lambda=0):
    theta_w_1 = model.W.data
    theta_a_1 = model.a.data
    b_a = r * N_a / torch.exp(loss)
    b_w = r * N_w / torch.exp(loss)
    with torch.no_grad():
        model.a.data = (theta_a_1 - lr * b_a) / (1 + lr * _lambda)
        model.W.data = (theta_w_1 - lr * b_w) / (1 + lr * _lambda)
        #=========Update SAV R================
        r = torch.exp(torch.log(r) + torch.sum(b_a * (model.a - theta_a_1)) + torch.sum(b_w * (model.W - theta_w_1)))
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

def RelSAV(model: Simple_Perceptron, N_a, N_w,lr, loss, X, Y, ratio_n = 0.99, C=1, _lambda=4):
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
        print('a == 0')
        ellipsis_0 = 0
    elif (b ** 2 - 4 * a * c) < 0:
        ellipsis_0 = 0
        print('b^2 - 4ac < 0')
    else: 
        ellipsis_0 = max((-b - torch.sqrt(b ** 2 - 4 * a * c)) / (2 * a), 0)
    model.r = ellipsis_0 * r_wave + (1 - ellipsis_0) * r_hat
    return ellipsis_0