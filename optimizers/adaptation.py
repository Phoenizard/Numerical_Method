from modules import Simple_Perceptron
import torch

def anti_adaptation(model: Simple_Perceptron, lr):
    N_a = model.a.grad.clone()
    N_w = model.W.grad.clone()
    adp_lr = lr
    return N_a, N_w, adp_lr

def adaptation(model: Simple_Perceptron, lr, cnt, epsilon=1e-8, beta_1=0.9, beta_2=0.999):
    #=========Nonlinear Term==========
    N_a_init = model.a.grad
    N_w_init = model.W.grad
    model.m_a = beta_1 * model.m_a + (1 - beta_1) * N_a_init
    model.m_w = beta_1 * model.m_w + (1 - beta_1) * N_w_init
    model.v = beta_2 * model.v + (1 - beta_2) * (torch.norm(N_a_init) ** 2 + torch.norm(N_w_init) ** 2)
    m_a_hat = model.m_a / (1 - beta_1 ** (cnt + 1))
    m_w_hat = model.m_w / (1 - beta_1 ** (cnt + 1))
    v_hat = model.v / (1 - beta_2 ** (cnt + 1))
    N_a = m_a_hat
    N_w = m_w_hat
    #=========Time Step Update========
    adaptive_lr = lr / (torch.sqrt(v_hat) + epsilon)
    return N_a, N_w, adaptive_lr