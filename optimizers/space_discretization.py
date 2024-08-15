import torch
from modules import Simple_Perceptron


def PM(model: Simple_Perceptron, x, y):
    loss = model.loss(model(x), y).mean()
    loss.backward()
    return loss

def SPM(model: Simple_Perceptron, x, y, J=10, h=0.0001):
    loss = 0
    for j in range(J):
        original_params = [model.W.clone(), model.a.clone()]
        for param in model.parameters():
            param.data += h * torch.randn_like(param)
        loss += model.loss(model(x), y).mean()
        model.W.data, model.a.data = original_params
    loss /= J
    loss.backward()
    return loss