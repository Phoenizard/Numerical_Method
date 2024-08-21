from torch import nn
import torch

class Simple_Perceptron(nn.Module):
    def __init__(self, input, hidden_layer, output):
        super(Simple_Perceptron, self).__init__()
        self.relu = nn.ReLU()
        self.hidden_dim = hidden_layer
        self.W = nn.Parameter(torch.rand(input, hidden_layer), requires_grad=True)
        # HE初始化
        nn.init.kaiming_normal_(self.W, mode='fan_in', nonlinearity='relu')
        self.a = nn.Parameter(torch.rand(hidden_layer, output), requires_grad=True)
        nn.init.kaiming_normal_(self.a, mode='fan_in', nonlinearity='relu')
        #========SAV Params=======
        self.r = 0
        #========Adaptation Params=======
        self.m_a, self.m_w, self.v = 0, 0, 0


    def forward(self, x):
        # print(x.shape)
        z1 = self.relu(torch.mm(x, self.W))
        # print(z1.shape)
        z2 = torch.mm(z1, self.a) / self.hidden_dim
        return z2
    
    def loss(self, y_pred, y_true):
        return (y_pred - y_true.reshape(y_pred.shape)) ** 2

    # 计算模型W和a的Norm
    def get_norm(self):
        return [torch.norm(self.W).item(), torch.norm(self.a).item()]