import torch, wandb, datetime
from modules.Simple_Perceptron import Simple_Perceptron
from data.grip_data import load_data
import time

def Transpose_Jacobian_Order(m, D, device=None):
    P = torch.zeros((m*(D+1), m*(D+1)), device=device)
    for i in range(D):
        for j in range(m):
            P[j*D + i, i*m + j] = 1
    # 令最后m行m列为单位矩阵
    for i in range(m*D, m*(D+1)):
        P[i, i] = 1
    return P

def G_modified(X, model):
    # 开始计时
    start = time.time()
    
    input_dim, m = model.W.shape  # m: 隐藏层神经元数量, input_dim: 输入维度
    batch_size = X.shape[0]       # batch_size: 批处理大小
    
    # 初始化 Jacobian 矩阵 J，大小为 (batch_size, m * (input_dim + 1))
    J = torch.zeros(batch_size, m * (input_dim + 1), device=X.device)
    
    # 计算所有样本的 <w_i, x> 和 ReLU 激活
    relu_input = X @ model.W  # (batch_size, m)
    relu_output = torch.relu(relu_input)  # (batch_size, m)
    
    # 对 w_i 的部分并行计算 Jacobian
    W_grad = torch.zeros(batch_size, input_dim, m)
    for j in range(m):
        mask = (relu_input[:, j] > 0).float()  # 只选择 ReLU 激活大于0的元素
        W_grad[:, :, j] = model.a[j] * X * mask.view(-1, 1) / m
    J[:, :m*input_dim] = W_grad.reshape(W_grad.shape[0], -1)
    # 对 a_i 的部分并行计算 Jacobian
    J[:, m*input_dim:] = relu_output / m
    end = time.time()
    print("计算Jacobian矩阵耗时：", end - start)
    return J

def G_modified_CUDA(X, model):
    # 开始计时
    # start = time.time()
    # 确保所有张量在 CUDA 设备上
    device = X.device
    
    input_dim, m = model.W.shape  # m: 隐藏层神经元数量, input_dim: 输入维度
    batch_size = X.shape[0]       # batch_size: 批处理大小
    
        # 初始化 Jacobian 矩阵 J，大小为 (batch_size, m * (input_dim + 1))
    J = torch.zeros(batch_size, m * (input_dim + 1), device=device)
    
    # 计算所有样本的 <w_i, x> 和 ReLU 激活
    relu_input = X @ model.W  # (batch_size, m)
    relu_output = torch.relu(relu_input)  # (batch_size, m)
    
    # 对 w_i 的部分并行计算 Jacobian
    mask = (relu_input > 0).float()  # (batch_size, m)
    
    # 使用广播机制计算 W_grad
    W_grad = (X.unsqueeze(2) * mask.unsqueeze(1)) / m  # (batch_size, input_dim, m)
    W_grad = W_grad * model.a.view(1, 1, m)  # (batch_size, input_dim, m)
    
    # 将 W_grad 转换为二维矩阵并赋值给 J
    J[:, :m*input_dim] = W_grad.reshape(batch_size, -1)
    
    # 对 a_i 的部分并行计算 Jacobian
    J[:, m*input_dim:] = relu_output / m
    
    # end = time.time()
    # print("CUDA计算Jacobian矩阵耗时：", end - start)
    return J

def validate(model, X_train, Y_train, X_test, Y_test, epoch, is_recoard=False, is_SAV=False, ellipsis=None, is_adaptive=False, adp_lr=None):
    with torch.no_grad():
        train_loss = model.loss(model(X_train), Y_train).mean().item()
        test_loss = model.loss(model(X_test), Y_test).mean().item()
        if is_recoard:
            wandb.log({'epoch': epoch + 1,
                       'train_loss': train_loss, 
                       'test_loss': test_loss,
                       'params_norm_W': model.get_norm()[0],
                       'params_norm_a': model.get_norm()[1]})
            if is_SAV:
                wandb.log({'r': model.r.item()})
            if is_adaptive:
                wandb.log({'learning_rate': adp_lr})
            if ellipsis:
                wandb.log({'ellipsis': ellipsis})
        else:
            print(f'Epoch {epoch + 1}: Train Loss: {train_loss}, Test Loss: {test_loss}')



def Adam():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} device')
    model = Simple_Perceptron(21, 10000, 1).to(device)
    train_loader, X_train, Y_train, X_test, Y_test, D = load_data(l=256, name='example_3_20D', device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1)
    for epoch in range(10000):
        for i, (X, Y) in enumerate(train_loader):
            loss = model.loss(model(X), Y).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            train_loss = model.loss(model(X_train), Y_train).mean()
            test_loss = model.loss(model(X_test), Y_test).mean()
            print(f'Epoch: {epoch + 1}, Train Loss: {train_loss.item()}, Test Loss: {test_loss.item()}')


def init_wandb(cfg: dict, title: str, notes=""):
    date = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    wandb.init(project="Numerical Method", config=cfg, name=f"{title}_{date}", notes=notes)


def N_grad(model: Simple_Perceptron, X, Y) -> tuple:
    '''
    手动计算模型的梯度, 返回W,a的梯度
    model: Simple_Perceptron 模型
    X: 输入数据
    Y: 输出数据
    '''
    # start = time.time()
    # 获取模型的参数 a 和 W
    with torch.no_grad():
        a, W, hidden_dim = model.a, model.W, model.W.shape[1]
        y_pred = model(X)
        # 计算损失
        grad_a = torch.zeros_like(a)  # 初始化 grad_a
        z1 = torch.nn.ReLU()(torch.mm(X, W))  # 计算 z1 (激活后的隐藏层输出)
        
        # 计算每个隐藏神经元的梯度
        for j in range(hidden_dim):
            grad_a[j] = (2 * (y_pred - Y.reshape(y_pred.shape)) * z1[:, j]).mean() / hidden_dim
        
        # Step 2: 对 W 的梯度
        grad_W = torch.zeros_like(W)  # 初始化 grad_W
        
        # 计算 ReLU 的梯度
        for j in range(hidden_dim):
            for i in range(X.shape[0]):  # 对每个样本进行求和
                if z1[i, j] > 0:  # ReLU 导数 (只有正值才有梯度)
                    grad_W[:, j] += (2 * (y_pred[i] - Y[i]) * a[j] / hidden_dim) * X[i]
        
        grad_W /= X.shape[0]  # 对样本数 N 进行归一化
    # end = time.time()
    # print("计算梯度耗时：", end - start)
    return grad_W, grad_a

def N_grad_optimized(model, X, Y):
    # start = time.time()
    
    # 获取模型的参数 a 和 W
    with torch.no_grad():
        a, W = model.a, model.W
        hidden_dim = W.shape[1]  # 隐藏层维度
        
        # 前向传播计算预测值和隐藏层激活输出
        z1 = torch.nn.ReLU()(torch.mm(X, W))  # 激活后的隐藏层输出
        y_pred = model(X)  # 模型的输出预测值

        # 计算 grad_a 的梯度
        error = 2 * (y_pred - Y.reshape(y_pred.shape))  # 计算误差
        grad_a = (error.unsqueeze(1) * z1).mean(dim=0) / hidden_dim  # 并行计算 grad_a

        # 计算 grad_W 的梯度
        relu_grad = (z1 > 0).float()  # ReLU 的梯度（非零部分为 1，其他为 0）
        weighted_error = (error.unsqueeze(1) * a) * relu_grad  # 误差与激活梯度和 a 的乘积
        grad_W = torch.mm(X.t(), weighted_error) / (X.shape[0] * hidden_dim)  # 并行计算 grad_W
        
    # end = time.time()
    # print("计算梯度耗时：", end - start)
    
    return grad_W, grad_a