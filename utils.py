import torch, wandb, datetime
from modules.Simple_Perceptron import Simple_Perceptron
from data.grip_data import load_data

def G_modified(X, model):
    # 开始计时
    # start = time.time()
    
    input_dim, m = model.W.shape  # m: 隐藏层神经元数量, input_dim: 输入维度
    batch_size = X.shape[0]       # batch_size: 批处理大小
    
    # 初始化 Jacobian 矩阵 J，大小为 (batch_size, m * (input_dim + 1))
    J = torch.zeros(batch_size, m * (input_dim + 1), device=X.device)
    
    # 计算所有样本的 <w_i, x> 和 ReLU 激活
    relu_input = X @ model.W  # (batch_size, m)
    relu_output = torch.relu(relu_input)  # (batch_size, m)
    
    # 对 w_i 的部分并行计算 Jacobian
    for j in range(m):
        mask = relu_output[:, j] > 0  # 只选择 ReLU 激活大于0的元素
        J[:, j*input_dim:(j+1)*input_dim] = (model.a[j] * X * mask.view(-1, 1)) / m
    # TODO: model.a[j] * X * mask.view(-1, 1) or model.a[j] * X
    # 对 a_i 的部分并行计算 Jacobian
    J[:, m*input_dim:] = relu_output / m

    # # 结束计时
    # end = time.time()
    # print("优化后Time: ", end - start)
    
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
