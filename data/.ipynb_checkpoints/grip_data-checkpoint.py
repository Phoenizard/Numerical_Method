import torch
from torch.utils.data import DataLoader, TensorDataset


path_dict = {
    'example_2_40D': ['./data/dataset_2_40D.pt', 40],
    'example_1_40D': ['./data/dataset_1_40D.pt', 40],
    'example_1_20D': ['./data/dataset_1_20D.pt', 20],
    'example_3_40D': ['./data/dataset_3_40D.pt', 40],
    'example_3_20D': ['./data/dataset_3_20D.pt', 20]
}


def load_data(l=64, name='example_2_40D', device='cuda'):
    path, D = path_dict[name]
    X_train, Y_train, X_test, Y_test = torch.load(path, weights_only=True)    
    # 将数据移动到适当的设备
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)
    # 使用 DataLoader 进行批处理
    l = 64
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=l, shuffle=True)
    return train_loader, X_train, Y_train, X_test, Y_test, D