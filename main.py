import torch
from modules.Simple_Perceptron import Simple_Perceptron
from data.grip_data import load_data
from config.config import config
import argparse
from utils import init_wandb
from train.train_script import PM_Euler, PM_ESAV, PM_MESAV, PM_SAV, PM_ReSAV, PM_RelSAV, SPM_Euler, SPM_ESAV, SPM_SAV, SPM_ReSAV
from train.train_script import PM_A_Euler, PM_A_SAV, PM_A_ESAV,PM_A_MEAV, PM_A_ReSAV, PM_A_RelSAV, SPM_A_Euler, SPM_A_SAV, SPM_A_ReSAV, SPM_A_ESAV
import torch.multiprocessing as mp

def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")
    
    parser.add_argument("--dataset_name", type=str, default=config["dataset_name"], help="Name of the dataset")
    parser.add_argument("--m", type=int, default=config["m"], help="Model hyperparameter m")
    parser.add_argument("--batch_size", type=int, default=config["batch_size"], help="Batch size")
    parser.add_argument("--epochs", type=int, default=config["epochs"], help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=config["lr"], help="Learning rate")
    parser.add_argument("--C", type=int, default=config["C"], help="SAV parameter C")
    parser.add_argument("--_lambda", type=int, default=config["_lambda"], help="SAV parameter lambda")
    parser.add_argument("--ratio_n", type=float, default=config["ratio_n"], help="RelSAV parameter ratio_n")
    parser.add_argument("--beta_1", type=float, default=config["beta_1"], help="Adaptive parameter beta_1")
    parser.add_argument("--beta_2", type=float, default=config["beta_2"], help="Adaptive parameter beta_2")
    parser.add_argument("--epsilon", type=float, default=config["epsilon"], help="Adaptive parameter epsilon")
    parser.add_argument("--J", type=int, default=config["J"], help="SPM parameter J")
    parser.add_argument("--h", type=float, default=config["h"], help="SPM parameter h")
    parser.add_argument("--recording", type=bool, default=config["recording"], help="Whether to record the training process")
    return parser.parse_args()


def train_process(method, device, lr=None):
    args = parse_args()
    if lr is not None:
        args.lr = lr
    #=========Load Data=========
    train_loader, X_train, Y_train, X_test, Y_test, D = load_data(l=args.batch_size, name=args.dataset_name, device=device)
    model = Simple_Perceptron(D+1, args.m, 1).to(device)
    #=========Model Initialization=========
    if args.recording:
        method_name = method.__name__
        init_wandb(args.__dict__, title=f'{method_name}', notes="Test for SAV 手动更新梯度")
    method(model, train_loader, X_train, Y_train, X_test, Y_test, args)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} device')

    # PM_Euler(model_1, train_loader, X_train, Y_train, X_test, Y_test, args)
    process = []
    process.append(mp.Process(target=train_process, args=(PM_SAV, device)))
    # process.append(mp.Process(target=train_process, args=(PM_ESAV, device, 0.1)))
    # process.append(mp.Process(target=train_process, args=(PM_ESAV, device, 0.01)))
    # process.append(mp.Process(target=train_process, args=(PM_A_ESAV, device, 0.01)))
    # process.append(mp.Process(target=train_process, args=(PM_ESAV, device, 0.001)))
    # process.append(mp.Process(target=train_process, args=(PM_A_ESAV, device, 0.001)))
    # process.append(mp.Process(target=train_process, args=(SPM_ESAV, device)))
    # process.append(mp.Process(target=train_process, args=(SPM_A_ESAV, device)))
    for p in process:
        p.start()

    for p in process:
        p.join()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    # Adam()
    main()
