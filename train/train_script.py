from optimizers.adaptation import adaptation, anti_adaptation
from optimizers.space_discretization import PM, SPM
from optimizers.time_discretization import Euler, SAV, ReSAV, RelSAV, ESAV, MESAV, RelESAV
from optimizers.time_discretization import IEQ, RelIEQ
from utils import validate, G_modified_CUDA
import torch
from tqdm import tqdm
import warnings

def PM_Euler(model, train_loader, X_train, Y_train, X_test, Y_test, args):
    for epoch in tqdm(range(args.epochs)):
        for x, y in train_loader:
            loss = PM(model, x, y)
            N_a, N_w, lr = anti_adaptation(model, args.lr)
            Euler(model, N_a, N_w, lr) 
        validate(model, X_train, Y_train, X_test, Y_test, epoch, args.recording, False)

def PM_SAV(model, train_loader, X_train, Y_train, X_test, Y_test, args):
    for epoch in tqdm(range(args.epochs)):
        flag = True
        for x, y in train_loader:
            loss = PM(model, x, y)
            if flag:
                model.r = torch.sqrt(loss + args.C)
                flag = False
            N_a, N_w, lr = anti_adaptation(model, args.lr)
            SAV(model, N_a, N_w, lr, loss=loss, C=args.C, _lambda=args._lambda)
        validate(model, X_train, Y_train, X_test, Y_test, epoch, args.recording, True)

def PM_ESAV(model, train_loader, X_train, Y_train, X_test, Y_test, args):
    for epoch in tqdm(range(args.epochs)):
        flag = True
        for x, y in train_loader:
            loss = PM(model, x, y)
            if flag:
                model.r = torch.exp(loss)
                flag = False
            N_a, N_w, lr = anti_adaptation(model, args.lr)
            ESAV(model, N_a, N_w, lr, loss=loss, _lambda=args._lambda)
        validate(model, X_train, Y_train, X_test, Y_test, epoch, args.recording, is_SAV=True)

def PM_ReESAV(model, train_loader, X_train, Y_train, X_test, Y_test, args):
    for epoch in tqdm(range(args.epochs)):
        for x, y in train_loader:
            loss = PM(model, x, y)
            model.r = torch.exp(loss)
            N_a, N_w, lr = anti_adaptation(model, args.lr)
            ESAV(model, N_a, N_w, lr, loss=loss, _lambda=args._lambda)
        validate(model, X_train, Y_train, X_test, Y_test, epoch, args.recording, is_SAV=True)

def PM_MESAV(model, train_loader, X_train, Y_train, X_test, Y_test, args):
    for epoch in tqdm(range(args.epochs)):
        flag = True
        for x, y in train_loader:
            loss = PM(model, x, y)
            if flag:
                model.r = torch.exp(loss)
                E_0 = abs(loss)
                flag = False
            N_a, N_w, lr = anti_adaptation(model, args.lr)
            MESAV(model, N_a, N_w, lr, loss=loss, E_0=E_0, _lambda=args._lambda)
        validate(model, X_train, Y_train, X_test, Y_test, epoch, args.recording, False)

def PM_ReSAV(model, train_loader, X_train, Y_train, X_test, Y_test, args):
    for epoch in tqdm(range(args.epochs)):
        for x, y in train_loader:
            loss = PM(model, x, y)
            N_a, N_w, lr = anti_adaptation(model, args.lr)
            ReSAV(model, N_a, N_w, lr, loss=loss, C=args.C, _lambda=args._lambda)
        validate(model, X_train, Y_train, X_test, Y_test, epoch, args.recording, True)

def PM_RelSAV(model, train_loader, X_train, Y_train, X_test, Y_test, args):
    for epoch in tqdm(range(args.epochs)):
        ellipsis_set = []
        flag = True
        for x, y in train_loader:
            loss = PM(model, x, y)
            if flag:
                model.r = torch.sqrt(loss + args.C)
                flag = False
            N_a, N_w, lr = anti_adaptation(model, args.lr)
            ellipsis_0 = RelSAV(model, N_a, N_w, lr, loss=loss, X=x, Y=y, ratio_n=args.ratio_n, C=args.C, _lambda=args._lambda, )
            ellipsis_set.append(ellipsis_0)
        validate(model, X_train, Y_train, X_test, Y_test, epoch, args.recording, True, ellipsis=sum(ellipsis_set)/len(ellipsis_set))

def PM_RelESAV(model, train_loader, X_train, Y_train, X_test, Y_test, args):
    for epoch in tqdm(range(args.epochs)):
        ellipsis_set = []
        flag = True
        for X, Y in train_loader:
            if flag:
                loss = PM(model, X, Y)
                model.r = torch.exp(loss)
                flag = False
            N_a, N_w, lr = anti_adaptation(model, args.lr)
            ellipsis_0 = RelESAV(model, N_a, N_w, lr, loss=loss, X=X, Y=Y, ratio_n=args.ratio_n, _lambda=args._lambda)
            ellipsis_set.append(ellipsis_0)
        validate(model, X_train, Y_train, X_test, Y_test, epoch, args.recording, True, ellipsis=sum(ellipsis_set)/len(ellipsis_set))
  

def PM_IEQ(model, train_loader, X_train, Y_train, X_test, Y_test, args):
    for epoch in tqdm(range(args.epochs), desc="Training Epochs"):
        flag = True
        for x, y in train_loader:
            if flag:
                flag = False
                model.U = (model.forward(x) - y.reshape(-1, 1))
            J = G_modified_CUDA(x, model)
            IEQ(model, J, args.lr) 
        validate(model, X_train, Y_train, X_test, Y_test, epoch, args.recording, False)       

def PM_ReIEQ(model, train_loader, X_train, Y_train, X_test, Y_test, args):
    for epoch in tqdm(range(args.epochs), desc="Training Epochs"):
        for x, y in train_loader:
            model.U = (model.forward(x) - y.reshape(-1, 1))
            J = G_modified_CUDA(x, model)
            IEQ(model, J, args.lr)
        validate(model, X_train, Y_train, X_test, Y_test, epoch, args.recording, False)

def PM_RelIEQ(model, train_loader, X_train, Y_train, X_test, Y_test, args):
    for epoch in tqdm(range(args.epochs), desc="Training Epochs"):
        ellipsis_set = []
        flag = True
        for x, y in train_loader:
            if flag:
                flag = False
                model.U = (model.forward(x) - y.reshape(-1, 1))
            J = G_modified_CUDA(x, model)
            ellipsis_0 = RelIEQ(model, J, X=x, Y=y, lr=args.lr, ratio_n=args.ratio_n)
            ellipsis_set.append(ellipsis_0)
        validate(model, X_train, Y_train, X_test, Y_test, epoch, args.recording, False, ellipsis=sum(ellipsis_set)/len(ellipsis_set))


def SPM_Euler(model, train_loader, X_train, Y_train, X_test, Y_test, args):
    for epoch in tqdm(range(args.epochs)):
        for x, y in train_loader:
            loss = SPM(model, x, y, J=args.J, h=args.h)
            N_a, N_w, lr = anti_adaptation(model, args.lr)
            Euler(model, N_a, N_w, lr)
        validate(model, X_train, Y_train, X_test, Y_test, epoch, args.recording, False)


def SPM_SAV(model, train_loader, X_train, Y_train, X_test, Y_test, args):
    for epoch in tqdm(range(args.epochs)):
        flag = True
        for x, y in train_loader:
            loss = SPM(model, x, y, J=args.J, h=args.h)
            if flag:
                model.r = torch.sqrt(loss + args.C)
                flag = False
            N_a, N_w, lr = anti_adaptation(model, args.lr)
            SAV(model, N_a, N_w, lr, loss=loss, C=args.C, _lambda=args._lambda)
        validate(model, X_train, Y_train, X_test, Y_test, epoch, args.recording, True)

def SPM_ESAV(model, train_loader, X_train, Y_train, X_test, Y_test, args):
    for epoch in tqdm(range(args.epochs)):
        flag = True
        for x, y in train_loader:
            loss = SPM(model, x, y, J=args.J, h=args.h)
            if flag:
                model.r = torch.exp(loss)
                flag = False
            N_a, N_w, lr = anti_adaptation(model, args.lr)
            ESAV(model, N_a, N_w, lr, loss=loss, _lambda=args._lambda)
        validate(model, X_train, Y_train, X_test, Y_test, epoch, args.recording, False)
        
def SPM_ReSAV(model, train_loader, X_train, Y_train, X_test, Y_test, args):
    for epoch in tqdm(range(args.epochs)):
        for x, y in train_loader:
            loss = SPM(model, x, y, J=args.J, h=args.h)
            N_a, N_w, lr = anti_adaptation(model, args.lr)
            ReSAV(model, N_a, N_w, lr, loss=loss, C=args.C, _lambda=args._lambda)
        validate(model, X_train, Y_train, X_test, Y_test, epoch, args.recording, True)


def SPM_RelSAV(model, train_loader, X_train, Y_train, X_test, Y_test, args):
    for epoch in tqdm(range(args.epochs)):
        ellipsis_set = []
        flag = True
        for x, y in train_loader:
            if flag:
                model.r = torch.sqrt(loss + args.C)
                flag = False
            loss = SPM(model, x, y, J=args.J, h=args.h)
            N_a, N_w, lr = anti_adaptation(model, args.lr)
            ellipsis_0 = RelSAV(model, N_a, N_w, lr, loss=loss, X=x, Y=y, ratio_n=args.ratio_n, C=args.C, _lambda=args._lambda)
            ellipsis_set.append(ellipsis_0)
        validate(model, X_train, Y_train, X_test, Y_test, epoch, args.recording, True, ellipsis=sum(ellipsis_set)/len(ellipsis_set))


def PM_A_Euler(model, train_loader, X_train, Y_train, X_test, Y_test, args):
    for epoch in tqdm(range(args.epochs)):
        cnt = 0
        lr_set = []
        for x, y in train_loader:
            loss = PM(model, x, y)
            N_a, N_w, adp_lr = adaptation(model, args.lr, cnt, epsilon=args.epsilon, beta_1=args.beta_1, beta_2=args.beta_2)
            lr_set.append(adp_lr)
            Euler(model, N_a, N_w, adp_lr)
            cnt += 1
        validate(model, X_train, Y_train, X_test, Y_test, epoch, 
                 is_recoard=args.recording, is_SAV=False, 
                 is_adaptive=True, adp_lr=sum(lr_set)/len(lr_set))


def PM_A_SAV(model, train_loader, X_train, Y_train, X_test, Y_test, args):
    for epoch in tqdm(range(args.epochs)):
        cnt = 0
        lr_set = []
        flag = True
        for x, y in train_loader:
            loss = PM(model, x, y)
            if flag:
                model.r = torch.sqrt(loss + args.C)
                flag = False
            N_a, N_w, adp_lr = adaptation(model, args.lr, cnt, epsilon=args.epsilon, beta_1=args.beta_1, beta_2=args.beta_2)
            lr_set.append(adp_lr)
            SAV(model, N_a, N_w, adp_lr, loss=loss, C=args.C, _lambda=args._lambda)
            cnt += 1
        validate(model, X_train, Y_train, X_test, Y_test, epoch, 
                 is_recoard=args.recording, is_SAV=True, 
                 is_adaptive=True, adp_lr=sum(lr_set)/len(lr_set))

def PM_A_ESAV(model, train_loader, X_train, Y_train, X_test, Y_test, args):
    for epoch in tqdm(range(args.epochs)):
        cnt = 0
        lr_set = []
        flag = True
        for x, y in train_loader:
            loss = PM(model, x, y)
            if flag:
                model.r = torch.exp(loss)
                flag = False
            N_a, N_w, adp_lr = adaptation(model, args.lr, cnt, epsilon=args.epsilon, beta_1=args.beta_1, beta_2=args.beta_2)
            lr_set.append(adp_lr)
            ESAV(model, N_a, N_w, adp_lr, loss=loss, _lambda=args._lambda)
            cnt += 1
        validate(model, X_train, Y_train, X_test, Y_test, epoch, 
                 is_recoard=args.recording, is_SAV=False, 
                 is_adaptive=True, adp_lr=sum(lr_set)/len(lr_set))       

def PM_A_ReSAV(model, train_loader, X_train, Y_train, X_test, Y_test, args):
    for epoch in tqdm(range(args.epochs)):
        cnt = 0
        lr_set = []
        for x, y in train_loader:
            loss = PM(model, x, y)
            N_a, N_w, adp_lr = adaptation(model, args.lr, cnt, epsilon=args.epsilon, beta_1=args.beta_1, beta_2=args.beta_2)
            lr_set.append(adp_lr)
            ReSAV(model, N_a, N_w, adp_lr, loss=loss, C=args.C, _lambda=args._lambda)
            cnt += 1
        validate(model, X_train, Y_train, X_test, Y_test, epoch, 
                 is_recoard=args.recording, is_SAV=True, 
                 is_adaptive=True, adp_lr=sum(lr_set)/len(lr_set))

def PM_A_MEAV(model, train_loader, X_train, Y_train, X_test, Y_test, args):
    for epoch in tqdm(range(args.epochs)):
        cnt = 0
        lr_set = []
        flag = True
        for x, y in train_loader:
            loss = PM(model, x, y)
            if flag:
                model.r = torch.exp(loss)
                E_0 = abs(loss)
                flag = False
            N_a, N_w, adp_lr = adaptation(model, args.lr, cnt, epsilon=args.epsilon, beta_1=args.beta_1, beta_2=args.beta_2)
            lr_set.append(adp_lr)
            MESAV(model, N_a, N_w, adp_lr, loss=loss, E_0=E_0, _lambda=args._lambda)
            cnt += 1
        validate(model, X_train, Y_train, X_test, Y_test, epoch, 
                 is_recoard=args.recording, is_SAV=True, 
                 is_adaptive=True, adp_lr=sum(lr_set)/len(lr_set))


def PM_A_RelSAV(model, train_loader, X_train, Y_train, X_test, Y_test, args):
    for epoch in tqdm(range(args.epochs)):
        cnt = 0
        lr_set = []
        ellipsis_set = []
        flag = True
        for x, y in train_loader:
            loss = PM(model, x, y)
            if flag:
                model.r = torch.sqrt(loss + args.C)
                flag = False
            N_a, N_w, adp_lr = adaptation(model, args.lr, cnt, epsilon=args.epsilon, beta_1=args.beta_1, beta_2=args.beta_2)
            lr_set.append(adp_lr)
            ellipsis_0 = RelSAV(model, N_a, N_w, adp_lr, loss=loss, X=x, Y=y, ratio_n=args.ratio_n, C=args.C, _lambda=args._lambda)
            ellipsis_set.append(ellipsis_0)
            cnt += 1
        validate(model, X_train, Y_train, X_test, Y_test, epoch, 
                 is_recoard=args.recording, is_SAV=True, 
                 is_adaptive=True, adp_lr=sum(lr_set)/len(lr_set), ellipsis=sum(ellipsis_set)/len(ellipsis_set))


def SPM_A_Euler(model, train_loader, X_train, Y_train, X_test, Y_test, args):
    for epoch in tqdm(range(args.epochs)):
        cnt = 0
        lr_set = []
        for x, y in train_loader:
            loss = SPM(model, x, y, J=args.J, h=args.h)
            N_a, N_w, adp_lr = adaptation(model, args.lr, cnt, epsilon=args.epsilon, beta_1=args.beta_1, beta_2=args.beta_2)
            lr_set.append(adp_lr)
            Euler(model, N_a, N_w, adp_lr)
            cnt += 1
        validate(model, X_train, Y_train, X_test, Y_test, epoch, 
                 is_recoard=args.recording, is_SAV=False, 
                 is_adaptive=True, adp_lr=sum(lr_set)/len(lr_set))
        

def SPM_A_SAV(model, train_loader, X_train, Y_train, X_test, Y_test, args):
    for epoch in tqdm(range(args.epochs)):
        cnt = 0
        lr_set = []
        flag = True
        for x, y in train_loader:
            loss = SPM(model, x, y, J=args.J, h=args.h)
            if flag:
                model.r = torch.sqrt(loss + args.C)
                flag = False
            N_a, N_w, adp_lr = adaptation(model, args.lr, cnt, epsilon=args.epsilon, beta_1=args.beta_1, beta_2=args.beta_2)
            lr_set.append(adp_lr)
            SAV(model, N_a, N_w, adp_lr, loss=loss, C=args.C, _lambda=args._lambda)
            cnt += 1
        validate(model, X_train, Y_train, X_test, Y_test, epoch, 
                 is_recoard=args.recording, is_SAV=True, 
                 is_adaptive=True, adp_lr=sum(lr_set)/len(lr_set))

def SPM_A_ESAV(model, train_loader, X_train, Y_train, X_test, Y_test, args):
    for epoch in tqdm(range(args.epochs)):
        cnt = 0
        lr_set = []
        flag = True
        for x, y in train_loader:
            loss = SPM(model, x, y, J=args.J, h=args.h)
            if flag:
                model.r = torch.exp(loss)
                flag = False
            N_a, N_w, adp_lr = adaptation(model, args.lr, cnt, epsilon=args.epsilon, beta_1=args.beta_1, beta_2=args.beta_2)
            lr_set.append(adp_lr)
            ESAV(model, N_a, N_w, adp_lr, loss=loss, _lambda=args._lambda)
            cnt += 1
        validate(model, X_train, Y_train, X_test, Y_test, epoch, 
                 is_recoard=args.recording, is_SAV=False, 
                 is_adaptive=True, adp_lr=sum(lr_set)/len(lr_set))


def SPM_A_ReSAV(model, train_loader, X_train, Y_train, X_test, Y_test, args):
    for epoch in tqdm(range(args.epochs)):
        cnt = 0
        lr_set = []
        for x, y in train_loader:
            loss = SPM(model, x, y, J=args.J, h=args.h)
            N_a, N_w, adp_lr = adaptation(model, args.lr, cnt, epsilon=args.epsilon, beta_1=args.beta_1, beta_2=args.beta_2)
            lr_set.append(adp_lr)
            ReSAV(model, N_a, N_w, adp_lr, loss=loss, C=args.C, _lambda=args._lambda)
            cnt += 1
        validate(model, X_train, Y_train, X_test, Y_test, epoch, 
                 is_recoard=args.recording, is_SAV=True, 
                 is_adaptive=True, adp_lr=sum(lr_set)/len(lr_set))
        

def SPM_A_RelSAV(model, train_loader, X_train, Y_train, X_test, Y_test, args):
    for epoch in tqdm(range(args.epochs)):
        cnt = 0
        lr_set = []
        ellipsis_set = []
        flag = True
        for x, y in train_loader:
            loss = SPM(model, x, y, J=args.J, h=args.h)
            if flag:
                model.r = torch.sqrt(loss + args.C)
                flag = False
            N_a, N_w, adp_lr = adaptation(model, args.lr, cnt, epsilon=args.epsilon, beta_1=args.beta_1, beta_2=args.beta_2)
            lr_set.append(adp_lr)
            ellipsis_0 = RelSAV(model, N_a, N_w, adp_lr, loss=loss, X=x, Y=y, ratio_n=args.ratio_n, C=args.C, _lambda=args._lambda)
            ellipsis_set.append(ellipsis_0)
            cnt += 1
        validate(model, X_train, Y_train, X_test, Y_test, epoch, 
                 is_recoard=args.recording, is_SAV=True, 
                 is_adaptive=True, adp_lr=sum(lr_set)/len(lr_set), ellipsis=sum(ellipsis_set)/len(ellipsis_set))
