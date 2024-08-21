import torch, wandb, datetime
from modules.Simple_Perceptron import Simple_Perceptron
from data.grip_data import load_data

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
