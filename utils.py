import torch, wandb, datetime

def validate(model, X_train, Y_train, X_test, Y_test, epoch, is_recoard=False, is_SAV=False, ellipsis=None, is_adaptive=False, adp_lr=None):
    with torch.no_grad():
        train_loss = model.loss(model(X_train), Y_train).mean().item()
        test_loss = model.loss(model(X_test), Y_test).mean().item()
        if is_recoard:
            wandb.log({'train_loss': train_loss, 
                       'test_loss': test_loss,
                       'params_norm': model.get_norm()})
            if is_SAV:
                wandb.log({'r': model.r.item()})
            if is_adaptive:
                wandb.log({'learning_rate': adp_lr})
            if ellipsis:
                wandb.log({'ellipsis': ellipsis})
        print(f'Epoch {epoch}: Train Loss: {train_loss}, Test Loss: {test_loss}')


def init_wandb(cfg: dict, title: str, notes=""):
    date = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    wandb.init(project="Numerical Method", config=cfg, name=f"{title}_{date}", notes=notes)
