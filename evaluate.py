import os

import torch
from torch import nn

import architecture
import preprocessing
import matplotlib.pyplot as plt


def load_model(config,model_name,epoch):
    G_AtoB_model = architecture.Generator(config)
    G_BtoA_model = architecture.Generator(config)

    G_AtoB = torch.load(os.path.join(os.path.dirname(__file__), 'models/saved_models',model_name,f'G_AtoB_{epoch}.pth'),weights_only=True)
    G_BtoA = torch.load(os.path.join(os.path.dirname(__file__), 'models/saved_models', model_name, f'G_BtoA_{epoch}.pth'),weights_only=True)

    G_AtoB_model.load_state_dict(G_AtoB)
    G_BtoA_model.load_state_dict(G_BtoA)

    return G_AtoB_model, G_BtoA_model

def evaluate(val_loader_a,val_loader_b,config,model_name,epoch):
    model_AtoB, model_BtoA = load_model(config,model_name,epoch)
    l1_loss = nn.L1Loss()
    model_AtoB.eval()
    model_BtoA.eval()
    loss_tot = 0
    with torch.no_grad():
        for real_a, real_b in zip(val_loader_a, val_loader_b):
            fake_b = model_AtoB(real_a)
            cycle = model_BtoA(fake_b)
            loss = l1_loss(cycle, real_a) * config['lambda_cycle']
            loss_tot += loss.item()
    return loss_tot / len(val_loader_a)

def plot_losses(g_losses, d_losses, val_losses,model_name):
    plt.plot(g_losses, label='Generator')
    plt.plot(d_losses, label='Discriminator')
    val_epochs = list(range(10, len(val_losses)*10+1, 10))
    plt.plot(val_epochs, val_losses, label='Validation')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Losses - {model_name}')

    plt.legend()
    save_dir = os.path.join(os.path.dirname(__file__), 'models/graphs/')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'models/graphs/losses_{model_name}.png')
    plt.show()

if __name__ == "__main__":
    config = preprocessing.load_config()
    G_AtoB,G_BtoA = load_model(config,"model1",10)