import torch
from torch import optim, nn

import architecture
import feeder
import preprocessing
import json
import os

def setup(config):
    G_AtoB = architecture.Generator(config)
    G_BtoA = architecture.Generator(config)
    D_A = architecture.Discriminator(config)
    D_B = architecture.Discriminator(config)
    optimizer = optim.Adam(list(G_AtoB.parameters()) + list(G_BtoA.parameters()), lr=config['lr'],betas=(config['beta1'], config['beta2']))
    optimizer2 = optim.Adam(list(D_A.parameters()) + list(D_B.parameters()), lr=config['lr'],betas=(config['beta1'], config['beta2']))
    dataset_a = feeder.VoiceDataset(os.path.join(os.path.dirname(__file__), "data/processed", config['source_path']))
    loader_a = feeder.get_dataloader(dataset_a, config)
    dataset_b = feeder.VoiceDataset(os.path.join(os.path.dirname(__file__), "data/processed", config['target_path']))
    loader_b = feeder.get_dataloader(dataset_b, config)
    return G_AtoB, G_BtoA, D_A, D_B, optimizer, optimizer2, loader_a, loader_b

def train_epoch(G_AtoB, G_BtoA, D_A, D_B, opt_G, opt_D, loader_a, loader_b, config):
    criterion = nn.MSELoss()
    l1_loss = nn.L1Loss()
    total_G_loss = 0
    total_D_loss = 0
    for real_a, real_b in zip(loader_a, loader_b):
        opt_D.zero_grad()
        fake_b = G_AtoB(real_a)
        fake_a = G_BtoA(real_b)
        pred_real_A = D_A(real_a)
        pred_A = D_A(fake_a)
        pred_real_B = D_B(real_b)
        pred_B = D_B(fake_b)
        loss_D_A = criterion(pred_real_A, torch.ones_like(pred_real_A)) + \
                   criterion(pred_A, torch.zeros_like(pred_A))
        loss_D_B = criterion(pred_B, torch.zeros_like(pred_B)) + \
                   criterion(pred_real_B, torch.ones_like(pred_real_B))
        loss_D = loss_D_A + loss_D_B
        total_D_loss += loss_D.item()
        loss_D.backward()
        opt_D.step()

        opt_G.zero_grad()

        loss_adv_G = criterion(D_B(fake_b), torch.ones_like(D_B(fake_b))) + \
                     criterion(D_A(fake_a), torch.ones_like(D_A(fake_a)))


        cycle_a = G_BtoA(fake_b)
        cycle_b = G_AtoB(fake_a)
        l1 = l1_loss(cycle_b, real_b) * config['lambda_cycle']
        l2 = l1_loss(cycle_a, real_a) * config['lambda_cycle']
        lose_cycle = l1 + l2


        identity_a = G_BtoA(real_a)
        identity_b = G_AtoB(real_b)
        l3 = l1_loss(identity_a, real_a) * config['lambda_identity']
        l4 = l1_loss(identity_b, real_b) * config['lambda_identity']
        lose_identity = l3 + l4

        loss_G =  lose_cycle +  lose_identity + loss_adv_G
        total_G_loss += loss_G.item()
        loss_G.backward()
        opt_G.step()
    return total_G_loss / len(loader_a), total_D_loss / len(loader_a)

def train():
    config = preprocessing.load_config()
    G_AtoB, G_BtoA, D_A, D_B, opt_G, opt_D, loader_a, loader_b = setup(config)

    for epoch in range(config['epochs']):
        loses =train_epoch(G_AtoB, G_BtoA, D_A, D_B, opt_G, opt_D, loader_a, loader_b, config)
        print(f"Epoch {epoch + 1}/{config['epochs']} complete")
        print(f"lose_g = {loses[0]} | lose_d = {loses[1]}")
        if (epoch+1)%10==0:
            torch.save(G_AtoB.state_dict(), os.path.join(os.path.dirname(__file__), 'models/saved_models',config['model_name'],f'G_AtoB({epoch+1}).pth'))
            torch.save(G_BtoA.state_dict(),os.path.join(os.path.dirname(__file__), 'models/saved_models', config['model_name'],f'G_BtoA({epoch+1}).pth'))
            torch.save(D_A.state_dict(),os.path.join(os.path.dirname(__file__), 'models/saved_models', config['model_name'],f'D_A({epoch+1}).pth'))
            torch.save(D_B.state_dict(),os.path.join(os.path.dirname(__file__), 'models/saved_models', config['model_name'],f'D_B({epoch+1}).pth'))



if __name__ == "__main__":
    train()