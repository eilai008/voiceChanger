from torch import optim

import architecture
import feeder
import preprocessing
import json
import os
def setup():
    config = preprocessing.load_config()
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