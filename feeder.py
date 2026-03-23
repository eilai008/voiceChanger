import os

import torch
from torch.utils.data import Dataset


class VoiceDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = []
        fi=os.walk(data_path)
        for x in fi:
            for y in x[2]:
                if not y.endswith(".pt"):
                    continue
                self.data.append(os.path.join(x[0], y))
        self.len = len(self.data)
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        return torch.load(self.data[idx],weights_only=True)


def get_dataloader(dataset, config,shuffle):
    return torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle =shuffle)


