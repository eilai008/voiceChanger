import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm1d(channels)


    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return out+x

class Generator(nn.Module):
    def __init__(self,config):
        super(Generator, self).__init__()
        self.channels = config['hidden_size']
        mel = config['n_mel_bands']
        ch=self.channels
        self.model = nn.Sequential(
            nn.Conv1d(mel, ch, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(ch),
            nn.ReLU(),
            nn.Conv1d(ch, ch*2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(ch*2),
            nn.ReLU(),
            nn.Conv1d(ch*2, ch*4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(ch *4),
            nn.ReLU(),
            nn.Conv1d(ch*4, ch*4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(ch * 4),
            nn.ReLU(),
            ResidualBlock(ch*4),
            ResidualBlock(ch*4),
            ResidualBlock(ch * 4),
            ResidualBlock(ch * 4),
            ResidualBlock(ch * 4),
            ResidualBlock(ch * 4),
            nn.ConvTranspose1d(ch*4,ch*4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(ch * 4),
            nn.ReLU(),
            nn.ConvTranspose1d(ch*4, ch*2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(ch * 2),
            nn.ReLU(),
            nn.ConvTranspose1d(ch*2, ch, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(ch),
            nn.ReLU(),
            nn.ConvTranspose1d(ch, mel, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self,config):
        super(Discriminator, self).__init__()
        self.channels = config['hidden_size']
        mel = config['n_mel_bands']
        ch=self.channels
        self.model = nn.Sequential(
            nn.Conv1d(mel, ch, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(ch, ch*2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(ch*2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(ch*2, ch*4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(ch*4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(ch * 4, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)