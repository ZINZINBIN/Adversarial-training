import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

class AutoEncoder(nn.Module):
    def __init__(self, 
    layer_filters : List[int] = [32,64], 
    input_shape : Tuple[int,int,int] = (1,28,28),
    kernel_size : int = 3,
    latent_dims : int = 16
    ):
        super(AutoEncoder, self).__init__()
        self.layer_filters = layer_filters
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.latent_dims = latent_dims

        self.encoder_head = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels=layer_filters[0], kernel_size = 3, stride=3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,stride = 2),
            nn.Conv2d(in_channels=layer_filters[0], out_channels=layer_filters[1], kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 1)
        )

        self.decoder_shape = self.get_decoder_input()
        self.encoder_tail = nn.Linear(self.decoder_shape[1] * self.decoder_shape[2] * self.decoder_shape[3], latent_dims)

        self.decoder_head = nn.Sequential(
            nn.Linear(latent_dims, self.decoder_shape[1] * self.decoder_shape[2] * self.decoder_shape[3]),
            nn.ReLU()
        )

        self.decoder_tail = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.decoder_shape[1],
                out_channels=layer_filters[1],
                kernel_size=3,
                stride=2
                ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=layer_filters[1],
                out_channels=layer_filters[0],
                kernel_size=5,
                stride=3,
                padding = 1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=layer_filters[0],
                out_channels=1,
                kernel_size=2,
                stride = 2,
                padding = 1
            ),
            nn.Sigmoid()
        )

    def get_decoder_input(self):
        input_shape = (1,) + self.input_shape
        with torch.no_grad():
            x = torch.zeros(input_shape).to(next(self.parameters()).device)
            x = self.encoder_head(x)
        return x.size()

    def forward(self, x):
        x = self.encoder_head(x)
        x = self.encoder_tail(x.view(x.size(0), -1))
        x = self.decoder_head(x)
        x = self.decoder_tail(x.view(x.size(0), self.decoder_shape[1], self.decoder_shape[2], self.decoder_shape[3]))
        return x

    def encode(self, x):
        x = self.encoder_head(x)
        x = self.encoder_tail(x.view(x.size(0), -1))
        return x


class VAE(nn.Module):
    def __init__(
        self, 
        layer_filters : List[int] = [32,64], 
        input_shape : Tuple[int,int,int] = (1,28,28),
        kernel_size : int = 3,
        latent_dims : int = 16
    ):
        super(VAE, self).__init__()
        self.layer_filters = layer_filters
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.latent_dims = latent_dims

        self.encoder_head = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels=layer_filters[0], kernel_size = 3, stride=3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,stride = 2),
            nn.Conv2d(in_channels=layer_filters[0], out_channels=layer_filters[1], kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 1)
        )

        self.decoder_shape = self.get_decoder_input()
        self.encoder_tail = nn.Linear(self.decoder_shape[1] * self.decoder_shape[2] * self.decoder_shape[3], latent_dims)

        self.decoder_head = nn.Sequential(
            nn.Linear(latent_dims, self.decoder_shape[1] * self.decoder_shape[2] * self.decoder_shape[3]),
            nn.ReLU()
        )

        self.decoder_tail = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.decoder_shape[1],
                out_channels=layer_filters[1],
                kernel_size=3,
                stride=2
                ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=layer_filters[1],
                out_channels=layer_filters[0],
                kernel_size=5,
                stride=3,
                padding = 1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=layer_filters[0],
                out_channels=1,
                kernel_size=2,
                stride = 2,
                padding = 1
            ),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar, device = 'cpu'):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size(), device = device).normal_()
        eps = torch.autograd.Variable(eps)
        return eps.mul(std).add_(mu)

    def get_decoder_input(self):
        input_shape = (1,) + self.input_shape
        with torch.no_grad():
            x = torch.zeros(input_shape).to(next(self.parameters()).device)
            x = self.encoder_head(x)
        return x.size()

    def forward(self, x):
        x = self.encoder_head(x)
        x = self.encoder_tail(x.view(x.size(0), -1))
        x = self.decoder_head(x)
        x = self.decoder_tail(x.view(x.size(0), self.decoder_shape[1], self.decoder_shape[2], self.decoder_shape[3]))
        return x

    def encode(self, x):
        x = self.encoder_head(x)
        x = self.encoder_tail(x.view(x.size(0), -1))
        return x

    def decode(self, z):
        x = self.decode_head(z)
        z = self.decode_tail(x.view(x.size(0), self.decoder_shape[1], self.decoder_shape[2], self.decoder_shape[3]))
        return z