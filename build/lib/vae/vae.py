import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
from tqdm import trange
import numpy as np


class Decoder(nn.Module):
    def __init__(self, input_dimension, hidden_dimension,latent_dimension):
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(latent_dimension, hidden_dimension),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dimension),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dimension),
            torch.nn.Linear(hidden_dimension, input_dimension)
        )

    def forward(self, z):
        x_hat = self.model(z)
        return x_hat



class Encoder(nn.Module):
    def __init__(self, input_dimension, latent_dimension,
                 hidden_dimension,normalize):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, hidden_dimension),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dimension),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dimension),
            torch.nn.Linear(hidden_dimension, latent_dimension)
        )
        self.var_est = torch.nn.Sequential(
            torch.nn.Linear(latent_dimension, latent_dimension),
            nn.ReLU(),
            nn.BatchNorm1d(latent_dimension),
            torch.nn.Linear(latent_dimension, latent_dimension),
            nn.ReLU(),
            nn.BatchNorm1d(latent_dimension),
            torch.nn.Linear(latent_dimension, latent_dimension)
        )
        self.bn=nn.BatchNorm1d(latent_dimension,affine=False,track_running_stats=False)
        self.normalize=normalize
    def forward(self, x):   
        mu=self.model(x)
        if self.normalize:
            mu=self.bn(mu)
        log_sigma=self.var_est(mu)
        return mu,log_sigma


class VAE(nn.Module):

    def __init__(self, input_dimension, hidden_dimension,latent_dimension,lr,normalize=False):
        super().__init__()
        self.log_scale=nn.Parameter(torch.Tensor([0.0]))
        self.latent_dimension=latent_dimension
        self.decoder = Decoder(input_dimension=input_dimension,
                                    latent_dimension=latent_dimension,hidden_dimension=hidden_dimension)

        self.encoder = Encoder(input_dimension=input_dimension,
                                    latent_dimension=latent_dimension,hidden_dimension=hidden_dimension,normalize=normalize)

        self.opt=torch.optim.AdamW(self.parameters(), lr=lr)

    def sample(self, ):
        return self.decoder(torch.randn(1,self.latent_dim))

    def compute_latent(self, x):
        x=torch.tensor(x).float()
        self.encoder=self.encoder.eval()
        return np.array(self.encoder(x)[0].detach())



    def train(self, dataloader, epochs=2000):
        for epoch in trange(epochs):
            for data in dataloader:
                self.opt.zero_grad()
                x,=data
                mu,sigma=self.encoder(x)
                q_1 = torch.distributions.Normal(mu, torch.exp(sigma))
                standard_1=torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
                z1_sampled = q_1.rsample()
                x_hat = self.decoder(z1_sampled)
                p_1=torch.distributions.Normal(x_hat,torch.exp(self.log_scale))
                reconstruction=p_1.log_prob(x).sum(dim=1)   
                reg=torch.distributions.kl_divergence(q_1, standard_1).sum(dim=1)
                elbo=(reconstruction-reg).mean(dim=0)
                loss=-elbo
                loss.backward()
                self.opt.step()
        print("hello")        


    def fit(self,value):
        h=torch.tensor(value).float()
        train = torch.utils.data.TensorDataset(h)
        train_loader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
        self.train(train_loader)    

    def reconstruct(self,z):
        z=torch.tensor(z).float()
        self.decoder=self.decoder.eval() 
        return np.array(self.decoder(z).detach())

    def generate_new_dataset(self,NUM_SAMPLES):
        z=torch.randn(NUM_SAMPLES,self.latent_dimension)
        values=self.reconstruct(z)
        return np.array(z),values



