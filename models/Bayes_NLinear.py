import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_mu=0.0, prior_sigma=1.0):
        super().__init__()
        # Parameters for weight distribution
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(out_features, in_features).fill_(-3.0))
        
        # Parameters for bias distribution
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_log_sigma = nn.Parameter(torch.Tensor(out_features).fill_(-3.0))
        
        # Prior distribution
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

    def forward(self, x):
        # Sample weights and biases using reparameterization
        weight_sigma = torch.exp(self.weight_log_sigma) 
        bias_sigma = torch.exp(self.bias_log_sigma)

        weight_eps = torch.randn_like(weight_sigma)
        bias_eps = torch.randn_like(bias_sigma)

        weight = self.weight_mu + weight_sigma * weight_eps
        bias = self.bias_mu + bias_sigma * bias_eps

        return F.linear(x, weight, bias)

    def kl_loss(self):
        # Compute KL divergence between learned distribution and prior (Gaussian)
        kl_weight = self._kl_divergence(self.weight_mu, self.weight_log_sigma)
        kl_bias = self._kl_divergence(self.bias_mu, self.bias_log_sigma)
        return kl_weight + kl_bias

    def _kl_divergence(self, mu, log_sigma):
        sigma = torch.exp(log_sigma)
        prior_sigma = self.prior_sigma
        prior_mu = self.prior_mu

        kl = (log_sigma.exp() ** 2 + (mu - prior_mu) ** 2) / (2 * prior_sigma ** 2) - 0.5 + log_sigma - torch.log(torch.tensor(prior_sigma))
        return kl.sum()
    
class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.channel
        self.individual = configs.individual

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        
        if self.individual:
            self.Linear = nn.ModuleList([
                BayesianLinear(self.seq_len, self.pred_len) for _ in range(self.channels)
            ])
        else:
            self.Linear = BayesianLinear(self.seq_len, self.pred_len)

    def forward(self, x, y):
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        kl_total = 0
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
                kl_total += self.Linear[i].kl_loss()
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
            kl_total += self.Linear.kl_loss()
        x = x + seq_last
        mse = F.mse_loss(x, y)
        return x, mse, kl_total