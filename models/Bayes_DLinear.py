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
    
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    DLinear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.channel

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(BayesianLinear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(BayesianLinear(self.seq_len,self.pred_len))
        else:
            self.Linear_Seasonal = BayesianLinear(self.seq_len,self.pred_len)
            self.Linear_Trend = BayesianLinear(self.seq_len,self.pred_len)

    def forward(self, x, y):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        kl_total=0
        if self.individual:
            seasonal_output = torch.zeros([x.size(0), self.channels, self.pred_len], dtype=x.dtype, device=x.device)
            trend_output = torch.zeros_like(seasonal_output)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
                kl_total += self.Linear_Seasonal[i].kl_loss()
                kl_total += self.Linear_Trend[i].kl_loss()
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)  # [B, D, P]
            trend_output = self.Linear_Trend(trend_init)
            kl_total += self.Linear_Seasonal.kl_loss()
            kl_total += self.Linear_Trend.kl_loss()
        
        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1)
        mse = F.mse_loss(x, y)
        return x, mse, kl_total # to [Batch, Output length, Channel]