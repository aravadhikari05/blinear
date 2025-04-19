import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import argparse
from tqdm import tqdm
from models import DLinear, NLinear, RLinear, Bayes_DLinear, Bayes_NLinear, Bayes_RLinear  # make sure these are in your path
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom  # your loaders
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_and_evaluate(ModelClass, configs, dataset_class, device):
    results = []
    for run in range(3): 
        set_seed(42 + run)

        train_dataset = dataset_class(
            root_path=configs.root_path,
            flag='train',
            size=(configs.seq_len, configs.pred_len),
            features='M',
            data_path=configs.data_path,
            target=configs.target,
            scale=True,
            timeenc=0
        )
        val_dataset = dataset_class(
            root_path=configs.root_path,
            flag='val',
            size=(configs.seq_len, configs.pred_len),
            features='M',
            data_path=configs.data_path,
            target=configs.target,
            scale=True,
            timeenc=0
        )
        train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False)

        # init model
        model = ModelClass(configs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)

        # train
        for epoch in range(configs.epochs):
            model.train()
            for batch in train_loader:
                batch_x, batch_y, _, _ = [b.float().to(device) for b in batch]
                outputs = model(batch_x, batch_y)
                if isinstance(outputs, tuple):
                    pred, mse, *kl = outputs
                    loss = mse + configs.beta * kl[0] if kl else mse
                else:
                    pred, loss = outputs
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            

        # eval
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch_x, batch_y, _, _ = [b.float().to(device) for b in batch]
                outputs = model(batch_x, batch_y)
                if isinstance(outputs, tuple):
                    pred, mse, *_ = outputs
                else:
                    pred, mse = outputs
                val_losses.append(mse.item())
                
        results.append(np.mean(val_losses))

    mean = np.mean(results)
    std = np.std(results)
    return mean, std

# hyperparameters
class Configs:
    def __init__(self, model_name, data_path, root_path, dataset_class, pred_len, channel):
        self.model_name = model_name
        self.data_path = data_path
        self.root_path = root_path
        self.dataset_class = dataset_class

        self.target = 'OT'
        self.seq_len = pred_len
        self.pred_len = pred_len
        self.channel = channel
        self.batch_size = 32
        self.lr = 0.005
        self.epochs = 15
        self.beta = 1e-3  
        self.drop = 0.2
        self.individual = False
        self.rev = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# map models and datasets
model_map = {
    "DLinear": DLinear.Model,
    "NLinear": NLinear.Model,
    "RLinear": RLinear.Model,
    "Bayes_DLinear": Bayes_DLinear.Model,
    "Bayes_NLinear": Bayes_NLinear.Model,
    "Bayes_RLinear": Bayes_RLinear.Model
}

dataset_map = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Electricity': Dataset_Custom,
    'Weather': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Exchange_Rate': Dataset_Custom
}

channel_map = {
    'ETTh1': 7, 'ETTh2': 7, 'ETTm1': 7, 'ETTm2': 7,
    'Electricity': 321, 'Weather': 21, 'Traffic': 862, 'Exchange_Rate': 8
}


if __name__ == "__main__":
    results = []
    #datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Electricity', 'Weather', 'Traffic', 'Exchange_Rate']
    datasets = ['ETTh1']
    model_names = ['DLinear', 'NLinear', 'RLinear', 'Bayes_DLinear', 'Bayes_NLinear', 'Bayes_RLinear']
    pred_lens = [96]
    #pred_lens = [96, 192, 336, 720]

    for dataset_name in datasets:
        for pred_len in pred_lens:
            data_file = dataset_name + ".csv"
            dataset_class = dataset_map[dataset_name]
            channel = channel_map[dataset_name]

            row_result = {
                "Dataset": dataset_name,
                "Pred_len": pred_len
            }

            for model_name in model_names:
                print(f"Running {model_name} on {dataset_name} with pred_len={pred_len}...")
                config = Configs(
                    model_name=model_name,
                    data_path=data_file,
                    root_path='./dataset/',
                    dataset_class=dataset_class,
                    pred_len=pred_len,
                    channel=channel
                )

                model_class = model_map[model_name]
                mean, std = train_and_evaluate(model_class, config, dataset_class, config.device)
                result_str = f"{mean:.3f} ± {std:.4f}"
                row_result[model_name] = result_str
                print(f"> {dataset_name} - {model_name} | pred_len={pred_len} | MSE: {result_str}")

            results.append(row_result)

import pandas as pd

df = pd.DataFrame(results)

# Sort by Dataset and Pred_len
df = df.sort_values(by=["Dataset", "Pred_len"])

df.reset_index(drop=True, inplace=True)

lines = []
lines.append(r"\begin{tabular}{|l|l|" + "c|" * len(model_names) + "}")
lines.append(r"\hline")
lines.append("Dataset & Pred\_len & " + " & ".join(model_names) + r" \\")
lines.append(r"\hline")

last_dataset = None
for i, row in df.iterrows():
    dataset_cell = row["Dataset"] if row["Dataset"] != last_dataset else ""
    if row["Dataset"] != last_dataset and last_dataset is not None:
        lines.append(r"\hline") 
    last_dataset = row["Dataset"]

    row_values = [dataset_cell, str(row["Pred_len"])] + [row[m] for m in model_names]
    lines.append(" & ".join(row_values) + r" \\")
lines.append(r"\hline")
lines.append(r"\end{tabular}")
latex_output = "\n".join(lines)
print(latex_output)

