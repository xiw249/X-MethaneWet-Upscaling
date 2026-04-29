import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
import argparse
from torch.utils.data import DataLoader, Dataset
import math
import json
import logging
from sklearn.metrics import mean_squared_error, r2_score

from config import Config
from model import LSTM, MyEALSTM, MultiTCN, Transformer, iTransformer, Pyraformer, HybridCNNLSTM

class Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
class TemporalDataset(Dataset):
    def __init__(self, years):
        self.years = years
        self.input_files = [f"../../processed_data/TEM-MDM/temporal/input_{year}.npy" for year in years]
        self.output_files = [f"../../processed_data/TEM-MDM/temporal/output_{year}.npy" for year in years]
        self.data_lengths = [np.load(f, mmap_mode='r').shape[0] for f in self.input_files] 
        self.cumsum_lengths = np.cumsum([0] + self.data_lengths)  
        self.mean_vals = np.load('../../data/TEM-MDM/mean_vals.npy')
        self.std_vals = np.load('../../data/TEM-MDM/std_vals.npy')

    def __len__(self):
        return self.cumsum_lengths[-1]  

    def __getitem__(self, idx):
        year_idx = np.searchsorted(self.cumsum_lengths, idx, side="right") - 1
        file_idx = idx - self.cumsum_lengths[year_idx]

        input_data = (np.load(self.input_files[year_idx], mmap_mode='r')[file_idx] - self.mean_vals) / self.std_vals 
        output_data = np.load(self.output_files[year_idx], mmap_mode='r')[file_idx] / 23

        return torch.tensor(input_data, dtype=torch.float32).to(config.device), torch.tensor(output_data, dtype=torch.float32).to(config.device)
    
parser = argparse.ArgumentParser(description="Pretraining script for methane dataset")
parser.add_argument("--valid_type", type=str, help="Use temporal dataset or spatial dataset")
parser.add_argument("--spatial_fold", type=int, default=0, help="The fold using as cross validation in spatial experiment")
parser.add_argument("--model", type=str, help="Model selection")
parser.add_argument("--load_pretrain", action="store_true", help="If load pretrain model")
parser.add_argument("--id", type=str, help="Experiment identifier")
parser.add_argument("--epoch", type=int, default=200, help="Training epoches")
parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
parser.add_argument("--percent", type=float, default=0, help="Training dataset available percentage")
args = parser.parse_args()
    
config = Config(n_hidden=15)
device = torch.device("cuda")

if args.valid_type == 'temporal':
    if args.percent == 0:
        train_x = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/train_data_x.npy'), dtype=torch.float32)
        train_y = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/train_data_y.npy'), dtype=torch.float32)
    else:
        train_x = torch.tensor(np.load(f'../../processed_data/FLUXNET-CH4/temporal/train_data_x_{args.percent}.npy'), dtype=torch.float32)
        train_y = torch.tensor(np.load(f'../../processed_data/FLUXNET-CH4/temporal/train_data_y_{args.percent}.npy'), dtype=torch.float32)
    test_x = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/test_data_x.npy'), dtype=torch.float32)
    test_y = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/test_data_y.npy'), dtype=torch.float32)
else:
    train_x_list, train_y_list = [], []
    for fold in range(5):
        if fold == args.spatial_fold:
            test_x = torch.tensor(np.load(f"../../processed_data/FLUXNET-CH4/spatial/data_x_{fold}.npy"), dtype=torch.float32)
            test_y = torch.tensor(np.load(f"../../processed_data/FLUXNET-CH4/spatial/data_y_{fold}.npy"), dtype=torch.float32)
        else:
            train_x_list.append(np.load(f"../../processed_data/FLUXNET-CH4/spatial/data_x_{fold}.npy"))
            train_y_list.append(np.load(f"../../processed_data/FLUXNET-CH4/spatial/data_y_{fold}.npy"))

    train_x = torch.tensor(np.concatenate(train_x_list, axis=0), dtype=torch.float32)
    train_y = torch.tensor(np.concatenate(train_y_list, axis=0), dtype=torch.float32)

mean = torch.tensor(np.load('../../data/TEM-MDM/mean_vals.npy'))
std = torch.tensor(np.load('../../data/TEM-MDM/std_vals.npy'))
train_x = (train_x - mean) / std
test_x = (test_x - mean) / std
train_dataset = Dataset(train_x.to(device), train_y.to(device))
test_dataset = Dataset(test_x.to(device), test_y.to(device))
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

source_dataset = TemporalDataset(range(1979, 2019))
source_dataloader = DataLoader(source_dataset, batch_size=128, shuffle=True)

if args.model == 'lstm':
    model = LSTM(config).to(device)
elif args.model == 'hybrid_cnn_lstm':
    model = HybridCNNLSTM(config).to(device)
elif args.model == 'ealstm':
    model = MyEALSTM(config).to(device)
elif args.model == 'tcn':
    model = MultiTCN(config).to(device)
elif args.model == 'transformer':
    model = Transformer(config).to(device)
elif args.model == 'itransformer':
    model = iTransformer(config).to(config.device)
elif args.model == 'pyraformer':
    model = Pyraformer(config).to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.MSELoss()

if args.percent == 0:
    if args.valid_type == 'temporal':
        logging.basicConfig(
            filename=f"log/adversarial_{args.model}_{args.valid_type}_{args.id}.log", 
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO 
        )
    else:
        logging.basicConfig(
            filename=f"log/adversarial_{args.model}_{args.valid_type}_fold{args.spatial_fold}.log", 
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO 
        )
else:
    if args.valid_type == 'temporal':
        logging.basicConfig(
            filename=f"log/adversarial_{args.model}_{args.valid_type}_{args.percent}_{args.id}.log", 
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO 
        )

logger = logging.getLogger()

logger.info("Command-line Arguments:")
logger.info(json.dumps(vars(args), indent=4)) 

logger.info("Model Configuration:")
logger.info(json.dumps(vars(config), indent=4))

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GRL(nn.Module):
    def __init__(self, alpha=1.0):
        super(GRL, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradReverse.apply(x, self.alpha)

class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim):
        super(DomainDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()  
        )

    def forward(self, x):
        return self.model(x)

domain_discriminator = DomainDiscriminator(config.hidden_size).to(device)
grl = GRL(alpha=1.0).to(device)
domain_criterion = nn.BCELoss()
 
for epoch in range(args.epoch):
    running_loss = 0.0
    lambda_weight = epoch / args.epoch
    model.train()
    for num_batches, ((source_x, source_y), (target_x, target_y)) in enumerate(zip(source_dataloader, train_dataloader), start=1):
        src_embedding, src_prediction = model(source_x)
        src_prediction = src_prediction.reshape(source_y.shape)
        src_loss = criterion(source_y, src_prediction)
        domain_pred_src = domain_discriminator(src_embedding)

        tgt_embedding, tgt_prediction = model(target_x)
        tgt_prediction = tgt_prediction.reshape(target_y.shape)
        tgt_loss = criterion(target_y, tgt_prediction)
        domain_pred_tgt = domain_discriminator(tgt_embedding)

        domain_labels_source = torch.ones(domain_pred_src.shape).to(config.device)
        domain_labels_target = torch.zeros(domain_pred_tgt.shape).to(config.device)

        loss_domain = domain_criterion(domain_pred_src, domain_labels_source) + \
                      domain_criterion(domain_pred_tgt, domain_labels_target)

        loss = lambda_weight * tgt_loss + (1 - lambda_weight) * (src_loss + loss_domain) 
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
    avg_loss = math.sqrt(running_loss / len(train_dataloader))
    
    model.eval()  
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_y in test_dataloader:
            embedding, prediction = model(batch_X)
            all_predictions.append(prediction.cpu().numpy().reshape(-1))
            all_targets.append(batch_y.cpu().numpy().reshape(-1))

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    mask = ~np.isnan(all_targets)

    rmse = np.sqrt(mean_squared_error(all_targets[mask], all_predictions[mask]))
    nrmse = rmse/np.mean(all_targets)
    r2 = r2_score(all_targets, all_predictions)

    logger.info(f'Epoch {epoch}: Training loss {avg_loss}, testing rmse {rmse}, nrmse {nrmse} r2 score {r2}')

for epoch in range(args.epoch):
    running_loss = 0.0
    model.train()
    for batch_X, batch_y in train_dataloader:
        _, prediction = model(batch_X)
        loss = criterion(batch_y, prediction.view_as(batch_y))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
    avg_loss = math.sqrt(running_loss / len(train_dataloader))
    
    model.eval()  
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_y in test_dataloader:
            _, prediction = model(batch_X)
            all_predictions.append(prediction.cpu().numpy().reshape(-1))
            all_targets.append(batch_y.cpu().numpy().reshape(-1))

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    mask = ~np.isnan(all_targets)

    rmse = np.sqrt(mean_squared_error(all_targets[mask], all_predictions[mask]))
    nrmse = rmse/np.mean(all_targets)
    r2 = r2_score(all_targets, all_predictions)

    logger.info(f'Epoch {epoch+args.epoch}: Training loss {avg_loss}, testing rmse {rmse}, nrmse {nrmse} r2 score {r2}')

if args.percent == 0:
    if args.load_pretrain:
        if args.valid_type == 'temporal':
            torch.save(model.state_dict(), f'../model_save/{args.model}/adversarial_{args.valid_type}_{args.id}.pth')
        else:
            torch.save(model.state_dict(), f'../model_save/{args.model}/adversarial_{args.valid_type}_fold{args.spatial_fold}.pth')
    else:
        if args.valid_type == 'temporal':
            torch.save(model.state_dict(), f'../model_save/{args.model}/adversarial_{args.valid_type}_{args.id}.pth')
        else:
            torch.save(model.state_dict(), f'../model_save/{args.model}/adversarial_{args.valid_type}_fold{args.spatial_fold}.pth')
else:
    if args.valid_type == 'temporal':
        torch.save(model.state_dict(), f'../model_save/{args.model}/adversarial_{args.valid_type}_{args.percent}_{args.id}.pth')
