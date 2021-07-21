from DataSet import ReconstructDataSet
from torch.utils.data.dataloader import DataLoader
from tqdm import trange
from models.model import Model
import argparse
from torch.nn.parallel import DataParallel
from tqdm import tqdm
import torch
import os
import utils
import yaml

config = 'config/config_train.yaml'

with open(config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

dataset = ReconstructDataSet(config['dataroot'], config)

print (dataset.filelists)
sampler = utils.TrainSampler(config['batchsize'], dataset.filelists)

for i in sampler:
    print (i)

exit()
data_loader = DataLoader(dataset, batch_sampler=sampler, num_workers=16, pin_memory=True)
