import os
import sys
import time

sys.path.insert(0, r'(D:\stuff\code\aiLab\neuroPharm\chemical_vae-main')
sys.path.insert(0, r'(D:\stuff\code\aiLab\neuroPharm\neural_net_pharm')

# print(sys.path)

import json
import numpy as np
import pandas as pd

# torch stuff
import torch
from torch.utils.data import DataLoader

# rdkit stuff
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import PandasTools

# plotting stuff
import matplotlib.pyplot as plt
import matplotlib as mpl

# PACCMANN stuff
from neural_net_pharm.paccmann_chemistry.paccmann_chemistry.models import (
    StackGRUEncoder, StackGRUDecoder, TeacherVAE)

from neural_net_pharm.paccmann_chemistry.paccmann_chemistry.logger import Logger
from neural_net_pharm.paccmann_chemistry.paccmann_chemistry.training import train_vae
from neural_net_pharm.paccmann_chemistry.paccmann_chemistry.utils import collate_fn, get_device
from pytoda.smiles.smiles_language import SMILESLanguage
from pytoda.datasets import SMILESDataset


# init logger and model filepaths
training_name = 'SVAE_train_model'

logger = Logger(logdir='logs/general_logs')  # initialize logger as an object
logger.info(f'Model with name {training_name} starts.')

# model_dir = os.path.join(model_path, training_name) - useless line from initial paccmann script

model_dir = 'SVAE_train_model/'
log_path = os.path.join(model_dir, 'logs')
val_dir = os.path.join(log_path, 'val_logs')
os.makedirs(os.path.join(model_dir, 'weights'), exist_ok=True)
os.makedirs(os.path.join(model_dir, 'results'), exist_ok=True)
os.makedirs(log_path, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
train_logger = Logger(log_path)
val_logger = Logger(val_dir)

# get params
params = dict()
with open('params.json') as f:
    params.update(json.load(f))

# datasets
train_smiles_filepath = 'data/train_smiles.csv'
test_smiles_filepath = 'data/test_smiles.csv'

# SMILES language
smiles_language = SMILESLanguage.load('data/Mol_dataset.csv')

params.update(
            {
                'input_size': smiles_language.number_of_tokens,
                'output_size': smiles_language.number_of_tokens,
                'pad_index': smiles_language.padding_index
            }
        )

# create SMILES eager dataset
smiles_train_data = SMILESDataset(
                        train_smiles_filepath,
                        smiles_language=smiles_language,
                        padding=False,
                        add_start_and_stop=params.get('start_stop', True),
                        backend='eager'
)
smiles_test_data = SMILESDataset(
                        test_smiles_filepath,
                        smiles_language=smiles_language,
                        padding=False,
                        add_start_and_stop=params.get('start_stop', True),
                        backend='eager'
)

vocab_dict = smiles_language.index_to_token

params.update(
                {
                    'start_index':
                    list(vocab_dict.keys())
                    [list(vocab_dict.values()).index('<START>')],
                    'end_index':
                    list(vocab_dict.keys())
                    [list(vocab_dict.values()).index('<STOP>')]
                }
)

with open(os.path.join(model_dir, 'model_params.json'), 'w') as fp:
    json.dump(params, fp)


# create DataLoaders
train_data_loader = DataLoader(
    smiles_train_data,
    batch_size=params.get('batch_size', 64),
    collate_fn=collate_fn,
    drop_last=True,
    pin_memory=params.get('pin_memory', True),
    num_workers=params.get('num_workers', 8))

test_data_loader = DataLoader(
    smiles_test_data,
    batch_size=params.get('batch_size', 64),
    collate_fn=collate_fn,
    drop_last=True,
    pin_memory=params.get('pin_memory', True),
    num_workers=params.get('num_workers', 8))

# initialize encoder and decoder
device = get_device()
gru_encoder = StackGRUEncoder(params).to(device)
gru_decoder = StackGRUDecoder(params).to(device)
gru_vae = TeacherVAE(gru_encoder, gru_decoder).to(device)

loss_tracker = {
    'test_loss_a': 10e4,
    'test_rec_a': 10e4,
    'test_kld_a': 10e4,
    'ep_loss': 0,
    'ep_rec': 0,
    'ep_kld': 0
}

# train for n_epoch epochs
logger.info(
    'Model creation and data processing done, Training starts.')
try:
    for epoch in range(params['epochs'] + 1):
        start_time = time.time()
        loss_tracker = train_vae(
            epoch,
            gru_vae,
            train_data_loader,
            test_data_loader,
            smiles_language,
            model_dir,
            optimizer=params.get('optimizer', 'Adadelta'),
            lr=params['learning_rate'],
            kl_growth=params['kl_growth'],
            input_keep=params['input_keep'],
            test_input_keep=params['test_input_keep'],
            start_index=params['start_index'],
            end_index=params['end_index'],
            generate_len=params['generate_len'],
            temperature=params['temperature'],
            log_interval=params['log_interval'],
            save_interval=params['save_interval'],
            eval_interval=params['eval_interval'],
            loss_tracker=loss_tracker,
            train_logger=train_logger,
            val_logger=val_logger,
            logger=logger
        )

        # logs
        print(f'Epoch {epoch}, took {time.time() - start_time} ms.')
        print(
            f"OVERALL:\n",
            f"Best loss = {loss_tracker['test_loss_a']} in Ep {loss_tracker['ep_loss']},\n "
            f"best Rec = {loss_tracker['test_rec_a']} in Ep {loss_tracker['ep_rec']},\n "
            f"best KLD = {loss_tracker['test_kld_a']} in Ep {loss_tracker['ep_kld']}")

        print('Training done, shutting down.')

except:
    raise RuntimeError('Exception occurred while running train_vae.py.')
