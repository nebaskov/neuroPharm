import os
import sys
import time

# sys.path.insert(0, r'(D:\stuff\code\aiLab\neuroPharm\chemical_vae-main')
# sys.path.insert(0, r'(D:\stuff\code\aiLab\neuroPharm\neural_net_pharm')
# sys.path.insert(0, r'(D:\stuff\code\aiLab\neuroPharm\neural_net_pharm\paccmann_chemistry\paccmann_chemistry')
# sys.path.insert(0, r'(D:\stuff\code\aiLab\neuroPharm\real_project')

# print(sys.path)

import json
import numpy as np
import pandas as pd

# torch stuff
import torch
from torch.utils.data import DataLoader, Dataset

# rdkit stuff
from rdkit.Chem import PandasTools
from rdkit.Chem import AllChem as Chem

# plotting stuff
import matplotlib as mpl
import matplotlib.pyplot as plt

# PACCMANN stuff
from paccmann_chemistry.models import (
    StackGRUEncoder, StackGRUDecoder, TeacherVAE)

from paccmann_chemistry.logger import Logger
from paccmann_chemistry.training import train_vae
from paccmann_chemistry.utils import collate_fn, get_device

from pytoda.smiles.smiles_language import SMILESLanguage
# from pytoda.datasets import SMILESDataset

# custom stuff
from CustomDataset import CustomDataset

# init logger and model filepaths
training_name = 'SVAE_train_model'

logger = Logger(logdir='real_projectlogs/general_logs')  # initialize logger as an object
logger.info(f'Model with name {training_name} starts.')

model_dir = 'real_project/logs/SVAE_train_model/'
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
with open('real_project/params.json') as f:
    params.update(json.load(f))

# datasets
train_smiles_filepath = 'real_project/data/train_smiles.csv'
test_smiles_filepath = 'real_project/data/test_smiles.csv'

# SMILES language
smiles_language = SMILESLanguage()
smiles_language.add_smi(r'real_project/data/chembl_cleaned.smi')

params.update(
            {
                'input_size': smiles_language.number_of_tokens,
                'output_size': smiles_language.number_of_tokens,
                'pad_index': smiles_language.padding_index
            }
        )

vocab_dict = smiles_language.index_to_token

# # create SMILES eager dataset
# smiles_train_data = SMILESDataset(
#                         train_smiles_filepath,
#                         smiles_language=smiles_language,
#                         padding=False,
#                         add_start_and_stop=params.get('start_stop', True),
#                         backend='eager'
# )
# smiles_test_data = SMILESDataset(
#                         test_smiles_filepath,
#                         smiles_language=smiles_language,
#                         padding=False,
#                         add_start_and_stop=params.get('start_stop', True),
#                         backend='eager'
# )

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

with open('real_project/model_params.json', 'w') as fp:
    json.dump(params, fp)
    
    
smiles_dataset = pd.read_csv('real_project/data/Mol_dataset.csv')

test_size = 0.2
train_smiles = smiles_dataset.sample(round(smiles_dataset.shape[0] * (1 - test_size)))
train_labels = np.fromiter([np.random.randint(2) for i in range(train_smiles.shape[0])], dtype=int)

test_smiles = smiles_dataset.drop(train_smiles.index)
test_labels = np.fromiter([np.random.randint(2) for i in range(test_smiles.shape[0])], dtype=int)

train_ds = CustomDataset(train_smiles['smiles'].to_numpy(), train_labels)
test_ds = CustomDataset(test_smiles['smiles'].to_numpy(), test_labels)

# create DataLoaders
train_data_loader = DataLoader(
    train_ds,
    batch_size=params.get('batch_size', 64),
    # collate_fn=collate_fn,
    drop_last=True,
    pin_memory=params.get('pin_memory', True),
    num_workers=params.get('num_workers', 8))

test_data_loader = DataLoader(
    train_ds,
    batch_size=params.get('batch_size', 64),
    # collate_fn=collate_fn,
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

# todo: fix the issue with tensor in Custom Dataset

# try:
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

# except:
    # raise RuntimeError('Exception occurred while running train_vae.py.')
