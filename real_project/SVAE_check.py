import json
import torch
import matplotlib
import pandas as pd

from rdkit.Chem import PandasTools
from tensorboard.notebook import display

from paccmann.paccmann_chemistry.paccmann_chemistry.models import (
    StackGRUEncoder, StackGRUDecoder, TeacherVAE)

model_path = "SVAE_train_model/weights/best_loss.pt"

params = dict()
with open('params.json') as f:
    params.update(json.load(f))

model = TeacherVAE(StackGRUEncoder(params), StackGRUDecoder(params)).load_model(model_path)
molecule_iter, molecule_ds = model.generate("C(=O)NOc1ccccc1C(N)=O".upper())  # aspirin molecule
print("SMILES: \n", molecule_ds)

# not sure that the following works
display(PandasTools.FrameToGridImage(molecule_ds, column='smiles', molsPerRow=5))
