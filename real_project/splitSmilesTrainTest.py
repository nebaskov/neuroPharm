import pandas as pd

ds = pd.read_csv('data/Mol_dataset.csv.csv')

print(ds.shape[0] * 0.2)
test_ds = ds.sample(7456)
train_ds = ds.drop(test_ds.index, axis=0)

train_ds.to_csv('data/train_smiles.csv')
test_ds.to_csv('data/test_smiles.csv')
