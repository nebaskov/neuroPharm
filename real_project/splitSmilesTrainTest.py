import pandas as pd

ds = pd.read_csv('real_project/data/Mol_dataset.csv')

print(ds.shape[0] * 0.2)
test_ds = ds.sample(int(round(ds.shape[0] * 0.2, 0)))
train_ds = ds.drop(test_ds.index, axis=0)

train_ds.to_csv('real_project/data/train_smiles.csv')
test_ds.to_csv('real_project/data/test_smiles.csv')
