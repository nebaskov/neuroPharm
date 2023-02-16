import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, text, labels=None):
        self.text = text
        
        self.labels = labels

    def __len__(self):
        """Returns sample number in the dataset

        Returns:
            int: number of samples
        """
        
        return self.text.shape[0]
            
    def __getitem__(self, idx):
        """Get the item from the dataset

        Args:
            idx (int): index of the item

        Returns:
            dict: dictionary consisting of {data: label} values
        """
        
        try:
            label = self.labels[idx]
        except:
            label = "Label is set to None"
            
        data = self.text[idx]
        
        # sample = {"Text": data, "Class": label}
        # return sample
        
        sample_data = torch.tensor(data)
        return sample_data