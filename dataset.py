'''
We implemented loading the data
'''
from torch.utils.data import Dataset

class MolT5Dataset(Dataset):
    def __init__(self, data_path='data/train.txt'):
        super(Dataset, self).__init__()
        self.cid_list, self.smiles_list, self.description_list = [], [], []
        with open(data_path, 'r') as f:
            for line in f.readlines()[1: ]:
                cid, smiles, description = line.strip().split('\t')
                self.cid_list.append(cid)
                self.smiles_list.append(smiles)
                self.description_list.append(description)
    
    def __len__(self):
        return len(self.cid_list)

    def __getitem__(self, idx):
        return self.cid_list[idx], self.smiles_list[idx], self.description_list[idx]