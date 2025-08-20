import torch
from torch.utils.data import Dataset
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

class CustomDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        #path = os.path.join(self.path, self.files[idx])
        data = torch.load(self.files[idx])
        return data['features'], data['label']
    
    def collate_fn(batch):
        sequences, labels = zip(*batch)
        max_size = max(i.shape[0] for i in sequences)

        padded_array = []
        padding_mask = torch.ones(len(sequences), max_size, dtype=torch.bool)

        for i, seq in enumerate(sequences):
            seq_length, feature_dim = seq.shape
            pad_size = max_size - seq_length
            padded = F.pad(seq, (0, 0, 0, pad_size))
            padded_array.append(padded)
            padding_mask[i, :seq_length] = False

        batch_seq = torch.stack(padded_array)
        batch_labels = torch.tensor(labels)

        return batch_seq, batch_labels, padding_mask
    
def get_data():
    input_path = "./final-project-code-submission-Sandwichyy/WavTokenizer/tokens/"
    x = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(".pt")]

    labels = []
    inputs = []
    # count = 0
    for i in range(len(x)):
        data = torch.load(x[i])
        labels.append(data['label'])
        inputs.append(data['features'])
        # print(count)
        # count = count + 1

    train, temp, train_labels, temp_labels = train_test_split(x, labels, test_size=0.3, random_state=42, stratify=labels)
    val, test, val_labels, test_labels = train_test_split(temp, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)

    #f = [f for f in os.listdir(input_path) if f.lower().endswith(".pt")]

    train_dataset = CustomDataset(train)
    val_dataset = CustomDataset(val)
    test_dataset = CustomDataset(test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, collate_fn=CustomDataset.collate_fn, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, collate_fn=CustomDataset.collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, collate_fn=CustomDataset.collate_fn)

    return train_loader, val_loader, test_loader


    








    



