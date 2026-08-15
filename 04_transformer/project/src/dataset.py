from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch

class AGNewsDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]

        text = item["text"]
        label = item["label"]

        return text, label

def collate_fn(batch, max_len=128):
    texts, labels = zip(*batch)

    texts = [torch.tensor(text[:max_len], dtype = torch.long)
             for text in texts]

    texts = pad_sequence(texts, batch_first = True, padding_value = 0)

    labels = torch.tensor(labels, dtype = torch.long)

    return texts, labels