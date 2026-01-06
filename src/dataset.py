import torch
from torch.utils.data import Dataset
import pandas as pd


class MultiTaskDataset(Dataset):
    def __init__(self, texts, labels, priorities, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.priorities = priorities
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.label_map = {'issues': 0, 'search': 1}
        self.priority_map = {1: 0, 2: 1, 3: 2}
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.label_map[self.labels[idx]]
        priority = self.priority_map[self.priorities[idx]]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'priority': torch.tensor(priority, dtype=torch.long)
        }


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df['text'].tolist(), df['label'].tolist(), df['priority'].tolist()
