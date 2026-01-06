import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from typing import Iterable, Union, List, Tuple


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


def load_data(csv_paths: Union[str, Path, Iterable[Union[str, Path]]]) -> Tuple[List[str], List[str], List[int]]:
    if isinstance(csv_paths, (str, Path)):
        paths = [Path(csv_paths)]
    else:
        paths = [Path(p) for p in csv_paths]

    if not paths:
        raise ValueError("No CSV paths provided")

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        missing = {"text", "label", "priority"} - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns {sorted(missing)} in {path}")
        frames.append(df[["text", "label", "priority"]])

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.dropna(subset=["text", "label", "priority"])
    return (
        df_all["text"].astype(str).tolist(),
        df_all["label"].astype(str).tolist(),
        df_all["priority"].astype(int).tolist(),
    )
