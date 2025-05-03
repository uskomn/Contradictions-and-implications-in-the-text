import torch
from torch.utils.data import Dataset


class NLIDataset(Dataset):
    def __init__(self,premise,hypothesis,tokenizer,label,max_len):
        self.premise=premise
        self.hypothesis=hypothesis
        self.tokenizer=tokenizer
        self.label=label
        self.max_len=max_len

    def __len__(self):
        return len(self.label)

    def __getitem__(self,idx):
        ecoded=self.tokenizer(
            self.premise[idx],
            self.hypothesis[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt',
            return_overflowing_tokens=False
        )
        items={key:val.squeeze(0) for key,val in ecoded.items()}
        items['label']=torch.tensor(self.label[idx],dtype=torch.long)
        return items