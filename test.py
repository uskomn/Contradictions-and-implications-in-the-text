import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader,Dataset
from transformers import XLMRobertaTokenizer
from model import NLIModel


class NLIDataset(Dataset):
    def __init__(self,premise,hypothesis,tokenizer,max_len):
        self.premise=premise
        self.hypothesis=hypothesis
        self.tokenizer=tokenizer
        self.max_len=max_len

    def __len__(self):
        return len(self.premise)

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
        return items

data=pd.read_csv('./data/test.csv')
id=data['id']

tokenizer=XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
test_dataset=NLIDataset(data['premise'].tolist(),data['hypothesis'].tolist(),tokenizer,max_len=256)

test_loader=DataLoader(test_dataset,batch_size=16,shuffle=False)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=NLIModel().to(device)
model.load_state_dict(torch.load('./model/best_model.pt'))

all_preds=[]
with torch.no_grad():
    for batch in test_loader:
        input_ids=batch['input_ids'].to(device)
        attention_mask=batch['attention_mask'].to(device)

        outputs=model(input_ids,attention_mask)
        preds=torch.argmax(outputs,dim=1)
        all_preds.extend(preds.cpu().numpy())

frame={
    'id':id,
    'prediction':all_preds
}

submission=pd.DataFrame(frame)
submission.to_csv('./data/submission.csv',index=False)