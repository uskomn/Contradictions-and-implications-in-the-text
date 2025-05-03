import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
from transformers import XLMRobertaTokenizer,AdamW,XLMRobertaModel
from sklearn.metrics import accuracy_score
import pandas as pd
from transformers.utils import logging
logging.set_verbosity_error()

data=pd.read_csv('/kaggle/input/contradictory-my-dear-watson/train.csv')
train_data,val_data=train_test_split(data,test_size=0.2,random_state=42)

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")

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

train_dataset=NLIDataset(train_data['premise'].tolist(),train_data['hypothesis'].tolist(),tokenizer,train_data['label'].tolist(),max_len=256)
val_dataset=NLIDataset(val_data['premise'].tolist(),val_data['hypothesis'].tolist(),tokenizer,val_data['label'].tolist(),max_len=256)

train_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=16)

MODEL_NAME = "xlm-roberta-large"

class NLIModel(nn.Module):
    def __init__(self):
        super(NLIModel, self).__init__()
        self.roberta = XLMRobertaModel.from_pretrained(MODEL_NAME)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model=NLIModel().to(device)
optimizer=AdamW(model.parameters(),lr=2e-5)
criterion=nn.CrossEntropyLoss()

epochs=5
for epoch in range(epochs):
    model.train()
    total_loss=0.0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs=model(input_ids,attention_mask)
        loss=criterion(outputs,labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss+=loss.item()

    print(f"Epoch {epoch+1} - Training Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    all_preds,all_labels=[],[]
    with torch.no_grad():
        for batch in val_loader:
            input_ids=batch["input_ids"].to(device)
            attention_mask=batch["attention_mask"].to(device)
            labels=batch["label"].to(device)

            outputs=model(input_ids,attention_mask)
            preds=torch.argmax(outputs,dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy=accuracy_score(all_labels,all_preds)
    print(f"Val Accuracy: {accuracy:.4f}")

class TestDataset(Dataset):
    def __init__(self,premise,hypothesis,ids,tokenizer,max_len):
        self.premise=premise
        self.hypothesis=hypothesis
        self.ids=ids
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
        items['id']=self.ids[idx]
        return items

data=pd.read_csv('/kaggle/input/contradictory-my-dear-watson/test.csv')


test_dataset=TestDataset(data['premise'].tolist(),data['hypothesis'].tolist(),data['id'].tolist(),tokenizer,max_len=256)

test_loader=DataLoader(test_dataset,batch_size=16,shuffle=False)

all_preds=[]
all_ids=[]
with torch.no_grad():
    for batch in test_loader:
        input_ids=batch['input_ids'].to(device)
        attention_mask=batch['attention_mask'].to(device)
        ids=batch['id']

        outputs=model(input_ids,attention_mask)
        preds=torch.argmax(outputs,dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_ids.extend(ids)

frame={
    'id':all_ids,
    'prediction':all_preds
}

submission=pd.DataFrame(frame)
submission.to_csv('/kaggle/working/submission.csv',index=False)