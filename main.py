import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
from transformers import XLMRobertaTokenizer,AdamW
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from model import NLIModel
from NLIDataset import NLIDataset
from transformers.utils import logging
logging.set_verbosity_error()

data=pd.read_csv('./data/train.csv')
train_data,temp_data=train_test_split(data,test_size=0.2,random_state=42)
val_data,test_data=train_test_split(temp_data,test_size=0.2,random_state=42)

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")

train_dataset=NLIDataset(train_data['premise'].tolist(),train_data['hypothesis'].tolist(),tokenizer,train_data['label'].tolist(),max_len=256)
test_dataset=NLIDataset(test_data['premise'].tolist(),test_data['hypothesis'].tolist(),tokenizer,test_data['label'].tolist(),max_len=256)
val_dataset=NLIDataset(val_data['premise'].tolist(),val_data['hypothesis'].tolist(),tokenizer,val_data['label'].tolist(),max_len=256)

train_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=16)
test_loader=DataLoader(test_dataset,batch_size=16)

if torch.cuda.is_available():
    print("gpu")
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model=NLIModel().to(device)
optimizer=AdamW(model.parameters(),lr=2e-5)
criterion=nn.CrossEntropyLoss()

epoches=5
for epoch in range(epoches):
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

model.eval()
all_preds,all_labels=[],[]
with torch.no_grad():
    for batch in test_loader:
        input_ids=batch["input_ids"].to(device)
        attention_mask=batch["attention_mask"].to(device)
        labels=batch["label"].to(device)

        outputs=model(input_ids,attention_mask)
        preds=torch.argmax(outputs,dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy=accuracy_score(all_labels,all_preds)
print(f"Test Accuracy: {accuracy:.4f}")

torch.save(model.state_dict(),"./model/best_model.pt")
print("model saved")
