import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import XLMRobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
import pandas as pd
from model import NLIModel
from NLIDataset import NLIDataset
from transformers.utils import logging
from torch.cuda.amp import autocast, GradScaler
logging.set_verbosity_error()

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.smoothing) + (1 - one_hot) * self.smoothing / (n_class - 1)
        log_prob = torch.nn.functional.log_softmax(pred, dim=1)
        return -(one_hot * log_prob).sum(dim=1).mean()


class FGM():
    def __init__(self, model, epsilon=1.0):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    def attack(self, embed_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and embed_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


data = pd.read_csv('./data/train.csv')
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.2, random_state=42)

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")

train_dataset = NLIDataset(train_data['premise'].tolist(), train_data['hypothesis'].tolist(), tokenizer, train_data['label'].tolist(), max_len=256)
val_dataset = NLIDataset(val_data['premise'].tolist(), val_data['hypothesis'].tolist(), tokenizer, val_data['label'].tolist(), max_len=256)
test_dataset = NLIDataset(test_data['premise'].tolist(), test_data['hypothesis'].tolist(), tokenizer, test_data['label'].tolist(), max_len=256)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using:", device)

model = NLIModel().to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = LabelSmoothingLoss(smoothing=0.1)


total_steps = len(train_loader) * 5
warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)


scaler = GradScaler()
fgm = FGM(model)


epoches = 5
for epoch in range(epoches):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

        # backward & update (amp)
        scaler.scale(loss).backward()

        # 对抗训练
        fgm.attack()
        with autocast():
            adv_outputs = model(input_ids, attention_mask)
            adv_loss = criterion(adv_outputs, labels)
        scaler.scale(adv_loss).backward()
        fgm.restore()

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} - Training Loss: {total_loss / len(train_loader):.4f}")

    # ---------- 验证 ----------
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with autocast():
                outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Val Accuracy: {accuracy:.4f}")

# ---------- 测试 ----------
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with autocast():
            outputs = model(input_ids, attention_mask)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy:.4f}")

# ---------- 保存模型 ----------
torch.save(model.state_dict(), "./model/best_model.pt")
print("Model saved.")
