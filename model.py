import torch
import torch.nn as nn
from transformers import XLMRobertaModel

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