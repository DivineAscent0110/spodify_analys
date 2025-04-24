import torch.nn as nn
from transformers import AutoModel

class TransformerClassifier(nn.Module):
    def __init__(self, model_name, num_labels=5):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0]  # CLS token
        logits = self.classifier(cls_token)
        return logits

