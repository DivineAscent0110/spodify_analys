import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class TransformerClassifier(nn.Module):
    def __init__(self, model_name, num_labels=5, dropout_prob=0.1, freeze_backbone=False, use_softmax=False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.use_softmax = use_softmax

        # 冻结 backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, return_embedding=False):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0]
        cls_token = self.dropout(cls_token)
        logits = self.classifier(cls_token)

        if return_embedding:
            return logits, cls_token  # 返回 logits + 特征向量

        if self.use_softmax:
            logits = F.softmax(logits, dim=-1)

        return logits
