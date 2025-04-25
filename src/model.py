import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class TransformerClassifier(nn.Module):
    def __init__(self, model_name, num_labels=5, dropout_prob=0.1, freeze_backbone=False, use_softmax=False, task_type='single_label'):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.use_softmax = use_softmax
        self.task_type = task_type

        # ==============================
        # Freeze Transformer backbone if needed
        # 提供灵活控制，便于实验不同微调策略（只训练分类器 vs. 全模型微调）
        # ==============================
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ==============================
        # Classification head
        # 适配不同任务（单标签/多标签）
        # ==============================
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, return_embedding=False):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0]
        cls_token = self.dropout(cls_token)
        logits = self.classifier(cls_token)

        if return_embedding:
            return logits, cls_token  # 提供中间特征向量，方便做特征可视化或下游任务

        # ==============================
        # Output activation logic
        # 单标签：softmax， 多标签：sigmoid
        # 设计为灵活可配置，适应不同任务类型
        # ==============================
        if self.use_softmax and self.task_type == 'single_label':
            logits = F.softmax(logits, dim=-1)
        elif self.task_type == 'multi_label':
            logits = torch.sigmoid(logits)

        return logits
