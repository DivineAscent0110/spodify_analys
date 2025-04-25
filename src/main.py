import torch
import pandas as pd
from transformers import AutoTokenizer

from src.config import Config
from src.preprocess import clean_data
from src.model import TransformerClassifier
from src.trainer import train

# DataLoader & Dataset
from torch.utils.data import DataLoader, Dataset


cfg = Config()

model = TransformerClassifier(
    model_name=cfg.model_name,
    num_labels=cfg.num_labels,
    dropout_prob=cfg.dropout_prob,
    freeze_backbone=cfg.freeze_backbone,
    use_softmax=cfg.use_softmax
)
