# ==============================
# Main training script
# 训练入口，支持命令行参数灵活配置（模型、任务类型、训练策略等）
# 自动适配单标签/多标签任务，DDP 分布式训练，支持 AMP 和梯度累计
# ==============================

import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse

from src.config import Config
from src.preprocess import clean_data
from src.model import TransformerClassifier
from src.trainer import train

# ==============================
# Dataset definition
# 自动适配单标签/多标签任务：读取 review_rating（单标签）或 labels（多标签）
# 灵活适配不同数据格式，便于扩展到其他任务
# ==============================
class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, task_type='single_label'):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['processed_review_text']
        if self.task_type == 'multi_label':
            labels = torch.tensor(eval(self.df.iloc[idx]['labels']), dtype=torch.float)
        else:
            labels = torch.tensor(self.df.iloc[idx]['review_rating'] - 1, dtype=torch.long)
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {key: val.squeeze(0) for key, val in encoding.items()}, labels

# ==============================
# DDP setup function
# 初始化分布式训练环境，支持单机多卡或多机多卡训练，提升训练效率
# ==============================
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, args):
    cfg = Config()
    cfg.model_name = args.model_name
    cfg.batch_size = args.batch_size
    cfg.max_epochs = args.max_epochs
    cfg.learning_rate = args.learning_rate
    cfg.freeze_backbone = args.freeze_backbone
    cfg.use_softmax = args.use_softmax
    cfg.gradient_accumulation_steps = args.gradient_accumulation_steps
    cfg.use_amp = args.use_amp
    cfg.task_type = args.task_type

    setup(rank, world_size)

    df = pd.read_csv(cfg.data_path)
    df = clean_data(df)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    dataset = ReviewDataset(df, tokenizer, cfg.max_seq_length, cfg.task_type)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, sampler=val_sampler)

    model = TransformerClassifier(
        model_name=cfg.model_name,
        num_labels=cfg.num_labels,
        dropout_prob=cfg.dropout_prob,
        freeze_backbone=cfg.freeze_backbone,
        use_softmax=cfg.use_softmax,
        task_type=cfg.task_type
    ).to(rank)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    train(model, train_loader, val_loader, cfg, rank)

    cleanup()

if __name__ == "__main__":
    # ==============================
    # Command-line argument parser
    # 灵活配置模型、训练超参数、AMP、梯度累计、任务类型（单标签/多标签）
    # 方便快速实验和调优
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-base')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--use_softmax', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--task_type', type=str, default='single_label', choices=['single_label', 'multi_label'])
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
