from dataclasses import dataclass

@dataclass
class Config:
    # 模型相关
    model_name: str = 'distilbert-base-uncased'  # Transformer 模型名称
    num_labels: int = 5                          # 情感分类：假设1-5星

    # 数据相关
    data_path: str = 'data/review_data.csv'      # 数据文件路径
    max_seq_length: int = 128                    # 最大序列长度

    # 训练相关
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_epochs: int = 3

    # 设备与保存
    device: str = 'cuda'                         # 设备：cuda / cpu
    save_dir: str = 'models/'                    # 模型保存路径
    log_dir: str = 'logs/'                       # 日志保存路径

