from dataclasses import dataclass

@dataclass
class Config:
    model_name: str = 'microsoft/deberta-v3-base'
    num_labels: int = 5
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_epochs: int = 3
    max_seq_length: int = 128
    save_dir: str = 'models/'

    # 模型细节
    dropout_prob: float = 0.1          # Dropout 概率
    freeze_backbone: bool = False      # 是否冻结 backbone
    use_softmax: bool = True            # 预测时是否加 Softmax
