@dataclass
class Config:
    model_name: str = 'microsoft/deberta-v3-base'
    num_labels: int = 5
    data_path: str = 'data/review_data.csv'
    batch_size: int = 16

    learning_rate: float = 2e-5
    max_epochs: int = 3
    max_seq_length: int = 128
    save_dir: str = 'models/'
    log_dir: str = 'logs/'

    dropout_prob: float = 0.1
    freeze_backbone: bool = True
    use_softmax: bool = False
