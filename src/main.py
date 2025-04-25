cfg = Config()

model = TransformerClassifier(
    model_name=cfg.model_name,
    num_labels=cfg.num_labels,
    dropout_prob=cfg.dropout_prob,
    freeze_backbone=cfg.freeze_backbone,
    use_softmax=cfg.use_softmax
)
