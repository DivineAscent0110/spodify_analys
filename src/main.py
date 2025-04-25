def main():
    cfg = Config()

    # 数据加载
    df = pd.read_csv(cfg.data_path)
    df = clean_data(df)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    class ReviewDataset(Dataset):
        def __init__(self, df, tokenizer, max_length):
            self.df = df
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            text = self.df.iloc[idx]['processed_review_text']
            label = self.df.iloc[idx]['review_rating'] - 1  # 1-5转0-4
            encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
            return {key: val.squeeze(0) for key, val in encoding.items()}, label

    dataset = ReviewDataset(df, tokenizer, cfg.max_seq_length)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    # 模型初始化
    model = TransformerClassifier(
        model_name=cfg.model_name,
        num_labels=cfg.num_labels,
        dropout_prob=cfg.dropout_prob,
        freeze_backbone=cfg.freeze_backbone,
        use_softmax=cfg.use_softmax
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 训练
    train(model, dataloader, cfg, device)

    # 模型保存
    torch.save(model.state_dict(), cfg.save_dir + 'best_model.pt')

if __name__ == "__main__":
    main()
