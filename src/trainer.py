def train(model, dataloader, cfg, device):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(cfg.max_epochs):
        for batch in dataloader:
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(**inputs, return_embedding=False)  # 训练时不需要 embedding
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
