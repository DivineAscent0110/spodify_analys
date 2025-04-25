import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train(model, train_loader, val_loader, cfg, device):
    model.to(device)

    # ==============================
    # Optimizer & Scheduler
    # 灵活替换不同优化器（如 AdaFactor、LAMB）
    # ==============================
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)

    # ==============================
    # Loss function selection
    # 根据任务类型自动选择适合的 loss，便于扩展自定义 loss
    # ==============================
    if cfg.task_type == 'single_label':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    writer = SummaryWriter(cfg.log_dir)
    best_f1 = 0
    patience_counter = 0

    model.train()
    for epoch in range(cfg.max_epochs):
        total_loss = 0

        # ==============================
        # Training loop
        # 完全手写，方便插入梯度裁剪、对抗训练、混合精度等
        # ==============================
        for step, batch in enumerate(train_loader):
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            if cfg.task_type == 'multi_label':
                labels = labels.float()

            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                outputs = model(**inputs)
                loss = criterion(outputs, labels) / cfg.gradient_accumulation_steps

            scaler.scale(loss).backward()

            # ==============================
            # Gradient accumulation
            # 支持大 batch 训练，节省显存
            # ==============================
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * cfg.gradient_accumulation_steps

        avg_loss = total_loss / len(train_loader)
        metrics = evaluate(model, val_loader, device, cfg.task_type)
        scheduler.step()

        # ==============================
        # TensorBoard logging
        # ==============================
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/val', metrics['accuracy'], epoch)
        writer.add_scalar('Precision/val', metrics['precision'], epoch)
        writer.add_scalar('Recall/val', metrics['recall'], epoch)
        writer.add_scalar('F1/val', metrics['f1'], epoch)

        print(f"Epoch {epoch+1}/{cfg.max_epochs}, Loss: {avg_loss:.4f}, Val Acc: {metrics['accuracy']:.4f}, Val F1: {metrics['f1']:.4f}")

        # ==============================
        # Early Stopping & Best Model Saving
        # 自动保存最优模型，避免过拟合，训练更高效
        # ==============================
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            patience_counter = 0
            torch.save(model.state_dict(), cfg.save_dir + 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                print("Early stopping triggered!")
                break

    writer.close()

def evaluate(model, dataloader, device, task_type):
    model.eval()
    preds_list = []
    labels_list = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs)
            if task_type == 'multi_label':
                preds = (outputs > 0.5).int()
                labels_list.extend(labels.cpu().tolist())
                preds_list.extend(preds.cpu().tolist())
            else:
                preds = outputs.argmax(dim=1)
                preds_list.extend(preds.cpu().tolist())
                labels_list.extend(labels.cpu().tolist())

    if task_type == 'multi_label':
        accuracy = accuracy_score(labels_list, preds_list)
        precision = precision_score(labels_list, preds_list, average='samples', zero_division=0)
        recall = recall_score(labels_list, preds_list, average='samples', zero_division=0)
        f1 = f1_score(labels_list, preds_list, average='samples', zero_division=0)
    else:
        accuracy = accuracy_score(labels_list, preds_list)
        precision = precision_score(labels_list, preds_list, average='weighted', zero_division=0)
        recall = recall_score(labels_list, preds_list, average='weighted', zero_division=0)
        f1 = f1_score(labels_list, preds_list, average='weighted', zero_division=0)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
