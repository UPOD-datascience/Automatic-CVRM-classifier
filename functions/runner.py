import time
import torch
from torchmetrics.functional import (
    accuracy,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision,
    recall,
)


def train(
        model, 
        dataloader, 
        optimizer, 
        criterion,
        device, 
        epoch,
        writer
    ):
    """Execute one training epoch across batches and record training metrics."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probas = []
    
    for data, tokens, masks, targets in dataloader:
        data = data.to(device)
        tokens = tokens.to(device)
        masks = masks.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(tokens, data, masks)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        probas = outputs.softmax(dim=1)
        pred_labels = torch.argmax(probas, dim=1)
        
        all_probas.append(probas.detach().cpu())
        all_preds.append(pred_labels.detach().cpu())
        all_targets.append(targets.detach().cpu())
        running_loss += loss.item()
    
    avg_loss = running_loss / len(dataloader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_probas = torch.cat(all_probas)
    
    acc = accuracy(all_preds, all_targets, 'multiclass', num_classes = 2)
    f1 = f1_score(all_preds, all_targets, 'multiclass', num_classes = 2)
    prc = precision(all_preds, all_targets, 'multiclass', num_classes = 2)
    mcc = matthews_corrcoef(all_preds, all_targets, 'multiclass', num_classes = 2)
    rec = recall(all_preds, all_targets, 'multiclass', num_classes = 2)
    
    # Log scalar metrics to TensorBoard
    writer.add_scalar('Accuracy/train', acc, epoch)
    writer.add_scalar('F1_Score/train', f1, epoch)
    writer.add_scalar('Precision/train', prc, epoch)
    writer.add_scalar('MCC_Score/train', mcc, epoch)
    writer.add_scalar('Recall/train', rec, epoch)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    
    for class_idx in range(2):
        writer.add_histogram(
            f'Probabilities/class_{class_idx}', 
            all_probas[:, class_idx], 
            global_step = epoch
        )
    
    current_time = time.strftime("%H:%M", time.localtime())
    print(f"Train Epoch: {epoch} \tTime: {current_time} \tLoss: {avg_loss:.4f} \tAccuracy: {acc:.4f}\n")


def validate(
        model, 
        dataloader,
        device,
        epoch, 
        criterion,
        writer
    ):
    """Evaluate model on validation split and return computed evaluation metrics."""
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []
    all_probas = []
    
    with torch.no_grad():
        for data, tokens, masks, targets in dataloader:
            data = data.to(device)
            tokens = tokens.to(device)
            masks = masks.to(device)
            targets = targets.to(device)
            
            outputs = model(tokens, data, masks)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            
            probas = outputs.softmax(dim=1)
            pred_labels = torch.argmax(probas, dim=1)
            
            all_probas.append(probas.cpu())
            all_preds.append(pred_labels.cpu())
            all_targets.append(targets.cpu())
    
    avg_loss = running_loss / len(dataloader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_probas = torch.cat(all_probas)
    
    acc = accuracy(all_preds, all_targets, 'multiclass', num_classes = 2)
    f1 = f1_score(all_preds, all_targets, 'multiclass', num_classes = 2)
    prc = precision(all_preds, all_targets, 'multiclass', num_classes = 2)
    mcc = matthews_corrcoef(all_preds, all_targets, 'multiclass', num_classes = 2)
    rec = recall(all_preds, all_targets, 'multiclass', num_classes = 2)
    
    # Log validation metrics to TensorBoard
    writer.add_scalar('Accuracy/val', acc, epoch)
    writer.add_scalar('F1_Score/val', f1, epoch)
    writer.add_scalar('Precision/val', prc, epoch)
    writer.add_scalar('MCC_Score/val', mcc, epoch)
    writer.add_scalar('Recall/val', rec, epoch)
    writer.add_scalar('Loss/val', avg_loss, epoch)
    
    current_time = time.strftime("%H:%M", time.localtime())
    print(f"Validation Epoch: {epoch} \tTime: {current_time} \tLoss: {avg_loss:.4f} \tAccuracy: {acc:.4f} \tPrecision: {prc:.4f} \tF1: {f1:.4f} \tMCC: {mcc:.4f}\n")
    
    return model, acc, f1, prc, mcc, rec


def test(
        model,
        dataloader, 
        device,
        epoch,
        writer
    ):
    """Evaluate trained model on held-out test split and log confusion matrix."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    model.eval()
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for data, tokens, masks, targets in dataloader:
            data = data.to(device)
            tokens = tokens.to(device)
            masks = masks.to(device)
            targets = targets.to(device)
            
            outputs = model(tokens, data, masks)
            probas = outputs.softmax(dim=1)
            pred_labels = torch.argmax(probas, dim=1)
            
            all_preds.append(pred_labels.cpu())
            all_targets.append(targets.cpu())
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    acc = accuracy(all_preds, all_targets, 'multiclass', num_classes = 2)
    f1 = f1_score(all_preds, all_targets, 'multiclass', num_classes = 2)
    prc = precision(all_preds, all_targets, 'multiclass', num_classes = 2)
    mcc = matthews_corrcoef(all_preds, all_targets, 'multiclass', num_classes = 2)
    rec = recall(all_preds, all_targets, 'multiclass', num_classes = 2)
    cm = confusion_matrix(all_preds, all_targets, 'multiclass', num_classes = 2)
    
    fig, ax = plt.subplots(figsize = (7, 6))
    sns.heatmap(
        cm.numpy(), 
        annot = True, 
        fmt = 'd', 
        cmap = 'Blues', 
        ax = ax, 
        cbar = False,
        xticklabels = ['Class 0', 'Class 1'], 
        yticklabels = ['Class 0', 'Class 1']
    )
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    
    writer.add_scalar('Accuracy/test', acc, epoch)
    writer.add_scalar('F1_Score/test', f1, epoch)
    writer.add_scalar('Precision/test', prc, epoch)
    writer.add_scalar('MCC_Score/test', mcc, epoch)
    writer.add_scalar('Recall/test', rec, epoch)
    writer.add_figure('Confusion_matrix/test', fig, epoch)
    plt.close(fig)
    
    current_time = time.strftime("%H:%M", time.localtime())
    print(f"Test Epoch: {epoch} \tTime: {current_time} \tAccuracy: {acc:.4f} \tPrecision: {prc:.4f} \tF1: {f1:.4f} \tMCC: {mcc:.4f}\n")
    
    return model, acc, f1, prc, mcc, rec
