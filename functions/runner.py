import torch

from torchmetrics.functional import f1_score, \
    precision, \
    recall, \
    matthews_corrcoef, \
    accuracy, \
    confusion_matrix

import seaborn as sns

import matplotlib.pyplot as plt

import time


def train(
        model, 
        dataloader, 
        optimizer, 
        criterion,
        device, 
        epoch,
        writer
    ):
    
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probas = []
    
    for data, tokens, masks, targets in dataloader:
        
        #print(data.shape, tokens.shape, targets.shape)
        
        data = data.to(device)
        tokens = tokens.to(device)
        targets = targets.to(device)
        masks = masks.to(device)
        
        #print(data.shape)
        
        #print(data.shape, tokens.shape, targets.shape)
        #print(tokens.shape, masks.shape)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(tokens, data, masks)
        
        #print(outputs.shape, outputs)
        
        # Compute loss
        loss = criterion(outputs, targets)
        
        # Back-propagation
        loss.backward()
        
        # Aggiornamento dei pesi su tutto il classificatore
        optimizer.step()
        
        # Apply softmax activation to outputs
        # del classificatore
        probas = outputs.softmax(dim=1)
        all_probas.append(probas)
        
        #print(outputs, probas)
        
        # Viene scelta la classe con probabilità più alta
        pred_labels = torch.argmax(probas, dim=1)
        
        #print(pred_labels.shape)
        
        all_preds.append(pred_labels) # store preds
        all_targets.append(targets) # store targets
        
        #logger.info(f"Loss: {loss}, k: {k}, epoch: {epoch}, preds: {pred_labels}, targets: {targets}")
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(dataloader)
    
    all_preds = torch.concat(all_preds)
    all_targets = torch.concat(all_targets)
    all_probas  = torch.concat(all_probas)
    
    #print(all_preds.shape, all_targets.shape)
    
    acc = accuracy(all_preds,           all_targets, 'multiclass', num_classes=2)
    f1  = f1_score(all_preds,           all_targets, 'multiclass', num_classes=2)
    prc = precision(all_preds,          all_targets, 'multiclass', num_classes=2)
    mcc = matthews_corrcoef(all_preds,  all_targets, 'multiclass', num_classes=2)
    rec = recall(all_preds,             all_targets, 'multiclass', num_classes=2)
    #cm = confusion_matrix(all_preds,    all_targets, 'multiclass', num_classes=2)
    
    
    # Log to TensorBoard
    writer.add_scalar('Accuracy/train',         acc,        epoch)
    writer.add_scalar('F1_Score/train',         f1,         epoch)
    writer.add_scalar('Precision/train',        prc,        epoch)
    writer.add_scalar('MCC_Score/train',        mcc,        epoch)
    writer.add_scalar('Recall/train',           rec,        epoch)
    writer.add_scalar('Loss/train',             avg_loss,   epoch)
    #writer.add_figure('Confusion_matrix/train', sns.heatmap(cm, annot=True).get_figure(),   epoch)
    
    for class_idx in range(2):
        writer.add_histogram(
            f'Probabilities/class_{class_idx}', 
            all_probas[:, class_idx], 
            global_step = 0
        )
    
    current_time = time.strftime("%H:%M", time.localtime())
    
    print(f"Train Epoch: {epoch} \tTime: {current_time} \tLoss: {avg_loss:.4f} \tAccuracy: {acc:.4f}\n")
    
    '''
    # Save model weights and metrics per epoch
    joblib.dump({
            'model_weights'  : model.state_dict(),
            'loss'   : criterion,
            'optim'  : optimizer,
            'epoch'  : epoch,
            'metrics': {
                'avg_loss' : avg_loss,
                'accuracy' : accuracy,
                'f1'       : f1,
                'prc'      : prc,
                'mcc'      : mcc,
                'probas'   : all_probas,
            }
        },
        os.path.join(writer.log_dir, f'train_epoch_{epoch}.joblib')
    )
    '''


def validate(
        model, 
        dataloader,
        device,
        epoch, 
        criterion,
        writer
    ):
    
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
            
            #print(outputs, targets)
            #print(outputs.shape, targets.shape)
            
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            
            # Apply softmax activation to outputs
            # del classificatore
            probas = outputs.softmax(dim=1)
            all_probas.append(probas)
            
            # Viene scelta la classe con probabilità più alta
            pred_labels = torch.argmax(probas, dim=1)
            
            all_preds.append(pred_labels)
            all_targets.append(targets)
    
    avg_loss = running_loss / len(dataloader)
    
    all_preds   = torch.concat(all_preds)
    all_targets = torch.concat(all_targets)
    all_probas  = torch.concat(all_probas)
    
    acc = accuracy(all_preds,           all_targets, 'multiclass', num_classes=2)
    f1  = f1_score(all_preds,           all_targets, 'multiclass', num_classes=2)
    prc = precision(all_preds,          all_targets, 'multiclass', num_classes=2)
    mcc = matthews_corrcoef(all_preds,  all_targets, 'multiclass', num_classes=2)
    rec = recall(all_preds,             all_targets, 'multiclass', num_classes=2)
    '''
    cm  = confusion_matrix(all_preds,   all_targets, 'multiclass', num_classes=2)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        ax=ax, 
        cbar=False,
        xticklabels=['Class 0', 'Class 1'], 
        yticklabels=['Class 0', 'Class 1']
    )
    
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    '''
    
    # Log to TensorBoard
    writer.add_scalar('Accuracy/val',           acc,        epoch)
    writer.add_scalar('F1_Score/val',           f1,         epoch)
    writer.add_scalar('Precision/val',          prc,        epoch)
    writer.add_scalar('MCC_Score/val',          mcc,        epoch)
    writer.add_scalar('Recall/val',             rec,        epoch)
    writer.add_scalar('Loss/val',               avg_loss,   epoch)
    #writer.add_figure('Confusion_matrix/val',   fig,        epoch)
    
    current_time = time.strftime("%H:%M", time.localtime())
    
    '''
    joblib.dump({
            'model'  : model.state_dict(),
            'loss'   : criterion,
            'epoch'  : epoch,
            'metrics': {
                'avg_loss' : avg_loss,
                'accuracy' : acc,
                'f1'       : f1,
                'prc'      : prc,
                'mcc'      : mcc
            }
        },
        os.path.join(writer.log_dir, f'val_epoch_{epoch}.joblib')
    )
    '''
    
    print(f"Validation Epoch: {epoch} \tTime: {current_time} \tLoss: {avg_loss:.4f} \tAccuracy: {acc:.4f} \tPrecision: {prc:.4f} \tF1: {f1:.4f} \tMCC: {mcc:.4f}\n")
    
    return model, acc, f1, prc, mcc, rec


def test(
        model,
        dataloader, 
        device,
        epoch,
        writer
    ):
    
    model.eval()
    all_targets = []
    all_preds   = []
    
    with torch.no_grad():
        for data, tokens, masks, targets in dataloader:
            
            data = data.to(device)
            tokens = tokens.to(device)
            masks = masks.to(device)
            targets = targets.to(device)
            
            #print(data.shape)         
            
            outputs = model(tokens, data, masks)
            
            # Apply softmax activation to outputs
            # del classificatore
            probas = outputs.softmax(dim=1)
            
            # Viene scelta la classe con probabilità più alta
            pred_labels = torch.argmax(probas, dim=1)
            
            all_preds.append(pred_labels)
            all_targets.append(targets)
    
    all_preds   = torch.concat(all_preds)
    all_targets = torch.concat(all_targets)
    
    all_preds = all_preds.cpu()
    all_targets = all_targets.cpu()
    
    acc = accuracy(all_preds,           all_targets, 'multiclass', num_classes=2)
    f1  = f1_score(all_preds,           all_targets, 'multiclass', num_classes=2)
    prc = precision(all_preds,          all_targets, 'multiclass', num_classes=2)
    mcc = matthews_corrcoef(all_preds,  all_targets, 'multiclass', num_classes=2)
    rec = recall(all_preds,             all_targets, 'multiclass', num_classes=2)
    cm  = confusion_matrix(all_preds,   all_targets, 'multiclass', num_classes=2)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        ax=ax, 
        cbar=False,
        xticklabels=['Class 0', 'Class 1'], 
        yticklabels=['Class 0', 'Class 1']
    )
    
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    
    # Log to TensorBoard
    writer.add_scalar('Accuracy/test',          acc,    epoch)
    writer.add_scalar('F1_Score/test',          f1,     epoch)
    writer.add_scalar('Precision/test',         prc,    epoch)
    writer.add_scalar('MCC_Score/test',         mcc,    epoch)
    writer.add_scalar('Recall/test',            rec,    epoch)
    writer.add_figure('Confusion_matrix/test',  fig,    epoch)
    
    current_time = time.strftime("%H:%M", time.localtime())
    
    print(f"Test Epoch: {epoch} \tTime: {current_time} \tAccuracy: {acc:.4f} \tPrecision: {prc:.4f} \tF1: {f1:.4f} \tMCC: {mcc:.4f}\n")
    
    return model, acc, f1, prc, mcc, rec


'''
def find_best_mcc(log_dir):
    best_mcc = -float('inf')  # Initialize best metric threshold
    best_epoch = -1
    best_model_weights = None
    
    # Scorri tutte le epoche salvate nella directory di log
    for epoch_file in os.listdir(log_dir):
        if epoch_file.startswith("val") and epoch_file.endswith(".joblib"):  # Verify file is a joblib file
            # Carica il salvataggio
            epoch_data = joblib.load(os.path.join(log_dir, epoch_file))
            
            # Estrai il valore di MCC
            mcc = epoch_data['metrics']['mcc']
            
            # Check if l'MCC corrente è il migliore
            if mcc > best_mcc:
                best_mcc = mcc
                best_epoch = int(epoch_file.split('_')[1].split('.')[0])  # Ottieni il numero dell'epoca
                best_model_weights = epoch_data['model_weights']
                
    # Return best weights, best epoch, and best MCC
    return best_model_weights, best_epoch, best_mcc
'''