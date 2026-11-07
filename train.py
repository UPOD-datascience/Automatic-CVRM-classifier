import os
import random
import argparse
from datetime import datetime as dt


def set_seed(seed: int) -> None:
    """Set random seed across Python, NumPy, and PyTorch for deterministic reproducibility."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(
        epochs                  : int   = 30, 
        model_name              : str   = 'Resnet',
        batch_size              : int   = 32,
        learning_rate           : float = 3e-5, 
        weight_decay            : float = 1e-4,
        tokens_len              : int   = 8192,
        embedding_dim           : int   = 512,
        num_classes             : int   = 2,
        amsgrad                 : bool  = True,
        n_splits                : int   = 5,
        test_size               : float = 0.2,
        desired_percentage      : float = 0.3,
        results_folder          : str   = 'results',
        pooling                 : str   = 'cls',
        dataset_path            : str   = None,
        text_col                : str   = None,
        tokenizer_dir           : str   = None,
        tokenizer_vocab_path    : str   = None,
        tokenizer_merges_path   : str   = None,
        device                  : str   = 'auto',
        seed                    : int   = 42
    ):
    """Execute high-speed stratified k-fold cross-validation pipeline."""
    import numpy as np
    import pandas as pd
    import joblib
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from imblearn.over_sampling import RandomOverSampler

    set_seed(seed)
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using compute device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    if dataset_path is None:
        raise ValueError("Please provide a dataset path via --dataset_path.")
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(script_dir, dataset_path)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at '{dataset_path}'.")
        
    df = pd.read_parquet(dataset_path) if dataset_path.endswith('.parquet') else pd.read_csv(dataset_path)
    labels = df.iloc[:, -1]
    df = df.iloc[:, 1:-1]
    
    majority_class = labels.value_counts().max()
    desired_minority = int(desired_percentage * len(labels))
    alpha = desired_minority / majority_class
    oversampler = RandomOverSampler(sampling_strategy = alpha, random_state = seed) if desired_percentage else None
    
    tabular_features = df.select_dtypes(include = ['number']).columns.tolist()
    if text_col and text_col in df.columns:
        unstructured_features = [text_col]
    else:
        unstructured_features = df.select_dtypes(include = ['object', 'string']).columns.tolist()
        if len(unstructured_features) == 0:
            raise ValueError("No text column detected in dataset.")
            
    print(f"Detected {len(tabular_features)} tabular features and {len(unstructured_features)} unstructured text feature(s).")
    texts = df[unstructured_features[0]].fillna("").astype(str).tolist() if len(unstructured_features) == 1 else df[unstructured_features].agg(" ".join, axis = 1).fillna("").astype(str).tolist()
    all_tabular = df[tabular_features].to_numpy(dtype = np.float32)
    all_labels = labels.to_numpy(dtype = np.int64)
    
    indices = np.arange(len(all_labels))
    trainval_indices, test_indices = train_test_split(
        indices,
        test_size = test_size, 
        stratify = all_labels, 
        random_state = seed
    )
    
    timestamp = dt.now().strftime("%Y_%m_%d_%H_%M")
    if model_name == 'HTransformer':
        log_dir = os.path.join('HTransformer1D', f'{timestamp}_{pooling.upper()}')
    elif model_name == 'Resnet':        
        log_dir = os.path.join('ResConsultNet', timestamp)
    elif model_name == 'LinearSVM':
        log_dir = os.path.join('LinearSVM', timestamp)
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")
    
    results_path = os.path.join(script_dir, results_folder)
    model_path = os.path.join(results_path, log_dir)
    os.makedirs(model_path, exist_ok = True)
    
    skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = seed)
    test_performance = []
    best_acc, best_f1, best_prc, best_rec, best_mcc = -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf')
    best_model = None
    labels_trainval = all_labels[trainval_indices]

    # =========================================================================
    # Linear SVM Pipeline
    # =========================================================================
    if model_name == 'LinearSVM':
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
        
        for split, (train_rel_idx, val_rel_idx) in enumerate(skf.split(trainval_indices, labels_trainval)):
            print(f"\n{'='*20} Fold {split + 1}/{n_splits} (LinearSVM) {'='*20}")
            fold_train_idx = trainval_indices[train_rel_idx]
            fold_val_idx = trainval_indices[val_rel_idx]
            
            if oversampler:
                oversampled_idx, fold_train_labels = oversampler.fit_resample(fold_train_idx.reshape(-1, 1), all_labels[fold_train_idx])
                fold_train_idx = oversampled_idx.flatten()
            else:
                fold_train_labels = all_labels[fold_train_idx]
                
            train_texts = [texts[i] for i in fold_train_idx]
            val_texts = [texts[i] for i in fold_val_idx]
            
            vectorizer = TfidfVectorizer(max_df = 0.98, min_df = 1, sublinear_tf = True)
            X_text_train = vectorizer.fit_transform(train_texts).toarray()
            X_text_val = vectorizer.transform(val_texts).toarray()
            
            if all_tabular.shape[1] > 0:
                X_train = np.concatenate([X_text_train, all_tabular[fold_train_idx]], axis = 1)
                X_val = np.concatenate([X_text_val, all_tabular[fold_val_idx]], axis = 1)
            else:
                X_train, X_val = X_text_train, X_text_val
                
            fold_scaler = StandardScaler()
            X_train = fold_scaler.fit_transform(X_train)
            X_val = fold_scaler.transform(X_val)
            
            svm = LinearSVC(random_state = seed, max_iter = 2000, dual = 'auto')
            svm.fit(X_train, fold_train_labels)
            
            val_preds = svm.predict(X_val)
            acc = accuracy_score(all_labels[fold_val_idx], val_preds)
            f1 = f1_score(all_labels[fold_val_idx], val_preds, average = 'macro')
            prc = precision_score(all_labels[fold_val_idx], val_preds, average = 'macro', zero_division = 0)
            rec = recall_score(all_labels[fold_val_idx], val_preds, average = 'macro', zero_division = 0)
            mcc = matthews_corrcoef(all_labels[fold_val_idx], val_preds)
            
            writer_path = os.path.join(model_path, f'split_{split}')
            writer = SummaryWriter(log_dir = writer_path)
            writer.add_scalar('Accuracy/val', acc, 0)
            writer.add_scalar('F1_Score/val', f1, 0)
            writer.add_scalar('MCC_Score/val', mcc, 0)
            
            joblib.dump({
                'model': svm,
                'scaler': fold_scaler,
                'vectorizer': vectorizer,
                'model_type': 'LinearSVM',
                'split': split,
                'accuracy': acc,
                'f1': f1,
                'precision': prc,
                'recall': rec,
                'mcc': mcc
            }, os.path.join(writer_path, f'split_{split}_best_model.joblib'))
            print(f"Fold {split + 1} Validation — Acc: {acc*100:.2f}%, F1: {f1*100:.2f}%, MCC: {mcc:.4f}")
            
            # Evaluate on held-out test split
            test_texts = [texts[i] for i in test_indices]
            X_text_test = vectorizer.transform(test_texts).toarray()
            X_test = np.concatenate([X_text_test, all_tabular[test_indices]], axis = 1) if all_tabular.shape[1] > 0 else X_text_test
            X_test = fold_scaler.transform(X_test)
            test_preds = svm.predict(X_test)
            
            t_acc = accuracy_score(all_labels[test_indices], test_preds)
            t_f1 = f1_score(all_labels[test_indices], test_preds, average = 'macro')
            t_prc = precision_score(all_labels[test_indices], test_preds, average = 'macro', zero_division = 0)
            t_mcc = matthews_corrcoef(all_labels[test_indices], test_preds)
            t_rec = recall_score(all_labels[test_indices], test_preds, average = 'macro', zero_division = 0)
            test_performance.append([t_acc * 100, t_f1 * 100, t_prc * 100, t_mcc, t_rec * 100])
            writer.close()
            
            if mcc > best_mcc:
                best_mcc, best_acc, best_f1, best_prc, best_rec = mcc, acc, f1, prc, rec
                best_model = {
                    'model': svm,
                    'scaler': fold_scaler,
                    'vectorizer': vectorizer,
                    'model_type': 'LinearSVM',
                    'accuracy': best_acc,
                    'f1': best_f1,
                    'precision': best_prc,
                    'recall': best_rec,
                    'mcc': best_mcc
                }
                
        joblib.dump(best_model, os.path.join(model_path, 'best_model.joblib'))

    # =========================================================================
    # Deep Neural Network Pipelines (ResConsultNet / HTransformer1D)
    # =========================================================================
    else:
        from torch import nn, optim
        from torch.utils.data import DataLoader
        from tokenizers import ByteLevelBPETokenizer
        from transformers import PreTrainedTokenizerFast
        from functions.dataset import EHR
        from functions.runner import train, test, validate
        from functions.model_htransformer import HTransformer1D
        from functions.model_resnet import ResConsultNet

        # Resolve tokenizer file paths
        if tokenizer_dir is not None:
            if not os.path.isabs(tokenizer_dir):
                tokenizer_dir = os.path.join(script_dir, tokenizer_dir)
            tokenizer_vocab_path = os.path.join(tokenizer_dir, 'vocab.json')
            tokenizer_merges_path = os.path.join(tokenizer_dir, 'merges.txt')
        else:
            if tokenizer_vocab_path is None or tokenizer_merges_path is None:
                raise ValueError("Tokenizer files must be specified using --tokenizer_dir or --tokenizer_vocab_path and --tokenizer_merges_path.")
            if not os.path.isabs(tokenizer_vocab_path):
                tokenizer_vocab_path = os.path.join(script_dir, tokenizer_vocab_path)
            if not os.path.isabs(tokenizer_merges_path):
                tokenizer_merges_path = os.path.join(script_dir, tokenizer_merges_path)
            
        if not os.path.exists(tokenizer_vocab_path) or not os.path.exists(tokenizer_merges_path):
            raise FileNotFoundError(f"Tokenizer files not found at '{tokenizer_vocab_path}' or '{tokenizer_merges_path}'.")
        
        tokenizer = ByteLevelBPETokenizer(vocab = tokenizer_vocab_path, merges = tokenizer_merges_path, add_prefix_space = True)
        tokenizer = PreTrainedTokenizerFast(tokenizer_object = tokenizer._tokenizer)
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        num_tokens = tokenizer.vocab_size

        print("Tokenizing clinical text corpus once up front...")
        tokenized_all = tokenizer(texts, truncation = True, padding = 'max_length', max_length = tokens_len, return_tensors = 'pt')
        all_tokens = tokenized_all['input_ids'].long()
        all_masks = tokenized_all['attention_mask'].bool()
        
        criterion = nn.CrossEntropyLoss()
        
        for split, (train_rel_idx, val_rel_idx) in enumerate(skf.split(trainval_indices, labels_trainval)):
            print(f"\n{'='*20} Fold {split + 1}/{n_splits} {'='*20}")
            fold_train_idx = trainval_indices[train_rel_idx]
            fold_val_idx = trainval_indices[val_rel_idx]
            
            if oversampler:
                oversampled_idx, fold_train_labels = oversampler.fit_resample(fold_train_idx.reshape(-1, 1), all_labels[fold_train_idx])
                fold_train_idx = oversampled_idx.flatten()
            else:
                fold_train_labels = all_labels[fold_train_idx]
                
            fold_scaler = StandardScaler()
            scaled_train_tab = fold_scaler.fit_transform(all_tabular[fold_train_idx])
            scaled_val_tab = fold_scaler.transform(all_tabular[fold_val_idx])
            
            dataset_train = EHR(tabular = scaled_train_tab, tokens = all_tokens[fold_train_idx], masks = all_masks[fold_train_idx], labels = fold_train_labels)
            dataset_val = EHR(tabular = scaled_val_tab, tokens = all_tokens[fold_val_idx], masks = all_masks[fold_val_idx], labels = all_labels[fold_val_idx])
            
            dataloader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = True, pin_memory = (device == 'cuda'))
            dataloader_val = DataLoader(dataset_val, batch_size = batch_size * 2, shuffle = False, pin_memory = (device == 'cuda'))
            
            if model_name == 'HTransformer':
                model = HTransformer1D(
                    depth = 3 if pooling == 'avg' else 6,
                    dim_head = 32,
                    ff_mult = 4,
                    heads = 4 if pooling == 'avg' else 8,
                    num_classes = num_classes,
                    max_seq_len = tokens_len,
                    block_size = 32,
                    embedding_dim = embedding_dim,
                    num_tokens = num_tokens,
                    pooling = pooling
                )
            elif model_name == 'Resnet':
                model = ResConsultNet(vocab_size = num_tokens, resnet_layers = [2, 2, 2, 2], pool_size = 4, max_seq_len = tokens_len, num_classes = num_classes, base_filters = 16)
                
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay, amsgrad = amsgrad)
            writer = SummaryWriter(log_dir = os.path.join(model_path, f'split_{split}'))
            
            split_best_epoch = -1
            split_best_mcc, split_best_acc, split_best_f1, split_best_prc, split_best_rec = -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf')
            split_best_model = None
            
            for epoch in range(epochs):
                train(model, dataloader_train, optimizer, criterion, device, epoch, writer)
                model, model_acc, model_f1, model_prc, model_mcc, model_rec = validate(model, dataloader_val, device, epoch, criterion, writer)
                if model_mcc > split_best_mcc:
                    split_best_model, split_best_epoch = model, epoch
                    split_best_acc, split_best_f1, split_best_prc, split_best_rec, split_best_mcc = model_acc, model_f1, model_prc, model_rec, model_mcc
                    
            joblib.dump({
                'model': split_best_model,
                'split': split,
                'epoch': split_best_epoch,
                'accuracy': split_best_acc,
                'f1': split_best_f1,
                'precision': split_best_prc,
                'recall': split_best_rec,
                'mcc': split_best_mcc
            }, os.path.join(writer.log_dir, f'split_{split}_best_model.joblib'))
            print(f"Fold {split + 1} Best Epoch: {split_best_epoch} (MCC: {split_best_mcc:.4f})")
            
            # Evaluate on test split
            scaled_test_tab = fold_scaler.transform(all_tabular[test_indices])
            dataset_test = EHR(tabular = scaled_test_tab, tokens = all_tokens[test_indices], masks = all_masks[test_indices], labels = all_labels[test_indices])
            dataloader_test = DataLoader(dataset_test, batch_size = batch_size * 2, shuffle = False, pin_memory = (device == 'cuda'))
            
            model, model_acc, model_f1, model_prc, model_mcc, model_rec = test(split_best_model, dataloader_test, device, epoch, writer)
            test_performance.append([model_acc * 100, model_f1 * 100, model_prc * 100, model_mcc, model_rec * 100])
            writer.close()
            
            if model_mcc > best_mcc:
                best_model, best_acc, best_f1, best_prc, best_rec, best_mcc = model, model_acc, model_f1, model_prc, model_rec, model_mcc
                
        joblib.dump({
            'model': best_model,
            'accuracy': best_acc,
            'f1': best_f1,
            'precision': best_prc,
            'recall': best_rec,
            'mcc': best_mcc
        }, os.path.join(model_path, 'best_model.joblib'))

    test_performance = np.array(test_performance)
    test_performance_avg = test_performance.mean(axis = 0)
    test_performance_std = np.std(test_performance, axis = 0)
    
    performance_path = os.path.join(model_path, 'performance_parameters.txt')
    with open(performance_path, 'w', encoding = 'utf-8') as file:
        file.write("Average Test Metrics across Folds\n")
        file.write(f"- Accuracy:  {test_performance_avg[0]:.2f}% (+/- {test_performance_std[0]:.2f}%)\n")
        file.write(f"- F1:        {test_performance_avg[1]:.2f}% (+/- {test_performance_std[1]:.2f}%)\n")
        file.write(f"- Precision: {test_performance_avg[2]:.2f}% (+/- {test_performance_std[2]:.2f}%)\n")
        file.write(f"- MCC:       {test_performance_avg[3]:.3f} (+/- {test_performance_std[3]:.3f})\n")
        file.write(f"- Recall:    {test_performance_avg[4]:.2f}% (+/- {test_performance_std[4]:.2f}%)\n")
        file.write("\nConfiguration Parameters:\n")
        file.write(f"Model: {model_name}\n")
        file.write(f"Pooling: {pooling}\n")
        file.write(f"Learning rate: {learning_rate}\n")
        file.write(f"Weight decay: {weight_decay}\n")
        file.write(f"Batch size: {batch_size}\n")
        file.write(f"Epochs: {epochs}\n")
        file.write(f"N splits: {n_splits}\n")
        file.write(f"Oversample minority ratio: {desired_percentage}\n")
        file.write(f"Tabular features count: {len(tabular_features)}\n")
        file.write(f"Unstructured features count: {len(unstructured_features)}\n")
        
    print("\n" + "="*50)
    print("Final Average Cross-Validation Test Performance:")
    print(f"  Accuracy:  {test_performance_avg[0]:.2f}% (+/- {test_performance_std[0]:.2f}%)")
    print(f"  F1:        {test_performance_avg[1]:.2f}% (+/- {test_performance_std[1]:.2f}%)")
    print(f"  Precision: {test_performance_avg[2]:.2f}% (+/- {test_performance_std[2]:.2f}%)")
    print(f"  MCC:       {test_performance_avg[3]:.3f} (+/- {test_performance_std[3]:.3f})")
    print(f"  Recall:    {test_performance_avg[4]:.2f}% (+/- {test_performance_std[4]:.2f}%)")
    print(f"Results saved to: {model_path}")
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = "Automatic CVRM Classifier - Training & Evaluation Pipeline",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    
    group_model = parser.add_argument_group("Model Options")
    group_model.add_argument("--model_name", type = str, default = 'Resnet', choices = ['Resnet', 'HTransformer', 'LinearSVM'], help = "Architecture")
    group_model.add_argument("--pooling", type = str, default = 'cls', choices = ['cls', 'avg'], help = "Pooling strategy for HTransformer")
    group_model.add_argument("--tokens_len", type = int, default = 8192, help = "Max token sequence length")
    group_model.add_argument("--embedding_dim", type = int, default = 512, help = "Token embedding dimension")
    
    group_train = parser.add_argument_group("Training Options")
    group_train.add_argument("--epochs", type = int, default = 30, help = "Number of epochs per fold")
    group_train.add_argument("--batch_size", type = int, default = 32, help = "Batch size for training")
    group_train.add_argument("--learning_rate", type = float, default = 3e-5, help = "Learning rate")
    group_train.add_argument("--weight_decay", type = float, default = 1e-4, help = "Weight decay")
    group_train.add_argument("--n_splits", type = int, default = 5, help = "Number of cross-validation folds")
    group_train.add_argument("--test_size", type = float, default = 0.2, help = "Test set fraction")
    group_train.add_argument("--desired_percentage", type = float, default = 0.3, help = "Minority class resampling proportion")
    group_train.add_argument("--seed", type = int, default = 42, help = "Random seed for reproducibility")
    
    group_env = parser.add_argument_group("Paths & Environment")
    group_env.add_argument("--dataset_path", type = str, required = True, help = "Path to input dataset file (.parquet or .csv)")
    group_env.add_argument("--text_col", type = str, default = None, help = "Name of text column (auto-detected if None)")
    group_env.add_argument("--tokenizer_dir", type = str, default = None, help = "Directory containing tokenizer vocab.json and merges.txt")
    group_env.add_argument("--tokenizer_vocab_path", type = str, default = None, help = "Path to tokenizer vocab.json")
    group_env.add_argument("--tokenizer_merges_path", type = str, default = None, help = "Path to tokenizer merges.txt")
    group_env.add_argument("--results_folder", type = str, default = 'results', help = "Directory to store outputs")
    group_env.add_argument("--device", type = str, default = 'auto', choices = ['auto', 'cuda', 'cpu'], help = "Compute device")
    
    args = parser.parse_args()
    main(**args.__dict__)
