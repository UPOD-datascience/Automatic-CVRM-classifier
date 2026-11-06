import os
import random
import argparse
from datetime import datetime as dt


def set_seed(seed: int) -> None:
    """Set random seed across Python, NumPy, and PyTorch for reproducibility."""
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
        tokenizer_dir           : str   = None,
        tokenizer_vocab_path    : str   = None,
        tokenizer_merges_path   : str   = None,
        device                  : str   = 'auto',
        seed                    : int   = 42
    ):
    
    # Lazy imports for fast CLI response
    import numpy as np
    import pandas as pd
    import joblib
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from imblearn.over_sampling import RandomOverSampler

    from functions.dataset import EHR
    from functions.runner import train, test, validate
    from functions.model_htransformer import (
        HTransformer1D_custom_averagePooling,
        HTransformer1D_custom_clsPooling,
    )
    from functions.model_resnet import ResConsultNet

    set_seed(seed)
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Resolve tokenizer paths
    if tokenizer_dir is not None:
        if not os.path.isabs(tokenizer_dir):
            tokenizer_dir = os.path.join(script_dir, tokenizer_dir)
        tokenizer_vocab_path = os.path.join(tokenizer_dir, 'vocab.json')
        tokenizer_merges_path = os.path.join(tokenizer_dir, 'merges.txt')
    else:
        if tokenizer_vocab_path is None or tokenizer_merges_path is None:
            raise ValueError(
                "Tokenizer files must be specified using either --tokenizer_dir (containing vocab.json and merges.txt) "
                "or both --tokenizer_vocab_path and --tokenizer_merges_path."
            )
        if not os.path.isabs(tokenizer_vocab_path):
            tokenizer_vocab_path = os.path.join(script_dir, tokenizer_vocab_path)
        if not os.path.isabs(tokenizer_merges_path):
            tokenizer_merges_path = os.path.join(script_dir, tokenizer_merges_path)
        
    if not os.path.exists(tokenizer_vocab_path) or not os.path.exists(tokenizer_merges_path):
        raise FileNotFoundError(
            f"Tokenizer files not found at '{tokenizer_vocab_path}' or '{tokenizer_merges_path}'. "
            "Please specify valid paths via --tokenizer_dir or --tokenizer_vocab_path / --tokenizer_merges_path."
        )
    
    # Initialize BPE tokenizer
    tokenizer = ByteLevelBPETokenizer(
        vocab = tokenizer_vocab_path,
        merges = tokenizer_merges_path,
        add_prefix_space = True
    )
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_object = tokenizer._tokenizer)
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    
    scaler = StandardScaler()
    num_tokens = tokenizer.vocab_size
    
    if dataset_path is None:
        raise ValueError("Please provide a dataset parquet file via --dataset_path.")
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(script_dir, dataset_path)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at '{dataset_path}'.")
        
    df = pd.read_parquet(dataset_path)
    
    labels = df.iloc[:, -1]
    df = df.iloc[:, 1:-1]
    class_counts = labels.value_counts()
    
    majority_class = class_counts.max()
    desired_minority_samples = int(desired_percentage * len(labels))
    
    alpha = desired_minority_samples / majority_class
    if desired_percentage:
        oversampler = RandomOverSampler(sampling_strategy = alpha, random_state = seed)
    else:
        oversampler = None
    
    tabular_features = df.select_dtypes(include=['number']).columns.tolist()
    unstructured_features = df.select_dtypes(include=['object']).columns.tolist()
    
    indices = np.arange(df.shape[0])
    trainval_indices, test_indices = train_test_split(
        indices,
        test_size = test_size, 
        stratify = labels, 
        random_state = seed
    )
    
    df_test = df.iloc[test_indices]
    labels_test = labels.iloc[test_indices]
    
    df_trainval = df.iloc[trainval_indices]
    labels_trainval = labels.iloc[trainval_indices]
    
    # Configure experiment log directory
    timestamp = dt.now().strftime("%Y_%m_%d_%H_%M")
    if model_name == 'HTransformer':
        log_dir = os.path.join('HTransformer1D', f'{timestamp}_{pooling.upper()}')
    elif model_name == 'Resnet':        
        log_dir = os.path.join('ResConsultNet', timestamp)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    criterion = nn.CrossEntropyLoss()
    
    best_acc   = -float('inf')
    best_f1    = -float('inf')
    best_prc   = -float('inf')
    best_rec   = -float('inf')
    best_mcc   = -float('inf')
    best_model = None
    
    skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = seed)
    
    results_path = os.path.join(script_dir, results_folder)
    os.makedirs(results_path, exist_ok = True)
    
    model_path = os.path.join(results_path, log_dir)
    os.makedirs(model_path, exist_ok = True)
    
    test_performance = []
    
    # Stratified K-fold Cross Validation
    for split, (train_indices, val_indices) in enumerate(skf.split(df_trainval, labels_trainval)):
        print(f"\n{'='*20} Fold {split + 1}/{n_splits} {'='*20}")
        
        if model_name == 'HTransformer':
            if pooling == 'avg':
                model = HTransformer1D_custom_averagePooling(
                    depth = 3,
                    dim_head = 32,
                    ff_mult = 4,
                    heads = 4,
                    num_classes = num_classes,
                    max_seq_len = tokens_len,
                    block_size = 32,
                    embedding_dim = embedding_dim,
                    num_tokens = num_tokens
                )
            else:
                model = HTransformer1D_custom_clsPooling(
                    depth = 6,
                    dim_head = 32,
                    ff_mult = 4,
                    heads = 8,
                    num_classes = num_classes,
                    max_seq_len = tokens_len,
                    block_size = 32,
                    embedding_dim = embedding_dim,
                    num_tokens = num_tokens
                )
        elif model_name == 'Resnet':
            model = ResConsultNet(
                vocab_size = num_tokens,
                resnet_layers = [2, 2, 2, 2],
                pool_size = 4,
                max_seq_len = tokens_len,
                num_classes = num_classes,
                base_filters = 16
            )
            
        model.to(device)
        
        optimizer = optim.Adam(
            model.parameters(),
            lr = learning_rate,
            weight_decay = weight_decay,
            amsgrad = amsgrad
        )
        
        writer_path = os.path.join(model_path, f'split_{split}')
        writer = SummaryWriter(log_dir = writer_path)
        
        df_train = df_trainval.iloc[train_indices]
        labels_train = labels_trainval.iloc[train_indices]
        
        if oversampler:
            df_train, labels_train = oversampler.fit_resample(df_train, labels_train)
            
        df_val = df_trainval.iloc[val_indices]
        labels_val = labels_trainval.iloc[val_indices]
        
        dataset_train = EHR(
            df = df_train,
            labels = labels_train,
            tokens_len = tokens_len,
            tabular_features = tabular_features,
            scaler = scaler,
            unstructured_features = unstructured_features,
            tokenizer = tokenizer
        )
        
        dataset_val = EHR(
            df = df_val,
            labels = labels_val,
            tokens_len = tokens_len,
            tabular_features = tabular_features,
            scaler = scaler,
            unstructured_features = unstructured_features,
            tokenizer = tokenizer
        )
        
        dataloader_train = DataLoader(
            dataset_train,
            batch_size = batch_size,
            shuffle = True,
            pin_memory = (device == 'cuda')
        )
        
        dataloader_val = DataLoader(
            dataset_val,
            batch_size = batch_size * 2,
            shuffle = False,
            pin_memory = (device == 'cuda')
        )
        
        split_best_epoch = -1
        split_best_mcc   = -float('inf')
        split_best_acc   = -float('inf')
        split_best_f1    = -float('inf')
        split_best_prc   = -float('inf')
        split_best_rec   = -float('inf')
        split_best_model = None
        
        for epoch in range(epochs):
            train(
                model = model,
                dataloader = dataloader_train,
                optimizer = optimizer,
                criterion = criterion,
                device = device,
                epoch = epoch,
                writer = writer
            )
            
            model, model_acc, model_f1, model_prc, model_mcc, model_rec = validate(
                model = model,
                dataloader = dataloader_val,
                criterion = criterion,
                device = device,
                epoch = epoch,
                writer = writer
            )
            
            if model_mcc > split_best_mcc:
                split_best_model = model
                split_best_epoch = epoch
                split_best_acc   = model_acc
                split_best_f1    = model_f1
                split_best_prc   = model_prc
                split_best_rec   = model_rec
                split_best_mcc   = model_mcc
                
        joblib.dump({
                'model'     : split_best_model,
                'split'     : split,
                'epoch'     : split_best_epoch,
                'accuracy'  : split_best_acc,
                'f1'        : split_best_f1,
                'precision' : split_best_prc,
                'recall'    : split_best_rec,
                'mcc'       : split_best_mcc
            },
            os.path.join(writer.log_dir, f'split_{split}_best_model.joblib')
        )
        
        print(f"Fold {split + 1} Best Epoch: {split_best_epoch} (MCC: {split_best_mcc:.4f})")
        
        # Evaluate on held-out test set
        dataset_test = EHR(
            df = df_test,
            labels = labels_test,
            tabular_features = tabular_features,
            scaler = scaler,
            unstructured_features = unstructured_features,
            tokenizer = tokenizer,
            tokens_len = tokens_len
        )
        
        dataloader_test = DataLoader(
            dataset_test,
            batch_size = batch_size * 2,
            shuffle = False,
            pin_memory = (device == 'cuda')
        )
        
        model, model_acc, model_f1, model_prc, model_mcc, model_rec = test(
            model = split_best_model,
            dataloader = dataloader_test,
            device = device,
            epoch = epoch,
            writer = writer
        )
        
        test_performance.append([model_acc * 100, model_f1 * 100, model_prc * 100, model_mcc, model_rec * 100])
        
        if model_mcc > best_mcc:
            best_model = model
            best_acc   = model_acc
            best_f1    = model_f1
            best_prc   = model_prc
            best_rec   = model_rec
            best_mcc   = model_mcc
            
    joblib.dump({
            'model'     : best_model,
            'accuracy'  : best_acc,
            'f1'        : best_f1,
            'precision' : best_prc,
            'recall'    : best_rec,
            'mcc'       : best_mcc
        },
        os.path.join(model_path, 'best_model.joblib')
    )
    
    writer.close()
    
    test_performance = np.array(test_performance)
    test_performance_avg = test_performance.mean(axis=0)
    test_performance_std = np.std(test_performance, axis=0)
    
    performance_path = os.path.join(model_path, 'performance_parameters.txt')
    with open(performance_path, 'w') as file:
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
    group_model.add_argument("--model_name", type=str, default='Resnet', choices=['Resnet', 'HTransformer'], help="Neural architecture")
    group_model.add_argument("--pooling", type=str, default='cls', choices=['cls', 'avg'], help="Pooling strategy for HTransformer")
    group_model.add_argument("--tokens_len", type=int, default=8192, help="Max token sequence length")
    group_model.add_argument("--embedding_dim", type=int, default=512, help="Token embedding dimension")
    
    group_train = parser.add_argument_group("Training Options")
    group_train.add_argument("--epochs", type=int, default=30, help="Number of epochs per fold")
    group_train.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    group_train.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    group_train.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    group_train.add_argument("--n_splits", type=int, default=5, help="Number of cross-validation folds")
    group_train.add_argument("--test_size", type=float, default=0.2, help="Test set fraction")
    group_train.add_argument("--desired_percentage", type=float, default=0.3, help="Minority class resampling proportion")
    group_train.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    group_env = parser.add_argument_group("Paths & Environment")
    group_env.add_argument("--dataset_path", type=str, required=True, help="Path to input dataset parquet file")
    group_env.add_argument("--tokenizer_dir", type=str, default=None, help="Directory containing tokenizer vocab.json and merges.txt")
    group_env.add_argument("--tokenizer_vocab_path", type=str, default=None, help="Path to tokenizer vocab.json")
    group_env.add_argument("--tokenizer_merges_path", type=str, default=None, help="Path to tokenizer merges.txt")
    group_env.add_argument("--results_folder", type=str, default='results', help="Directory to store outputs")
    group_env.add_argument("--device", type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help="Compute device")
    
    args = parser.parse_args()
    main(**args.__dict__)
