import os
import sys
import argparse
from pathlib import Path


def safe_load_checkpoint(checkpoint_path: str, device: str = 'cpu'):
    """Load joblib or PyTorch checkpoint, mapping storage safely to target device."""
    import joblib
    import torch
    
    orig_load = torch.load
    target_device = torch.device(device)
    
    def device_mapped_load(*args, **kwargs):
        kwargs['map_location'] = target_device
        return orig_load(*args, **kwargs)
        
    torch.load = device_mapped_load
    try:
        data = joblib.load(checkpoint_path)
    finally:
        torch.load = orig_load
        
    return data


def get_expected_tabular_dim(model) -> int:
    """Infer expected tabular feature dimension from model architecture."""
    if hasattr(model, 'resnet') and hasattr(model.resnet, 'neural_net'):
        return max(0, model.resnet.neural_net[0].in_features - 1024)
    elif hasattr(model, 'mlp') and hasattr(model, 'token_emb'):
        return max(0, model.mlp[0].in_features - model.token_emb.embedding_dim)
    return 0


def predict_single(
    text: str,
    tabular_features,
    model,
    tokenizer = None,
    tokens_len: int = 8192,
    device: str = 'cpu'
):
    """Run inference for a single clinical consultation sample (neural or linear model)."""
    import numpy as np
    
    # Linear SVM Checkpoint
    if isinstance(model, dict) and ('vectorizer' in model or model.get('model_type') == 'LinearSVM'):
        vec = model['vectorizer']
        scaler = model['scaler']
        svm = model['model']
        
        text_feat = vec.transform([text]).toarray()
        expected_tab = max(0, scaler.mean_.shape[0] - text_feat.shape[1])
        if tabular_features is None:
            tab_feat = np.zeros((1, expected_tab), dtype = np.float32)
        else:
            tab_feat = np.asarray(tabular_features, dtype = np.float32)
            if tab_feat.ndim == 1:
                tab_feat = np.expand_dims(tab_feat, 0)
            if tab_feat.shape[1] < expected_tab:
                pad = np.zeros((1, expected_tab - tab_feat.shape[1]), dtype = np.float32)
                tab_feat = np.concatenate([tab_feat, pad], axis = 1)
            elif tab_feat.shape[1] > expected_tab:
                tab_feat = tab_feat[:, :expected_tab]
                
        X = np.concatenate([text_feat, tab_feat], axis = 1) if expected_tab > 0 else text_feat
        X_scaled = scaler.transform(X)
        pred_class = int(svm.predict(X_scaled)[0])
        margin = float(svm.decision_function(X_scaled)[0])
        prob_1 = float(1.0 / (1.0 + np.exp(-margin)))
        prob_0 = 1.0 - prob_1
        probs = [prob_0, prob_1]
        confidence = float(probs[pred_class])
        return pred_class, confidence, probs

    # PyTorch Neural Model
    import torch
    import torch.nn.functional as F

    tokens_dict = tokenizer(
        text,
        truncation = True,
        padding = 'max_length',
        max_length = tokens_len,
        return_tensors = 'pt'
    )
    input_ids = tokens_dict['input_ids'].to(device)
    attention_mask = tokens_dict['attention_mask'].bool().to(device)
    
    expected_tab = get_expected_tabular_dim(model)
    if tabular_features is None:
        tabular_tensor = torch.zeros((1, expected_tab), dtype = torch.float32, device = device)
    else:
        tab_arr = np.asarray(tabular_features, dtype = np.float32)
        if tab_arr.ndim == 1:
            tab_arr = np.expand_dims(tab_arr, 0)
        if tab_arr.shape[1] < expected_tab:
            pad = np.zeros((tab_arr.shape[0], expected_tab - tab_arr.shape[1]), dtype = np.float32)
            tab_arr = np.concatenate([tab_arr, pad], axis = 1)
        elif tab_arr.shape[1] > expected_tab:
            tab_arr = tab_arr[:, :expected_tab]
        tabular_tensor = torch.tensor(tab_arr, dtype = torch.float32, device = device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, tabular_tensor, attention_mask)
        probs = F.softmax(outputs, dim = 1).squeeze(0).cpu().numpy()
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

    return pred_class, confidence, probs


def predict_batch(
    texts: list,
    tabular_features,
    model,
    tokenizer = None,
    tokens_len: int = 8192,
    batch_size: int = 32,
    device: str = 'cpu'
):
    """Execute batched inference over clinical texts for accelerated throughput."""
    import numpy as np

    # Linear SVM Batched Inference
    if isinstance(model, dict) and ('vectorizer' in model or model.get('model_type') == 'LinearSVM'):
        vec = model['vectorizer']
        scaler = model['scaler']
        svm = model['model']
        
        cleaned_texts = [str(t) if t is not None and not (isinstance(t, float) and np.isnan(t)) else "" for t in texts]
        text_feats = vec.transform(cleaned_texts).toarray()
        expected_tab = max(0, scaler.mean_.shape[0] - text_feats.shape[1])
        
        if tabular_features is None or expected_tab == 0:
            tab_feats = np.zeros((len(cleaned_texts), expected_tab), dtype = np.float32)
        else:
            tab_feats = np.asarray(tabular_features, dtype = np.float32)
            if tab_feats.shape[1] < expected_tab:
                pad = np.zeros((len(cleaned_texts), expected_tab - tab_feats.shape[1]), dtype = np.float32)
                tab_feats = np.concatenate([tab_feats, pad], axis = 1)
            elif tab_feats.shape[1] > expected_tab:
                tab_feats = tab_feats[:, :expected_tab]
                
        X = np.concatenate([text_feats, tab_feats], axis = 1) if expected_tab > 0 else text_feats
        X_scaled = scaler.transform(X)
        preds = svm.predict(X_scaled).astype(int).tolist()
        margins = svm.decision_function(X_scaled)
        p1 = 1.0 / (1.0 + np.exp(-margins))
        p0 = 1.0 - p1
        probs = np.stack([p0, p1], axis = 1)
        confs = np.max(probs, axis = 1).tolist()
        return preds, confs, probs.tolist()

    # PyTorch Neural Batched Inference
    import torch
    import torch.nn.functional as F

    model.eval()
    expected_tab = get_expected_tabular_dim(model)
    total_samples = len(texts)
    all_preds, all_confs, all_probs = [], [], []

    with torch.no_grad():
        for start_idx in range(0, total_samples, batch_size):
            batch_texts = [str(t) if t is not None and not (isinstance(t, float) and np.isnan(t)) else "" for t in texts[start_idx:start_idx + batch_size]]
            tokenized = tokenizer(
                batch_texts,
                truncation = True,
                padding = 'max_length',
                max_length = tokens_len,
                return_tensors = 'pt'
            )
            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].bool().to(device)
            bsz = input_ids.shape[0]

            if tabular_features is None:
                tab_tensor = torch.zeros((bsz, expected_tab), dtype = torch.float32, device = device)
            else:
                tab_batch = np.asarray(tabular_features[start_idx:start_idx + batch_size], dtype = np.float32)
                if tab_batch.shape[1] < expected_tab:
                    pad = np.zeros((bsz, expected_tab - tab_batch.shape[1]), dtype = np.float32)
                    tab_batch = np.concatenate([tab_batch, pad], axis = 1)
                elif tab_batch.shape[1] > expected_tab:
                    tab_batch = tab_batch[:, :expected_tab]
                tab_tensor = torch.tensor(tab_batch, dtype = torch.float32, device = device)

            outputs = model(input_ids, tab_tensor, attention_mask)
            probs = F.softmax(outputs, dim = 1).cpu().numpy()
            preds = np.argmax(probs, axis = 1)
            confs = probs[np.arange(bsz), preds]

            all_preds.extend(preds.tolist())
            all_confs.extend(confs.tolist())
            all_probs.extend(probs.tolist())

    return all_preds, all_confs, all_probs


def main():
    parser = argparse.ArgumentParser(
        description = "Automatic CVRM Classifier - Model Inference CLI",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    
    input_group = parser.add_argument_group("Input Data")
    input_group.add_argument("--text", type = str, default = None, help = "Input clinical consultation note text")
    input_group.add_argument("--text_file", type = str, default = None, help = "Path to text file containing clinical note")
    input_group.add_argument("--input_file", type = str, default = None, help = "Path to tabular CSV/Parquet file for batch prediction")
    input_group.add_argument("--text_col", type = str, default = None, help = "Name of text column in batch file (auto-detected if None)")
    input_group.add_argument("--output_file", type = str, default = "predictions.csv", help = "Output path for batch predictions")
    input_group.add_argument("--batch_size", type = int, default = 32, help = "Batch size for batch inference")
    
    model_group = parser.add_argument_group("Model & Checkpoint")
    model_group.add_argument("--checkpoint", type = str, required = True, help = "Path to model checkpoint (.joblib or .pt)")
    model_group.add_argument("--tokens_len", type = int, default = 8192, help = "Max token sequence length (neural models)")
    model_group.add_argument("--tokenizer_dir", type = str, default = None, help = "Directory containing tokenizer files (required for neural models)")
    model_group.add_argument("--device", type = str, default = "auto", choices = ["auto", "cuda", "cpu"], help = "Compute device")
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    sys.path.insert(0, script_dir)

    text_input = args.text
    if args.text_file:
        if not os.path.isabs(args.text_file):
            args.text_file = os.path.join(script_dir, args.text_file)
        if not os.path.exists(args.text_file):
            raise FileNotFoundError(f"Text file not found: {args.text_file}")
        with open(args.text_file, "r", encoding = "utf-8") as f:
            text_input = f.read()

    if text_input is None and args.input_file is None:
        parser.error("Either --text, --text_file, or --input_file must be specified.")

    import pandas as pd
    device = args.device
    if device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(script_dir, ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print(f"Loading checkpoint from: {ckpt_path} on {device}")
    ckpt = safe_load_checkpoint(ckpt_path, device = device)
    
    is_svm = isinstance(ckpt, dict) and ('vectorizer' in ckpt or ckpt.get('model_type') == 'LinearSVM')
    tokenizer = None
    model = ckpt

    if is_svm:
        print("Detected LinearSVM model pipeline.")
    else:
        if args.tokenizer_dir is None:
            parser.error("--tokenizer_dir is required for neural network checkpoints.")
        if not os.path.isabs(args.tokenizer_dir):
            args.tokenizer_dir = os.path.join(script_dir, args.tokenizer_dir)
            
        vocab_path = os.path.join(args.tokenizer_dir, "vocab.json")
        merges_path = os.path.join(args.tokenizer_dir, "merges.txt")
        if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
            raise FileNotFoundError(f"Tokenizer files not found in {args.tokenizer_dir}")

        from tokenizers import ByteLevelBPETokenizer
        from transformers import PreTrainedTokenizerFast
        bpe = ByteLevelBPETokenizer(vocab = vocab_path, merges = merges_path, add_prefix_space = True)
        tokenizer = PreTrainedTokenizerFast(tokenizer_object = bpe._tokenizer)
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        
        model = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        if hasattr(model, 'to'):
            model = model.to(device)

    # Single text inference
    if text_input is not None:
        pred_class, conf, probs = predict_single(
            text = text_input,
            tabular_features = None,
            model = model,
            tokenizer = tokenizer,
            tokens_len = args.tokens_len,
            device = device
        )
        class_label = "CVRM Eligible (1)" if pred_class == 1 else "Not Eligible (0)"
        print("\n" + "="*50)
        print("Inference Result:")
        print(f"  Predicted Class: {class_label}")
        print(f"  Confidence:      {conf * 100:.2f}%")
        print(f"  Class 0 Prob:    {probs[0] * 100:.2f}%")
        print(f"  Class 1 Prob:    {probs[1] * 100:.2f}%")
        print("="*50)

    # Batched file inference
    elif args.input_file is not None:
        in_path = args.input_file
        if not os.path.isabs(in_path):
            in_path = os.path.join(script_dir, in_path)

        df = pd.read_parquet(in_path) if in_path.endswith(".parquet") else pd.read_csv(in_path)
        if args.text_col and args.text_col in df.columns:
            text_col = args.text_col
        else:
            text_cols = df.select_dtypes(include = ['object', 'string']).columns
            if len(text_cols) == 0:
                raise ValueError("No text column found in input file.")
            text_col = text_cols[0]
            
        print(f"Running batched prediction on {len(df)} samples using column '{text_col}' (batch_size={args.batch_size})...")
        preds, confs, _ = predict_batch(
            texts = df[text_col].tolist(),
            tabular_features = None,
            model = model,
            tokenizer = tokenizer,
            tokens_len = args.tokens_len,
            batch_size = args.batch_size,
            device = device
        )
        df['cvrm_prediction'] = preds
        df['cvrm_confidence'] = confs
        out_path = args.output_file
        if not os.path.isabs(out_path):
            out_path = os.path.join(script_dir, out_path)
        df.to_csv(out_path, index = False)
        print(f"Batch predictions saved to: {out_path}")


if __name__ == '__main__':
    main()
