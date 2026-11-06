import os
import sys
import argparse
from pathlib import Path


def safe_load_checkpoint(checkpoint_path: str, device: str = 'cpu'):
    """Load joblib checkpoint safely, mapping PyTorch storage to target device."""
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
    tokenizer,
    tokens_len: int = 8192,
    device: str = 'cpu'
):
    """Run inference for a single clinical consultation sample."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    # If scikit-learn model
    if not hasattr(model, 'forward'):
        raise NotImplementedError("For scikit-learn models (LinearSVM), use pipeline evaluation.")

    # PyTorch model inference
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
        tabular_tensor = torch.zeros((1, expected_tab), dtype=torch.float32).to(device)
    else:
        tab_arr = np.array(tabular_features, dtype=np.float32)
        if tab_arr.ndim == 1:
            tab_arr = np.expand_dims(tab_arr, 0)
        # Pad or truncate if necessary to match expected dimension
        if tab_arr.shape[1] < expected_tab:
            pad = np.zeros((tab_arr.shape[0], expected_tab - tab_arr.shape[1]), dtype=np.float32)
            tab_arr = np.concatenate([tab_arr, pad], axis=1)
        elif tab_arr.shape[1] > expected_tab:
            tab_arr = tab_arr[:, :expected_tab]
        tabular_tensor = torch.tensor(tab_arr, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, tabular_tensor, attention_mask)
        probs = F.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

    return pred_class, confidence, probs


def main():
    parser = argparse.ArgumentParser(
        description = "Automatic CVRM Classifier - Model Inference CLI",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    
    input_group = parser.add_argument_group("Input Data")
    input_group.add_argument("--text", type=str, default=None, help="Input clinical consultation note text")
    input_group.add_argument("--text_file", type=str, default=None, help="Path to text file containing clinical note")
    input_group.add_argument("--input_file", type=str, default=None, help="Path to tabular CSV/Parquet file for batch prediction")
    input_group.add_argument("--output_file", type=str, default="predictions.csv", help="Output path for batch predictions")
    
    model_group = parser.add_argument_group("Model & Checkpoint")
    model_group.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint file (.joblib or .pt)")
    model_group.add_argument("--tokens_len", type=int, default=8192, help="Max token sequence length")
    model_group.add_argument("--tokenizer_dir", type=str, required=True, help="Directory containing tokenizer vocab.json and merges.txt")
    model_group.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    sys.path.insert(0, script_dir)
    
    if not os.path.isabs(args.tokenizer_dir):
        args.tokenizer_dir = os.path.join(script_dir, args.tokenizer_dir)

    # Determine text input
    text_input = args.text
    if args.text_file:
        if not os.path.isabs(args.text_file):
            args.text_file = os.path.join(script_dir, args.text_file)
        if not os.path.exists(args.text_file):
            raise FileNotFoundError(f"Text file not found: {args.text_file}")
        with open(args.text_file, "r", encoding="utf-8") as f:
            text_input = f.read()

    if text_input is None and args.input_file is None:
        parser.error("Either --text, --text_file, or --input_file must be specified.")

    # Lazy load heavy dependencies
    import torch
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast
    import pandas as pd

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(script_dir, ckpt_path)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # Load tokenizer
    vocab_path = os.path.join(args.tokenizer_dir, "vocab.json")
    merges_path = os.path.join(args.tokenizer_dir, "merges.txt")
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        raise FileNotFoundError(f"Tokenizer files not found in {args.tokenizer_dir}")

    bpe = ByteLevelBPETokenizer(vocab=vocab_path, merges=merges_path, add_prefix_space=True)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=bpe._tokenizer)
    tokenizer.add_special_tokens({'pad_token': '<pad>'})

    # Load model checkpoint
    print(f"Loading checkpoint from: {ckpt_path} on {device}")
    ckpt = safe_load_checkpoint(ckpt_path, device=device)
    model = ckpt['model']
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

    # Batch file inference
    elif args.input_file is not None:
        in_path = args.input_file
        if not os.path.isabs(in_path):
            in_path = os.path.join(script_dir, in_path)

        if in_path.endswith(".parquet"):
            df = pd.read_parquet(in_path)
        else:
            df = pd.read_csv(in_path)

        text_col = 'consult' if 'consult' in df.columns else df.select_dtypes(include=['object']).columns[0]
        print(f"Running batch prediction on {len(df)} samples using text column '{text_col}'...")

        preds, confs = [], []
        for text in df[text_col]:
            p, c, _ = predict_single(
                text = str(text) if pd.notna(text) else "",
                tabular_features = None,
                model = model,
                tokenizer = tokenizer,
                tokens_len = args.tokens_len,
                device = device
            )
            preds.append(p)
            confs.append(c)

        df['cvrm_prediction'] = preds
        df['cvrm_confidence'] = confs
        out_path = args.output_file
        if not os.path.isabs(out_path):
            out_path = os.path.join(script_dir, out_path)
        df.to_csv(out_path, index=False)
        print(f"Batch predictions saved to: {out_path}")


if __name__ == '__main__':
    main()
