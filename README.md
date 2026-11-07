# Automatic CVRM Classifier

Automatic Cardiovascular Risk Management (CVRM) classification using Electronic Health Records (EHR).

This repository provides deep learning architectures, classical machine learning baselines, and command-line interfaces (CLI) to classify Cardiovascular Risk Management eligibility from multimodal clinical data (unstructured clinical text and tabular variables).

---

## Architectures

- **ResConsultNet**: 1D Convolutional Residual Network tailored for clinical notes and multimodal feature fusion.
- **H-Transformer 1D**: Hierarchical 1D Transformer for long-context clinical sequences supporting CLS and Average pooling.
- **Linear SVM**: Fast baseline Support Vector Classifier combining TF-IDF text features with scaled tabular clinical indicators.

---

## Installation & Setup

This repository uses [uv](https://github.com/astral-sh/uv) for fast, isolated, and reproducible dependency management.

### 1. Install `uv`
If not already installed:
- **Windows (PowerShell)**:
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **Linux / macOS**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### 2. Clone & Sync Dependencies
```bash
git clone https://github.com/UPOD-datascience/Automatic-CVRM-classifier.git
cd Automatic-CVRM-classifier
uv sync
```

---

## Model Inference CLI (`predict.py`)

Run inference on clinical notes directly from the command line using CPU or GPU.

### 1. Single Consultation Note Inference

Run prediction for a text snippet:
```bash
# Neural model checkpoint (requires tokenizer)
uv run python predict.py \
    --checkpoint path/to/checkpoint.joblib \
    --tokenizer_dir path/to/tokenizer_dir \
    --text "Patient presents with cardiovascular symptoms and hypertension."

# Linear SVM checkpoint (no tokenizer required)
uv run python predict.py \
    --checkpoint path/to/svm_checkpoint.joblib \
    --text "Patient presents with cardiovascular symptoms and hypertension."
```

Run prediction from a text file:
```bash
uv run python predict.py \
    --checkpoint path/to/checkpoint.joblib \
    --tokenizer_dir path/to/tokenizer_dir \
    --text_file path/to/consultation.txt
```

Example output:
```text
==================================================
Inference Result:
  Predicted Class: CVRM Eligible (1)
  Confidence:      92.40%
  Class 0 Prob:    7.60%
  Class 1 Prob:    92.40%
==================================================
```

### 2. Batch Inference from File

Classify a tabular CSV or Parquet file containing consultation notes:
```bash
uv run python predict.py \
    --checkpoint path/to/checkpoint.joblib \
    --tokenizer_dir path/to/tokenizer_dir \
    --input_file path/to/records.parquet \
    --output_file path/to/predictions.csv \
    --batch_size 64
```

### Inference Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--checkpoint` | `str` | *(Required)* | Path to trained model checkpoint (`.joblib` or `.pt`) |
| `--tokenizer_dir` | `str` | `None` | Directory containing `vocab.json` and `merges.txt` (required for neural models) |
| `--text` | `str` | `None` | Input clinical consultation note string |
| `--text_file` | `str` | `None` | Path to text file containing consultation note |
| `--input_file` | `str` | `None` | Path to CSV/Parquet file for batch prediction |
| `--text_col` | `str` | `None` | Text column name in batch input file (auto-detected if None) |
| `--output_file` | `str` | `predictions.csv` | Output destination for batch predictions |
| `--batch_size` | `int` | `32` | Batch size for accelerated batch inference |
| `--tokens_len` | `int` | `8192` | Maximum token sequence length for neural models |
| `--device` | `str` | `auto` | Compute device (`auto`, `cuda`, `cpu`) |

---

## Training Pipeline CLI (`train.py`)

Train and cross-validate neural models or baseline linear classifiers on multimodal tabular and text datasets:

```bash
# Train ResConsultNet
uv run python train.py \
    --model_name Resnet \
    --dataset_path path/to/dataset.parquet \
    --tokenizer_dir path/to/tokenizer_dir \
    --epochs 30 \
    --batch_size 32

# Train H-Transformer 1D (CLS pooling)
uv run python train.py \
    --model_name HTransformer \
    --pooling cls \
    --dataset_path path/to/dataset.parquet \
    --tokenizer_dir path/to/tokenizer_dir \
    --epochs 30 \
    --batch_size 16

# Train H-Transformer 1D (Average pooling)
uv run python train.py \
    --model_name HTransformer \
    --pooling avg \
    --dataset_path path/to/dataset.parquet \
    --tokenizer_dir path/to/tokenizer_dir \
    --epochs 30 \
    --batch_size 16

# Train Linear SVM baseline (no tokenizer needed)
uv run python train.py \
    --model_name LinearSVM \
    --dataset_path path/to/dataset.parquet
```

### Training Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--dataset_path` | `str` | *(Required)* | Path to dataset file (`.parquet` or `.csv`) |
| `--model_name` | `str` | `Resnet` | Architecture (`Resnet`, `HTransformer`, `LinearSVM`) |
| `--text_col` | `str` | `None` | Text column name (auto-detected if None) |
| `--pooling` | `str` | `cls` | Pooling for HTransformer (`cls`, `avg`) |
| `--tokenizer_dir` | `str` | `None` | Directory containing `vocab.json` and `merges.txt` (neural models) |
| `--tokenizer_vocab_path`| `str` | `None` | Direct path to tokenizer `vocab.json` |
| `--tokenizer_merges_path`| `str` | `None` | Direct path to tokenizer `merges.txt` |
| `--tokens_len` | `int` | `8192` | Maximum token sequence length (neural models) |
| `--embedding_dim` | `int` | `512` | Token embedding dimension |
| `--epochs` | `int` | `30` | Number of epochs per fold |
| `--batch_size` | `int` | `32` | Training batch size |
| `--learning_rate` | `float` | `3e-5` | Learning rate |
| `--weight_decay` | `float` | `1e-4` | Weight decay |
| `--n_splits` | `int` | `5` | Stratified cross-validation folds |
| `--results_folder` | `str` | `results` | Directory to save checkpoints and logs |
| `--device` | `str` | `auto` | Compute device (`auto`, `cuda`, `cpu`) |

---

## Repository Structure

```text
Automatic-CVRM-classifier/
├── functions/                    # Neural architectures & data utilities
│   ├── dataset.py                # Contiguous memory Dataset for multimodal data
│   ├── model_htransformer.py     # HTransformer1D unified architecture
│   ├── model_resnet.py           # ResConsultNet multimodal architecture
│   ├── resnet_1d.py              # Backward-compatible layer re-exports
│   └── runner.py                 # Training, validation, and testing routines
├── predict.py                    # Inference CLI for single or batch predictions
├── train.py                      # Cross-validation pipeline for neural & linear models
├── pyproject.toml                # Project dependencies (managed by uv)
├── uv.lock                       # Deterministic dependency lockfile
├── .gitignore                    # Git ignore file
├── CITATION.cff                  # Citation metadata (CFF format)
└── LICENSE                       # License file
```

---

## Citation

If you use this work in your research, please cite:

```text
Vitale, Jacopo, et al. "Automatic Cardiac Risk Management Classification using large-context Electronic Patients Health Records." arXiv preprint arXiv:2603.09685 (2026).
```

```bibtex
@article{vitale2026automatic,
  title={Automatic Cardiac Risk Management Classification using large-context Electronic Patients Health Records},
  author={Vitale, Jacopo and others},
  journal={arXiv preprint arXiv:2603.09685},
  year={2026},
  url={https://arxiv.org/abs/2603.09685}
}
```

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
