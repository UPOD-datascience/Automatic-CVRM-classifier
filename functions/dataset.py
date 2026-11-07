import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class EHR(Dataset):
    """
    Multimodal Electronic Health Records (EHR) Dataset.
    High-performance storage backed by contiguous memory PyTorch tensors.
    """
    def __init__(
            self,
            tabular = None,
            tokens = None,
            masks = None,
            labels = None,
            *,
            df: pd.DataFrame = None,
            tabular_features: list = None,
            unstructured_features: list = None,
            tokenizer: object = None,
            tokens_len: int = 8192,
            scaler: StandardScaler = None
        ):
        # Support DataFrame initialization
        if df is not None:
            for feat in tabular_features:
                if not pd.api.types.is_numeric_dtype(df[feat]):
                    raise ValueError(f"Column '{feat}' must contain numeric data.")
            
            raw_tab = df[tabular_features].to_numpy(dtype = np.float32)
            scaled = scaler.fit_transform(raw_tab) if not hasattr(scaler, 'scale_') else scaler.transform(raw_tab)
            self.tabular = torch.from_numpy(np.asarray(scaled, dtype = np.float32))
            
            texts = df[unstructured_features[0]] if len(unstructured_features) == 1 else df[unstructured_features].agg(" ".join, axis = 1)
            tok = tokenizer(
                texts.fillna("").astype(str).tolist(),
                truncation = True,
                padding = 'max_length',
                max_length = tokens_len,
                return_tensors = 'pt'
            )
            self.tokens = tok['input_ids'].long()
            self.masks = tok['attention_mask'].bool()
            self.labels = torch.tensor(labels.to_numpy() if hasattr(labels, 'to_numpy') else np.asarray(labels), dtype = torch.long)
            self.dataset_tabular = df[tabular_features]
        else:
            self.tabular = torch.as_tensor(tabular, dtype = torch.float32)
            self.tokens = torch.as_tensor(tokens, dtype = torch.long)
            self.masks = torch.as_tensor(masks, dtype = torch.bool)
            self.labels = torch.as_tensor(labels, dtype = torch.long)
    
    def __getitem__(self, index: int):
        return self.tabular[index], self.tokens[index], self.masks[index], self.labels[index]
    
    def __len__(self) -> int:
        return len(self.labels)
