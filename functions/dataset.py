import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler

import pandas as pd


class EHR(Dataset):
    """
    Multimodal EHR dataset combining tabular clinical features and
    tokenized unstructured clinical text.
    """
    def __init__(
            self,
            df                      : pd.DataFrame,
            labels                  : pd.Series,
            tabular_features        : list,
            unstructured_features   : list,
            tokenizer               : object,
            tokens_len              : int,
            scaler                  : StandardScaler
        ):
        
        # Verify that all tabular features are numeric
        for feature in tabular_features:
            if not pd.api.types.is_numeric_dtype(df[feature]):
                raise ValueError(f"The column '{feature}' doesn't have numerical data.")
        
        dataset_tabular = df[tabular_features].astype('float32')
        dataset_unstructured = df[unstructured_features]
        
        if not hasattr(scaler, 'scale_'):
            dataset_tabular[tabular_features] = scaler.fit_transform(dataset_tabular)
        else:
            dataset_tabular[tabular_features] = scaler.transform(dataset_tabular)
        
        for feature in unstructured_features:
            tokens = tokenizer(
                list(dataset_unstructured[feature]),
                truncation = True, 
                padding = 'max_length', 
                max_length = tokens_len,
                return_tensors = 'np'
            )
            dataset_unstructured.loc[:, feature] = tokens['input_ids'].tolist()
        
        self.dataset_tabular = dataset_tabular
        self.dataset_unstructured = dataset_unstructured
        self.labels = labels
        self.masks = tokens['attention_mask'].tolist()
    
    def __getitem__(self, index):
        return torch.tensor(self.dataset_tabular.iloc[[index]].values, dtype=torch.float32).squeeze(), \
            torch.tensor(self.dataset_unstructured.iloc[[index]].values[0][0], dtype=torch.long), \
            torch.tensor(self.masks[index], dtype=torch.bool), \
            torch.tensor(self.labels.iloc[index], dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
