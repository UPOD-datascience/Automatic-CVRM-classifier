from typing import List

import torch
import torch.nn as nn

from functions.resnet_1d import ResNet1D


class ResConsultNet(nn.Module):
    def __init__(
            self, 
            vocab_size            : int       = 5000,
            padding_size          : int       = 5000,
            max_seq_len           : int       = 5000,
            embedding_dim         : int       = 512,
            resnet_layers         : List[int] = [2,2,2,2,2],
            base_filters          : int       = 32,
            pool_size             : int       = 2,
            num_classes           : int       = 2,
            *args, 
            **kwargs
        ):
        
        '''
        This is the ResConsultNet.
        
        Parameters
        ---
        - vocab_size            : vocabulary size of tokenizer
        - embedding_dim         : embeddig dimension
        - resnet_layers         : resnet layer, list of how many resnet basic block the more the deeper
        - base_filters          : starting filters, filter double along network depth
        - pool_size             : Max pooling size, default 2
        - num_classes           : Number of classes, default 2
        - device                : default 'cuda'
        '''
        
        super().__init__(*args, **kwargs)
        
        self.padding_size = padding_size
        self.max_seq_len = max_seq_len
        
        self.atc_norm = nn.LazyBatchNorm1d()
        
        self.embeddings = nn.Embedding(
            num_embeddings = vocab_size + 1,
            embedding_dim  = embedding_dim
        )
        
        self.resnet = ResNet1D(
            layers = resnet_layers,
            base_filters = base_filters,
            pool_size = pool_size,
            num_classes = num_classes
        )
    
    def forward(self, tokens, data, *args, **kwargs):
        
        #print(consults_tokens)
        
        embeddings = self.embeddings(tokens)
        
        # Estrazione delle features dagli embeddings ad opera della ResConsultNet
        embeddings_features = self.resnet.backbone(embeddings)
        
        #print(consults_features,data)
        
        #print(embeddings_features.shape, data.shape)
        
        inputs = torch.cat([embeddings_features, data], dim=1)
        
        #print(f'Embedded total shape: {embedding_total.shape}\n {embedding_total.float()}')
        
        outputs = self.resnet.neural_net(inputs)
        
        #print(f'ResNet output: {outputs.shape}')
        
        return outputs



if __name__ == '__main__':
    
    net = ResConsultNet(
        vocab_size      = 5000,
        padding_size    = 5000,
        embedding_dim   = 512,
        resnet_layers   = [2,2,2,2,2],
        base_filters    = 32,
        pool_size       = 2,
        num_classes     = 2,
        device          = 'cpu'
    )
    
    # Quick test with mock clinical consultation text
    print(net([
        'patient presenteert zich met cardiovasculaire klachten en hypertensie',
        'controle consult voor medicatie evaluatie en bloeddrukmeting'
    ]))