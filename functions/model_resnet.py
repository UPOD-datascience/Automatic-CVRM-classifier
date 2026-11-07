from typing import List
import torch
import torch.nn as nn


class Block(nn.Module):
    """1D Convolutional Residual Block."""
    def __init__(self, out_channels: int, kernel_size: int = 3, is_first: bool = False):
        super().__init__()
        self.is_first = is_first
        self.conv = nn.LazyConv1d(out_channels, kernel_size, padding = 'same')
        self.batchn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout1d(0.3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.batchn(self.conv(x)))
        return out if self.is_first else out + x


class ResNet1D(nn.Module):
    """1D Residual Network backbone with pooling and MLP classification head."""
    def __init__(self, layers: List[int], base_filters: int = 32, num_classes: int = 2, pool_size: int = 2):
        super().__init__()
        self.backbone = nn.Sequential()
        for level, num_blocks in enumerate(layers):
            for i in range(num_blocks):
                self.backbone.append(Block(
                    out_channels = base_filters * (2 ** level),
                    kernel_size = 3,
                    is_first = (i == 0)
                ))
            if level < len(layers) - 1:
                self.backbone.append(nn.MaxPool1d(pool_size))
        self.backbone.append(nn.Flatten())
        
        self.neural_net = nn.Sequential(
            nn.LazyLinear(num_classes * 10),
            nn.BatchNorm1d(num_classes * 10),
            nn.ReLU(),
            nn.Dropout1d(p = 0.5),
            nn.LazyLinear(num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.neural_net(self.backbone(x))


class ResConsultNet(nn.Module):
    """
    Multimodal 1D Residual Network for Electronic Health Records.
    Fuses tokenized clinical consultation notes with tabular clinical variables.
    """
    def __init__(
            self, 
            vocab_size: int = 5000,
            padding_size: int = 5000,
            max_seq_len: int = 5000,
            embedding_dim: int = 512,
            resnet_layers: List[int] = None,
            base_filters: int = 32,
            pool_size: int = 2,
            num_classes: int = 2,
            *args, **kwargs
        ):
        super().__init__()
        self.padding_size = padding_size
        self.max_seq_len = max_seq_len
        self.atc_norm = nn.LazyBatchNorm1d()
        self.embeddings = nn.Embedding(vocab_size + 1, embedding_dim)
        self.resnet = ResNet1D(
            layers = resnet_layers or [2, 2, 2, 2, 2],
            base_filters = base_filters,
            pool_size = pool_size,
            num_classes = num_classes
        )
    
    def forward(self, tokens: torch.Tensor, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        features = self.resnet.backbone(self.embeddings(tokens))
        return self.resnet.neural_net(torch.cat([features, data], dim = 1))


if __name__ == '__main__':
    net = ResConsultNet(vocab_size = 5000, embedding_dim = 128, resnet_layers = [2, 2], base_filters = 16)
    out = net(torch.randint(0, 5000, (2, 64)), torch.randn(2, 8))
    print(f"ResConsultNet test successful: {out.shape}")
