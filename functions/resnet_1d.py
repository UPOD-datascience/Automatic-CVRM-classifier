import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(
            self, 
            out_channels,
            kernel_size,
            is_first,
            *args, 
            **kwargs
        ) -> None:
        
        super().__init__(*args, **kwargs)
        
        self.is_first = is_first
        
        self.conv = nn.LazyConv1d(
            out_channels = out_channels,
            kernel_size  = kernel_size,
            padding = 'same'
        )
        
        self.batchn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.do = nn.Dropout1d(0.3)
    
    def forward(self, x:torch.Tensor):
        x = x
        
        if self.is_first:
            return self.act(self.batchn(self.conv(x)))
        
        else:
            res = x
            return self.act(self.batchn(self.conv(x))) + res


class ResNet1D(nn.Module):
    def __init__(
            self, 
            layers       : list[int], 
            base_filters : int, 
            num_classes  : int,
            pool_size    : int,
            *args, **kwargs
        ) -> None:
        
        super(ResNet1D, self).__init__(*args, **kwargs)
        
        self.backbone = nn.Sequential()
        
        for l, layer in enumerate(layers):
            for i in range(layer):
                self.backbone.append(Block(
                    out_channels = base_filters*(2**l),
                    kernel_size = 3,
                    is_first = i==0,
                    device = self.device
                ))
            
            if l == len(layers)-1:
                continue
            
            self.backbone.append(nn.MaxPool1d(pool_size))
        
        self.backbone.append(nn.Flatten())
        
        # Classification head
        self.neural_net = nn.Sequential(
            nn.LazyLinear(num_classes*10),
            nn.BatchNorm1d(num_classes*10),
            nn.ReLU(),
            nn.Dropout1d(p = 0.5),
            nn.LazyLinear(num_classes)
        )
        
    
    def forward(self, x):
        x = x
        
        return self.neural_net(self.backbone(x))