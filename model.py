from torch import nn
import torch
import torchvision.transforms.functional as TF

class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, pre_activation = None, padding = 0):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding)
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        if pre_activation is None or not isinstance(pre_activation, nn.Module):
            self.pre_activation = nn.Identity()
        else:
            self.pre_activation = pre_activation
    def forward(self, x):
        x = self.pre_activation(x)
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x
    
class StackedConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, stack = 2, kernel_size = 3, pre_activation = None, padding = 0):
        super().__init__()
        self.stack = nn.Sequential(ConvReLU(in_channels, out_channels, kernel_size, padding=padding))
        for _ in range(stack - 1):
            self.stack.append(ConvReLU(out_channels, out_channels, kernel_size, padding=padding))
        if pre_activation is None or not isinstance(pre_activation, nn.Module):
            self.pre_activation = nn.Identity()
        else:
            self.pre_activation = pre_activation
    def forward(self, x):
        x = self.pre_activation(x)
        return self.stack(x)
    
class Up(StackedConvReLU):
    def __init__(self, in_channels, out_channels, stack = 2, kernel_size = 3, pre_activation = None, padding = 1):
        '''A child class with modifications so that the forward will concat 2 tensors before send them to self.stack.foward'''
        super().__init__(in_channels=in_channels, out_channels=out_channels, stack=stack, kernel_size=kernel_size, pre_activation=pre_activation, padding=padding)
    def forward(self, x, skip):
        x = self.pre_activation(x)
        skip = TF.center_crop(skip, [x.shape[-2], x.shape[-1]])
        stacked = torch.cat([x, skip], dim=1)
        return self.stack(stacked)
    
class UNet(nn.Module):
    def __init__(self, encoder_channels: list, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(StackedConvReLU(in_channels, encoder_channels[0]))
        self.encoder.extend([StackedConvReLU(encoder_channels[idx],
                                             encoder_channels[idx + 1],
                                             pre_activation = nn.MaxPool2d(2)) for idx in range(0, len(encoder_channels) - 1)])
        
        self.decoder = nn.Sequential(*[Up(encoder_channels[idx],
                                                       encoder_channels[idx - 1],
                                                       pre_activation = nn.ConvTranspose2d(encoder_channels[idx], encoder_channels[idx - 1] , 2 , 2)) for idx in range(len(encoder_channels)-1, 0, -1)]) 
        self.head = nn.Conv2d(in_channels = encoder_channels[0], out_channels = out_channels, kernel_size = 1)
    def forward(self, x):
        encode = []
        for layer in self.encoder:
            x = layer(x)
            encode.insert(0, x)
        encode = encode[1:]

        for idx, layer in enumerate(self.decoder):
            x = layer(x, encode[idx])
        return self.head(x)