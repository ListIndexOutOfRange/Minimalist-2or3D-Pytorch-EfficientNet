# Minimalist Pytorch 2/3D EfficientNet

A minimalist Pytorch implementation of EfficientNet directly based on the original paper
[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946), 
Mingxing Tan, Quoc V. Le, 2019. 

It allows 2D or 3D inputs.

A small Decoder is also provided in order to use EfficientNet as the backbone for an autoencoder.


# Usage

The network can be instanciated either classicaly or by the **.from_name()** method.
This method supports 'efficientnet-b{i}', for any i from 0 to 7. 
The **dim** param tells the network wether the inputs are 2 or 3 D.

## Basic

```python

import torch
from efficientnet import EfficientNet

# ____________________ base params ____________________ #
batch_size  = 64
in_channels = 1
input_size  = 32
dim         = 2  # can be 2 or 3
num_classes = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net_params = dict(in_channels=in_channels, dim=dim, num_classes=num_classes)
net = EfficientNet.from_name('efficientnet-b0', **net_params).to(device)

x   = torch.randn(batch_size, in_channels, *(dim * (input_size, ))).to(device)
y   = net(x)
```


## With decoder

```python

from efficientnet import Decoder

decoder = Decoder(out_size=input_size, dim=dim, latent_dim=net.latent_dim).to(device)

x_hat = decoder(net.extract_features(x))
```
