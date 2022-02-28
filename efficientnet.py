""" A minimalist Pytorch implementation of EfficientNet directly based on the original paper:
    "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
    Mingxing Tan, Quoc V. Le, 2019, https://arxiv.org/abs/1905.11946
    It allows 2D or 3D inputs thanks to the small LayerNd functions.
    A small Decoder is also provided in order to use EfficientNet as the backbone for an
    autoencoder.
"""

import math
import torch
from torch import nn


# +------------------------------------------------------------------------------------------+ #
# |                                         ND LAYERS                                        | #
# +------------------------------------------------------------------------------------------+ #

def ConvNd(*args, dim=2, **kwargs):
    return nn.Conv2d(*args, **kwargs) if dim == 2 else nn.Conv3d(*args, **kwargs)


def BatchNormNd(*args, dim=2, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs) if dim == 2 else nn.BatchNorm3d(*args, **kwargs)


def AdaptativeAvgPoolNd(*args, dim=2, **kwargs):
    if dim == 2:
        return nn.AdaptiveAvgPool2d(*args, **kwargs)
    return nn.AdaptiveAvgPool3d(*args, **kwargs)


def ConvTransposeNd(*args, dim=2, **kwargs):
    if dim == 2:
        return nn.ConvTranspose2d(*args, **kwargs)
    return nn.ConvTranspose3d(*args, **kwargs)


# +------------------------------------------------------------------------------------------+ #
# |                                      BUILDING BLOCKS                                     | #
# +------------------------------------------------------------------------------------------+ #

class ConvBnAct(nn.Sequential):

    """ Layer grouping a convolution, a batchnorm, and optionaly an activation function.

    Quoting original paper Section 5.2: "We train our EfficientNet models with [...] batch norm
    momentum 0.99 [...]. We also use SiLU (Swish-1) activation (Ramachandran et al., 2018;
    Elfwing et al., 2018; Hendrycks & Gimpel, 2016).
    """

    def __init__(self, n_in, n_out, kernel_size=3, stride=1, padding=0, groups=1,
                 bias=False, bn=True, act=True, dim=2):
        super().__init__()
        conv_params = {'kernel_size': kernel_size, 'stride': stride, 'padding': padding,
                       'groups': groups, 'bias': bias}
        self.add_module('conv', ConvNd(n_in, n_out, dim=dim, **conv_params))
        self.add_module('bn', BatchNormNd(n_out, dim=dim, momentum=0.99) if bn else nn.Identity())
        self.add_module('act', nn.SiLU() if act else nn.Identity())


class SEBlock(nn.Module):

    """ Squeeze-and-excitation block. """

    def __init__(self, n_in, r=24, dim=2):
        super().__init__()
        self.squeeze    = AdaptativeAvgPoolNd(1, dim=dim)
        self.excitation = nn.Sequential(
            ConvNd(n_in, n_in // r, dim=dim, kernel_size=1),
            nn.SiLU(),
            ConvNd(n_in // r, n_in, dim=dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.excitation(self.squeeze(x))


class DropSample(nn.Module):

    """ Drops each sample in x with probability p during training (a sort of DropConnect).

    In the original paper Dropout regularization is mentionned but it's not what the official
    repo shows. See this discussion: https://github.com/tensorflow/tpu/issues/494.

    Official tensorflow code:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/utils.py#L276
    """

    def __init__(self, p=0, dim=2):
        super().__init__()
        self.p = p
        self.size = (dim + 1) * (1, )

    def forward(self, x):
        if (not self.p) or (not self.training):
            return x
        random_tensor = torch.FloatTensor(len(x), *self.size).uniform_().to(x.device)
        bit_mask = self.p < random_tensor
        x = x.div(1 - self.p)
        x = x * bit_mask
        return x


# +------------------------------------------------------------------------------------------+ #
# |                                        MBCONV BLOCK                                      | #
# +------------------------------------------------------------------------------------------+ #

class MBConv(nn.Module):

    """ MBConv with expansion, plus squeeze-and-excitation and dropsample.

    Replace costly 3 X 3 convolutions with depthwise convolutions and follows a
    narrow -> wide -> narrow structure as opposed to the wide -> narrow -> wide one found in
    original residual blocks.

    Quoting original paper Section 4: "Its main building block is mobile inverted bottleneck
    MBConv (Sandler et al., 2018; Tan et al., 2019), to which we also add squeeze-and-excitation
    optimization (Hu et al., 2018)."
    """
    def __init__(self, n_in, n_out, expand_factor, kernel_size=3, stride=1, r=24, p=0, dim=2):
        super().__init__()

        expanded = expand_factor * n_in
        padding  = (kernel_size - 1) // 2
        depthwise_conv_params = {'kernel_size': kernel_size, 'padding': padding,
                                 'stride': stride, 'groups': expanded}

        self.skip_connection = (n_in == n_out) and (stride == 1)

        if expand_factor == 1:
            self.expand_pw = nn.Identity()
        else:
            self.expand_pw = ConvBnAct(n_in, expanded, kernel_size=1, dim=dim)
        self.depthwise  = ConvBnAct(expanded, expanded, **depthwise_conv_params, dim=dim)
        self.se         = SEBlock(expanded, r=r, dim=dim)
        self.reduce_pw  = ConvBnAct(expanded, n_out, kernel_size=1, act=False, dim=dim)
        self.dropsample = DropSample(p, dim)

    def forward(self, x):
        residual = x
        x = self.reduce_pw(self.se(self.depthwise(self.expand_pw(x))))
        if self.skip_connection:
            x = self.dropsample(x)
            x = x + residual
        return x


# +------------------------------------------------------------------------------------------+ #
# |                                       EFFICIENT NET                                      | #
# +------------------------------------------------------------------------------------------+ #

class EfficientNet(nn.Module):

    """ Generic EfficientNet that takes in the width and depth scale factors and scales
        accordingly. It can handle 2D or 3D data through the 'dim' argument.
    """

    def __init__(self, in_channels=1, width_factor=1, depth_factor=1, num_classes=1000, dim=2):
        super().__init__()

        base_widths   = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        base_depths   = [1, 2, 2, 3, 3, 4, 1]
        kernel_sizes  = [3, 3, 5, 3, 5, 5, 3]
        strides       = [1, 2, 2, 2, 1, 2, 1]
        drop_probas   = [0, 0.029, 0.057, 0.086, 0.114, 0.143, 0.171]
        reduce        = [4] + 6 * [24]
        expands       = [1] + 6 * [6]

        scaled_widths = [self.scale_width(w, width_factor) for w in base_widths]
        scaled_depths = [math.ceil(depth_factor * d) for d in base_depths]

        stage_params  = [(scaled_widths[i], scaled_widths[i + 1], scaled_depths[i],
                          expands[i], kernel_sizes[i], strides[i],
                          reduce[i], drop_probas[i], dim) for i in range(7)]

        # Stage 1 in the original paper Table 1
        stem_params = {'stride': 2, 'padding': 1, 'dim': dim}
        self.stem = ConvBnAct(in_channels, scaled_widths[0], **stem_params)

        # Stages 2 to 7 in the original paper Table 1
        self.stages = nn.Sequential(*[self.create_stage(*stage_params[i]) for i in range(7)])

        # First layer of stage 9 in the original paper Table 1
        self.pre_head = ConvBnAct(scaled_widths[-2], scaled_widths[-1], kernel_size=1, dim=dim)

        # Second and third layers of stage 9 in the original paper Table 1
        self.head = nn.Sequential(AdaptativeAvgPoolNd(1, dim=dim),
                                  nn.Flatten(),
                                  nn.Linear(scaled_widths[-1], num_classes))

        self.latent_dim = scaled_widths[-1]

    @staticmethod
    def create_stage(n_in, n_out, num_layers, expand, kernel_size, stride, r, p, dim):
        """ Creates a Sequential of MBConv. """
        common_params = {'kernel_size': kernel_size, 'r': r, 'p': p, 'dim': dim}
        layers = [MBConv(n_in, n_out, expand, stride=stride, **common_params)]
        layers += [MBConv(n_out, n_out, expand, **common_params) for _ in range(num_layers - 1)]
        return nn.Sequential(*layers)

    @staticmethod
    def scale_width(w, w_factor):
        """ Scales width given a scale factor.
        See:
        https://stackoverflow.com/questions/60583868/how-is-the-number-of-channels-adjusted-in-efficientnet.
        """
        w *= w_factor
        new_w = (int(w + 4) // 8) * 8
        new_w = max(8, new_w)
        if new_w < 0.9 * w:
            new_w += 8
        return int(new_w)

    def extract_features(self, x):
        return self.pre_head(self.stages(self.stem(x)))

    def forward(self, x):
        return self.head(self.extract_features(x))

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_name(cls, network_name, **kwargs):
        """ Official tensorflow code:
        https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_builder.py#L35
        """
        config = {
            'efficientnet-b0': {'width_factor': 1.0, 'depth_factor': 1.0},
            'efficientnet-b1': {'width_factor': 1.0, 'depth_factor': 1.1},
            'efficientnet-b2': {'width_factor': 1.1, 'depth_factor': 1.2},
            'efficientnet-b3': {'width_factor': 1.2, 'depth_factor': 1.4},
            'efficientnet-b4': {'width_factor': 1.4, 'depth_factor': 1.8},
            'efficientnet-b5': {'width_factor': 1.6, 'depth_factor': 2.2},
            'efficientnet-b6': {'width_factor': 1.8, 'depth_factor': 2.6},
            'efficientnet-b7': {'width_factor': 2.0, 'depth_factor': 3.1},
        }
        return cls(**{**config[network_name], **kwargs})


# +------------------------------------------------------------------------------------------+ #
# |                                          DECODER                                         | #
# +------------------------------------------------------------------------------------------+ #

class Decoder(nn.Module):

    """ A simple Decoder Module that can be used with EfficientNet as an encoder, e.g:
        > dim  = 2  # can be 2 or 3
        > size = 24
        > encoder = EfficientNet.from_name('efficientnet-b0', dim=dim)
        > decoder = Decoder(out_size=size, dim=dim, latent_dim=encoder.latent_dim)
        > x = torch.randn(1, 1, *(dim * (size, )))
        > x_hat = decoder(encoder.extract_features(x))
    """

    def __init__(self, out_channels=1, out_size=24, dim=2, hidden_dim=128,
                 num_features=(8, 16, 32), latent_dim=4) -> None:

        super().__init__()

        self.out_size = out_size
        unflattened_size = (num_features[-1], *(dim * (3, )))

        first_conv_params = {'kernel_size': 3, 'stride': 2, 'output_padding': 0}
        other_conv_params = {'kernel_size': 3, 'stride': 2, 'output_padding': 1, 'padding': 1}

        self.linear = nn.Sequential(
            AdaptativeAvgPoolNd(1, dim=dim),
            nn.Flatten(),
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, torch.as_tensor(unflattened_size).prod().item()),
            nn.ReLU(inplace=True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=unflattened_size)

        self.conv = nn.Sequential(
            ConvTransposeNd(num_features[-1], num_features[-2], dim=dim, **first_conv_params),
            BatchNormNd(num_features[-2], dim=dim),
            nn.ReLU(True),
            ConvTransposeNd(num_features[-2], num_features[-3], dim=dim, **other_conv_params),
            BatchNormNd(num_features[-3], dim=dim),
            nn.ReLU(True),
            ConvTransposeNd(num_features[-3], out_channels,     dim=dim, **other_conv_params)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # linear -> unflatten -> conv -> resize -> sigmoid
        x = self.conv(self.unflatten(self.linear(x)))
        x = nn.functional.interpolate(x, size=self.out_size)
        return x.sigmoid()

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
