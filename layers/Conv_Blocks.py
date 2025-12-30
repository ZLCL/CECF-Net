import torch
import torch.nn as nn


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        self.stride = stride

        # Create multiple convolution kernels with different sizes
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i, stride=stride)
            )
        self.kernels = nn.ModuleList(kernels)

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        # Initialization for convolution layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of Inception Block.
        Input:  x -> (B, C_in, H, W)
        Output: res -> (B, C_out, H, W)
        """
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class Inception_Trans_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_kernels=6, init_weight=True):
        super(Inception_Trans_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        self.stride = stride

        # Create multiple transposed convolution kernels with different sizes
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i, stride=stride)
            )
        self.kernels = nn.ModuleList(kernels)

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        # Initialization for transposed convolution layers
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, output_size):
        """
        Forward pass of Inception Transposed Block.
        Input:  x -> (B, C_in, H, W)
        Output: res -> (B, C_out, H_out, W_out)
        """
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x, output_size=output_size))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class BasicConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, degree, stride=1, padding=0, dilation=1,
                 groups=1, act=False, bn=False, bias=False, dropout=0.):
        super(BasicConv, self).__init__()
        self.out_channels = c_out
        self.conv = nn.Conv1d(
            c_in, c_out, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
            dilation=dilation, groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm1d(c_out) if bn else None
        self.act = nn.GELU() if act else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Basic 1D convolution module.
        Input:  x -> (B, T, C)
        Output: x -> (B, T, C)
        """
        if self.bn is not None:
            x = self.bn(x)
        # Apply 1D convolution
        x = self.conv(x.transpose(-1, -2)).transpose(-1, -2)
        # Apply activation
        if self.act is not None:
            x = self.act(x)
        # Apply dropout
        if self.dropout is not None:
            x = self.dropout(x)
        return x
