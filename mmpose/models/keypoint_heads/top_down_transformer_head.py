import math

import torch
import torch.nn as nn

from .. import builder
from ..registry import HEADS


@HEADS.register_module()
class TopDownTransformerHead(nn.Module):
    """Top-down model head of simple baseline paper ref: Bin Xiao. ``Simple
    Baselines for Human Pose Estimation and Tracking``.

    TopDownSimpleHead is consisted of (>=0) number of deconv layers
    and a simple conv2d layer.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
    """

    def __init__(self, in_channels, out_channels, extra=None):
        super().__init__()

        self.in_channels = in_channels
        self.dim_model = 64
        self.dim_feedforward = 128
        self.n_head = 1
        self.encoder_layers = 4
        self.heatmap_size = [64, 48]
        self.transformer_encoders = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.dim_model, self.n_head,
                                       self.dim_feedforward),
            self.encoder_layers)
        self.register_buffer(
            'pe', self.positionalencoding2d(self.dim_model,
                                            *self.heatmap_size))
        if extra is not None:
            self.extra_header = builder.build_head(extra)

    def forward(self, x):
        """Forward function."""
        if isinstance(x, list):
            x = x[0]
        x_position_encoded = self.pe + x
        bs, d, w, h = x_position_encoded.shape
        x_reshaped = x_position_encoded.reshape(bs, d, -1)
        x_reshaped = x_reshaped.permute(0, 2, 1)
        output = self.transformer_encoders(x_reshaped)
        output_reshaped = output.permute(0, 2, 1)
        output_reshaped = output_reshaped.reshape(bs, d, w, h)
        if self.extra_header is not None:
            output_reshaped = self.extra_header(output_reshaped)
        return output_reshaped

    def init_weights(self):
        """Initialize model weights."""
        pass
        # for _, m in self.deconv_layers.named_modules():
        #     if isinstance(m, nn.ConvTranspose2d):
        #         normal_init(m, std=0.001)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         constant_init(m, 1)
        # for m in self.final_layer.modules():
        #     if isinstance(m, nn.Conv2d):
        #         normal_init(m, std=0.001, bias=0)

    def positionalencoding2d(self, d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError('Cannot use sin/cos positional encoding with '
                             'odd dimension (got dim={:d})'.format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe
