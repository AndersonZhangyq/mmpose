import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from .. import builder
from ..registry import HEADS


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):

    def __init__(self,
                 encoder_layer,
                 num_layers,
                 norm=None,
                 pe_only_at_begin=False,
                 return_atten_map=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.pe_only_at_begin = pe_only_at_begin
        self.return_atten_map = return_atten_map
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        atten_maps_list = []
        for layer in self.layers:
            if self.return_atten_map:
                output, att_map = layer(
                    output,
                    src_mask=mask,
                    pos=pos,
                    src_key_padding_mask=src_key_padding_mask)
                atten_maps_list.append(att_map)
            else:
                output = layer(
                    output,
                    src_mask=mask,
                    pos=pos,
                    src_key_padding_mask=src_key_padding_mask)

            # only add position embedding to the first atttention layer
            pos = None if self.pe_only_at_begin else pos

        if self.norm is not None:
            output = self.norm(output)

        if self.return_atten_map:
            return output, torch.stack(atten_maps_list)
        else:
            return output


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerEncoderLayer(nn.Module):
    """ Modified from https://github.com/facebookresearch/detr/blob/master/models/transformer.py"""

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False,
                 return_atten_map=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.return_atten_map = return_atten_map

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        if self.return_atten_map:
            src2, att_map = self.self_attn(
                q,
                k,
                value=src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask)
        else:
            src2 = self.self_attn(
                q,
                k,
                value=src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if self.return_atten_map:
            return src, att_map
        else:
            return src

    def forward_pre(self,
                    src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        if self.return_atten_map:
            src2, att_map = self.self_attn(
                q,
                k,
                value=src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask)
        else:
            src2 = self.self_attn(
                q,
                k,
                value=src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        if self.return_atten_map:
            return src, att_map
        else:
            return src

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


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
        self.transformer_encoders = TransformerEncoder(
            TransformerEncoderLayer(self.dim_model, self.n_head,
                                       self.dim_feedforward),
            self.encoder_layers)
        self.register_buffer(
            'pe', self._make_position_embedding(*self.heatmap_size, self.dim_model))
        if extra is not None:
            self.extra_header = builder.build_head(extra)

    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        assert pe_type in ['none', 'learnable', 'sine']
        if pe_type == 'none':
            pos_embedding = None
        else:
            with torch.no_grad():
                self.pe_h = h# // 4
                self.pe_w = w# // 4
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                pos_embedding = nn.Parameter(
                    torch.randn(length, 1, d_model))
            else:
                pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
        return pos_embedding

    def _make_sine_position_embedding(self, d_model, temperature=10000,
                                      scale=2 * math.pi):
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(2, 0, 1)
        return pos  # [h*w, 1, d_model]

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
