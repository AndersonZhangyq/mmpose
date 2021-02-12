import copy

import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      normal_init)
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.utils import get_root_logger
from ..registry import BACKBONES
from .hrnet import HRModule
from .resnet import BasicBlock, Bottleneck, get_expansion
from .utils import load_checkpoint

import math
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


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


class Transformer(nn.Module):
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

    def __init__(self,
                 dim_model=64,
                 dim_feedforward=128,
                 n_head=1,
                 n_encoder_layers=4,
                 heatmap_size=[64, 48]):
        super().__init__()

        self.dim_model = dim_model
        self.dim_feedforward = dim_feedforward
        self.n_head = n_head
        self.n_encoder_layers = n_encoder_layers
        self.heatmap_size = heatmap_size
        self.transformer_encoders = TransformerEncoder(
            TransformerEncoderLayer(self.dim_model, self.n_head,
                                    self.dim_feedforward),
            self.n_encoder_layers)
        self.register_buffer(
            'pe',
            self._make_position_embedding(*self.heatmap_size, self.dim_model))

    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        assert pe_type in ['none', 'learnable', 'sine']
        if pe_type == 'none':
            pos_embedding = None
        else:
            with torch.no_grad():
                self.pe_h = h  # // 4
                self.pe_w = w  # // 4
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                pos_embedding = nn.Parameter(torch.randn(length, 1, d_model))
            else:
                pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
        return pos_embedding

    def _make_sine_position_embedding(self,
                                      d_model,
                                      temperature=10000,
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
        dim_t = temperature**(2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(2, 0, 1)
        return pos  # [h*w, 1, d_model]

    def forward(self, x):
        """Forward function."""
        if isinstance(x, list):
            x = x[0]
        bs, d, w, h = x.shape
        x = x.flatten(2).permute(2, 0, 1)

        output = self.transformer_encoders(x, pos=self.pe)  # [h*w, bs, d]
        output_reshaped = output.permute(1, 2, 0)
        output_reshaped = output_reshaped.reshape(bs, d, w, h)
        return output_reshaped


@BACKBONES.register_module()
class HRNetForTranspose(nn.Module):

    blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}

    def __init__(self,
                 extra,
                 transformer_cfg=dict(),
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=False):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        self.extra = extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        # stem net
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(self.norm_cfg, 64, postfix=2)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            self.conv_cfg,
            64,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)

        # stage 1
        self.stage1_cfg = self.extra['stage1']
        num_channels = self.stage1_cfg['num_channels'][0]
        block_type = self.stage1_cfg['block']
        num_blocks = self.stage1_cfg['num_blocks'][0]

        block = self.blocks_dict[block_type]
        stage1_out_channels = num_channels * get_expansion(block)
        self.layer1 = self._make_layer(block, 64, stage1_out_channels,
                                       num_blocks)

        # stage 2
        self.stage2_cfg = self.extra['stage2']
        num_channels = self.stage2_cfg['num_channels']
        block_type = self.stage2_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [
            channel * get_expansion(block) for channel in num_channels
        ]
        self.transition1 = self._make_transition_layer([stage1_out_channels],
                                                       num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        # stage 3
        self.stage3_cfg = self.extra['stage3']
        num_channels = self.stage3_cfg['num_channels']
        block_type = self.stage3_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [
            channel * get_expansion(block) for channel in num_channels
        ]
        self.transition2 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.match_dim = nn.Conv2d(pre_stage_channels[0],
                                   transformer_cfg['dim_model'], 1)

        self.global_encoder = Transformer(**transformer_cfg)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    def _make_transition_layer(self, num_channels_pre_layer,
                               num_channels_cur_layer):
        """Make transition layer."""
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False),
                            build_norm_layer(self.norm_cfg,
                                             num_channels_cur_layer[i])[1],
                            nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(nn.Identity())
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else in_channels
                    conv_downsamples.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels,
                                out_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False),
                            build_norm_layer(self.norm_cfg, out_channels)[1],
                            nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv_downsamples))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        """Make layer."""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, out_channels)[1])

        layers = []
        layers.append(
            block(
                in_channels,
                out_channels,
                stride=stride,
                downsample=downsample,
                with_cp=self.with_cp,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg))
        for _ in range(1, blocks):
            layers.append(
                block(
                    out_channels,
                    out_channels,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, in_channels, multiscale_output=True):
        """Make stage."""
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = self.blocks_dict[layer_config['block']]

        hr_modules = []
        for i in range(num_modules):
            # multi_scale_output is only used for the last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            hr_modules.append(
                HRModule(
                    num_branches,
                    block,
                    num_blocks,
                    in_channels,
                    num_channels,
                    reset_multiscale_output,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg))

            in_channels = hr_modules[-1].in_channels

        return nn.Sequential(*hr_modules), in_channels

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if not isinstance(self.transition1[i], nn.Identity):
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if not isinstance(self.transition2[i], nn.Identity):
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        feature = self.match_dim(y_list[0])
        output = self.global_encoder(feature)

        return output

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
