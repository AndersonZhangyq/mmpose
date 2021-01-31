import copy
import math
from collections import OrderedDict

import numpy as np
import torch.nn as nn
from mmcv.cnn.bricks import Swish

from ..registry import BACKBONES

DEFAULT_BLOCKS_ARGS = [{
    'kernel_size': 3,
    'repeats': 1,
    'filters_out': 16,
    'expand_ratio': 1,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 2,
    'filters_out': 24,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 2,
    'filters_out': 40,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 3,
    'filters_out': 80,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 3,
    'filters_out': 112,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 4,
    'filters_out': 192,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 1,
    'filters_out': 320,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}]

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

ACTIVATION = {'swish': Swish(), 'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid()}


def blocks_args_from_genotype(genotype):
    blocks_args = copy.deepcopy(DEFAULT_BLOCKS_ARGS)
    for i, args in enumerate(genotype):
        blocks_args[i]['kernel_size'] = int(args[0])
        blocks_args[i]['repeats'] = int(args[1])
        blocks_args[i]['filters_out'] = int(args[2] * 8)
        blocks_args[i]['strides'] = int(args[3])
    return blocks_args


def genotype_from_blocks_args(blocks_args):
    genotype = []
    for args in copy.deepcopy(blocks_args):
        genotype.append([
            args['kernel_size'], args['repeats'], args['filters_out'] // 8,
            args['strides']
        ])
    return genotype


def round_filters(filters, width_coefficient, divisor=8):
    """Round number of filters based on depth multiplier."""
    filters *= width_coefficient
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""
    return int(math.ceil(depth_coefficient * repeats))


def scaling_parameters(input_shape,
                       default_size=224,
                       alpha=1.2,
                       beta=1.1,
                       gamma=1.15):
    size = sum(input_shape[:2]) / 2
    if size <= 240:
        drop_connect_rate = 0.2
    elif size <= 300:
        drop_connect_rate = 0.3
    elif size <= 456:
        drop_connect_rate = 0.4
    else:
        drop_connect_rate = 0.5
    phi = (math.log(size) - math.log(default_size)) / math.log(gamma)
    d = alpha**phi
    w = beta**phi
    return d, w, drop_connect_rate


def array_in_list(arr, l):
    return next((True for elem in l if np.array_equal(elem, arr)), False)


class SEModule(nn.Module):

    def __init__(self, filters_in, filters, se_ratio, name, activation):
        super(SEModule, self).__init__()
        self.filters = filters
        filters_se = max(1, int(filters_in * se_ratio))
        self.gap = nn.AdaptiveAvgPool2d(1)
        # se = layers.Reshape((1, 1, filters), name + 'se_reshape')(se)
        self.net = nn.Sequential(
            OrderedDict([
                (name + 'se_reduce', nn.Conv2d(filters, filters_se, 1)),
                (name + 'se_reduce_act', ACTIVATION[activation]),
                (name + 'se_expand', nn.Conv2d(filters_se, filters, 1)),
                (name + 'se_expand_act', ACTIVATION['sigmoid'])
            ]))

    def forward(self, x):
        se = self.gap(x)
        se = se.view((-1, self.filters, 1, 1))
        se = self.net(se)
        return x * se


class Block(nn.Module):

    def __init__(self,
                 in_channels,
                 activation='swish',
                 drop_rate=0.,
                 name='',
                 filters_out=16,
                 kernel_size=3,
                 strides=1,
                 expand_ratio=1,
                 se_ratio=0.,
                 id_skip=True,
                 project=True):
        super(Block, self).__init__()
        filters_in = in_channels

        # Expansion phase
        filters = filters_in * expand_ratio
        expansion_phase = []
        if expand_ratio != 1:
            expansion_phase.append((name + 'expand_conv', (nn.Conv2d(
                in_channels=filters_in,
                out_channels=filters,
                kernel_size=1,
                bias=False,
            ))))
            expansion_phase.append(
                (name + 'expand_bn', nn.BatchNorm2d(filters)))
            expansion_phase.append(
                (name + 'expand_activation', ACTIVATION[activation]))
        else:
            expansion_phase.append((name + 'identity', nn.Identity()))

        self.expansion_phase = nn.Sequential(OrderedDict(expansion_phase))

        # Depthwise Convolution
        self.depthwise = nn.Sequential(
            OrderedDict([(name + 'dwconv',
                          nn.Conv2d(
                              filters,
                              filters,
                              kernel_size=kernel_size,
                              stride=strides,
                              padding=kernel_size // 2,
                              groups=filters,
                              bias=False)),
                         (name + 'bn', nn.BatchNorm2d(filters)),
                         (name + 'activation', ACTIVATION[activation])]))

        # Squeeze and Excitation phase
        self.se_phase = nn.Identity()
        if 0 < se_ratio <= 1:
            self.se_phase = SEModule(filters_in, filters, se_ratio, name,
                                     activation)

        # Output phase
        self.project = nn.Identity()
        if project:
            self.project = nn.Sequential(
                OrderedDict([(name + 'project_conv',
                              nn.Conv2d(filters, filters_out, 1, bias=False)),
                             (name + 'project_bn', nn.BatchNorm2d(filters_out))
                             ]))

        self.final = nn.Identity()
        if id_skip and strides == 1 and filters_in == filters_out and project:
            self.final = nn.Sequential()
            if drop_rate > 0:
                self.final.add_module(name + 'drop', nn.Dropout2d(drop_rate))
            # x = layers.add([x, inputs], name + 'add')

    def forward(self, inputs):
        x = self.expansion_phase(inputs)
        x = self.depthwise(x)
        if not isinstance(self.se_phase, nn.Identity):
            x = self.se_phase(x)
        if not isinstance(self.project, nn.Identity):
            x = self.project(x)
        if not isinstance(self.final, nn.Identity):
            x = self.final(x)
            x = x + inputs
        return x


@BACKBONES.register_module()
class EvoPose2D(nn.Module):

    def __init__(self,
                 load_weights=True,
                 parent=None,
                 genotype=None,
                 width_coefficient=1.,
                 depth_coefficient=1.,
                 depth_divisor=8,
                 activation='swish',
                 head_blocks=3,
                 head_kernel=3,
                 head_channels=128,
                 head_activation='swish',
                 final_kernel=3,
                 save_dir='models'):
        # Protect mutable default arguments
        super().__init__()
        d, w, drop_connect_rate = scaling_parameters([384, 288, 3])
        self.load_weights = load_weights
        self.parent = parent
        self.genotype = genotype
        self.width_coefficient = width_coefficient * w
        self.depth_coefficient = depth_coefficient * d
        self.depth_divisor = depth_divisor
        self.activation = activation
        self.head_blocks = head_blocks
        self.head_kernel = head_kernel
        self.head_channels = head_channels
        self.head_activation = head_activation
        self.final_kernel = final_kernel
        self.save_dir = save_dir
        if self.genotype is None:
            blocks_args = DEFAULT_BLOCKS_ARGS
        else:
            blocks_args = blocks_args_from_genotype(genotype)

        out_channels = round_filters(32, self.width_coefficient,
                                     self.depth_divisor)
        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False), nn.BatchNorm2d(out_channels),
            ACTIVATION[self.activation])
        # Build blocks
        blocks_args = copy.deepcopy(blocks_args)

        b = 0
        blocks = float(
            sum(
                round_repeats(args['repeats'], self.depth_coefficient)
                for args in blocks_args))
        self.main_part = nn.Sequential()
        for (i, args) in enumerate(blocks_args):
            assert args['repeats'] > 0
            # Update block input and output filters based on depth multiplier.
            args['filters_out'] = round_filters(args['filters_out'],
                                                self.width_coefficient,
                                                self.depth_divisor)
            repeats = args['repeats']
            args.pop('repeats')
            current_block = []
            for j in range(round_repeats(repeats, self.depth_coefficient)):
                # The first block needs to take care of stride and filter size increase.
                if j > 0:
                    args['strides'] = 1
                current_block.append(
                    Block(
                        out_channels if j == 0 else args['filters_out'],
                        activation,
                        drop_connect_rate * b / blocks,
                        name='block{}{}_'.format(i + 1, chr(j + 97)),
                        **args))
                b += 1
            self.main_part.add_module(f'block{i + 1}',
                                      nn.Sequential(*current_block))
            out_channels = args['filters_out']
        self.deconv = nn.Sequential()
        last_channels = out_channels
        for i in range(self.head_blocks):
            filters = round_filters(self.head_channels, self.width_coefficient,
                                    self.depth_divisor)
            self.deconv.add_module('head_block{}_upsample'.format(i + 1),
                                   nn.Upsample(scale_factor=2, mode='nearest'))
            self.deconv.add_module(
                'head_block{}_conv'.format(i + 1),
                nn.Conv2d(
                    last_channels,
                    filters,
                    head_kernel,
                    stride=1,
                    padding=head_kernel // 2,
                    bias=False))
            self.deconv.add_module('head_block{}_bn'.format(i + 1),
                                   nn.BatchNorm2d(filters))
            self.deconv.add_module('head_block{}_activation'.format(i + 1),
                                   ACTIVATION[self.head_activation])
            last_channels = filters
        print(self)

    def forward(self, inputs):
        x = self.stem(inputs)
        x = self.main_part(x)
        x = self.deconv(x)
        print(f'deconv: {x.shape}')
        return x
