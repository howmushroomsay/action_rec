import numpy
import torch
from torch import nn

from .. import utils as U
from .attentions import Attention_Layer
from .layers import Spatial_Graph_Layer, Temporal_Basic_Layer


class EfficientGCN(nn.Module):
    def __init__(self, data_shape, block_args, fusion_stage, stem_channel, **kwargs):
        #fusion_stage = 2, stem_channel = 64 
        super(EfficientGCN, self).__init__()

        num_input, num_channel, _, _, _ = data_shape
        self.data_process = nn.Conv3d(in_channels=1, out_channels=6, kernel_size=3, padding=1, bias=False)
        #3,6
        # input branches
        self.input_branches = nn.ModuleList([EfficientGCN_Blocks(
            init_channel = stem_channel,
            block_args = block_args[:fusion_stage],
            input_channel = num_channel,
            **kwargs
        ) for _ in range(num_input)])

        # main stream
        last_channel = stem_channel if fusion_stage == 0 else block_args[fusion_stage-1][0]
        self.main_stream = EfficientGCN_Blocks(
            init_channel = num_input * last_channel,
            block_args = block_args[fusion_stage:],
            **kwargs
        )

        # output
        last_channel = num_input * block_args[-1][0] if fusion_stage == len(block_args) else block_args[-1][0]
        self.classifier = EfficientGCN_Classifier(last_channel, **kwargs)

        # init parameters
        init_param(self.modules())

    def forward(self, x):

        x = x.squeeze(-1).unsqueeze(dim=1)  # [batchsize, 1, 3, frame, 25]

        x = self.data_process(x).unsqueeze(-1)  # [bathchsize, 6, 3, frame, 25, 1]
        x = x.permute(0, 2, 1, 3, 4, 5)
        
        N, I, C, T, V, M = x.size()
        #N:? I:3种不同类别的数据 C每种数据在3维上分别有两个指标 T:frame V:25 M:人
        x = x.permute(1, 0, 5, 2, 3, 4).contiguous().view(I, N*M, C, T, V)
        #展开数据， 按I, N*M, C, T, V 展开
        #[1,3,6,300,25,2]->[3,1,2,6,300,25]->[3,2,6,300,25]
        # input branches
        x = torch.cat([branch(x[i]) for i, branch in enumerate(self.input_branches)], dim=1)
        
        # main stream
        x = self.main_stream(x)

        # output
        _, C, T, V = x.size()
        feature = x.view(N, M, C, T, V).permute(0, 2, 3, 4, 1)
        out = self.classifier(feature).view(N, -1)

        return out, feature


class EfficientGCN_Blocks(nn.Sequential):
    def __init__(self, init_channel, block_args, layer_type, kernel_size, input_channel=0, **kwargs):
        super(EfficientGCN_Blocks, self).__init__()

        temporal_window_size, max_graph_distance = kernel_size

        if input_channel > 0:  # if the blocks in the input branches
            self.add_module('init_bn', nn.BatchNorm2d(input_channel))
            self.add_module('stem_scn', Spatial_Graph_Layer(input_channel, init_channel, max_graph_distance, **kwargs))
            self.add_module('stem_tcn', Temporal_Basic_Layer(init_channel, temporal_window_size, **kwargs))

        last_channel = init_channel

        temporal_layer = U.import_class(f'auxiliary_tools.action_model.src.model.layers.Temporal_{layer_type}_Layer')

        for i, [channel, stride, depth] in enumerate(block_args):
            self.add_module(f'block-{i}_scn', Spatial_Graph_Layer(last_channel, channel, max_graph_distance, **kwargs))
            for j in range(depth):
                s = stride if j == 0 else 1
                self.add_module(f'block-{i}_tcn-{j}', temporal_layer(channel, temporal_window_size, stride=s, **kwargs))
            self.add_module(f'block-{i}_att', Attention_Layer(channel, **kwargs))
            last_channel = channel


class EfficientGCN_Classifier(nn.Sequential):
    def __init__(self, curr_channel, num_class, drop_prob, **kwargs):
        super(EfficientGCN_Classifier, self).__init__()

        self.add_module('gap', nn.AdaptiveAvgPool3d(1))
        self.add_module('dropout', nn.Dropout(drop_prob, inplace=True))
        self.add_module('fc', nn.Conv3d(curr_channel, num_class, kernel_size=1))
        self.add_module('softmax',nn.Softmax(1))


def init_param(modules):
    for m in modules:
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
