# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import ROTATED_NECKS


class MutiChannelAtt(BaseModule):
    def __init__(self, out_channel, num_feats=3, expand_factor=16):
        super(MutiChannelAtt, self).__init__()
        self.out_channel = out_channel

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(num_feats, num_feats * expand_factor, 1, 1, 0, bias=False),
            nn.Conv2d(num_feats * expand_factor, num_feats, 1, 1, 0, bias=False)
        )

        self.sigmoid = nn.Sigmoid()
        self.down_sample = nn.MaxPool2d(kernel_size=2)

        self.feats_weight = nn.Parameter(torch.ones(num_feats, dtype=torch.float32), requires_grad=True)
        self.feats_weight_down = nn.Parameter(torch.ones(num_feats - 1, dtype=torch.float32), requires_grad=True)
        self.weight_act = nn.ReLU()
        self.epsilon = 0.0001

    def forward(self, multi_feats):
        # 均值池化建模
        feats_avg = torch.cat([self.avg_pool(feat).permute(0, 3, 2, 1) for i, feat in enumerate(multi_feats)], dim=1)

        # 最大值池化建模
        feats_max = torch.cat([self.max_pool(feat).permute(0, 3, 2, 1) for i, feat in enumerate(multi_feats)], dim=1)

        # MLP建模
        avg_outs = self.shared_MLP(feats_avg).permute(0, 3, 2, 1)
        max_outs = self.shared_MLP(feats_max).permute(0, 3, 2, 1)

        # 概率预测
        sum_outs = (avg_outs + max_outs).split([1, ] * len(multi_feats), dim=-1)
        probs = [self.sigmoid(out) for out in sum_outs]

        feats_weight = self.weight_act(self.feats_weight)
        outs = [feat + probs[i] * feat * feats_weight[i] for i, feat in enumerate(multi_feats)]

        feats_weight_down = self.weight_act(self.feats_weight_down)
        for i in range(len(outs), 1, -1):
            j = len(outs) - i
            outs[i - 1] = outs[i - 1] + self.down_sample(outs[i - 2]) * feats_weight_down[j]
        return outs

class AlignedModule(nn.Module):
    def __init__(self, inplane, outplane):
        super(AlignedModule, self).__init__()
        self.feat_extract = nn.Conv2d(inplane, outplane, 3, 1, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane * 2, 2, 1, 1, 0)
        self.feats_weight = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        self.weight_act = nn.ReLU()
        self.epsilon = 0.0001

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w
        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)               # 50 * 50
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)               # 50 * 50 * 2
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)      # 只用这里的grid的话，就是双线性插值
        grid = grid + flow.permute(0, 2, 3, 1) / norm                       # flow相当于就是特征对齐所需要的采样偏移量

        output = F.grid_sample(input, grid)
        return output

    def forward(self, low_feature, h_feature):
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        # 低层特征 3, 1, 1 信息交互
        low_feature = self.feat_extract(low_feature)
        # 高层特征 1, 1, 0 信息交互
        h_feature_expand = F.interpolate(h_feature, size=size)
        h_feature = self.feat_extract(h_feature_expand)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        low_feature_from_high_feat = self.flow_warp(h_feature_orign, flow, size=size)        # 用光流进行上采样

        feats_weight = self.weight_act(self.feats_weight)
        return low_feature + low_feature_from_high_feat * feats_weight[0]

@ROTATED_NECKS.register_module()
class FPNTest(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(FPNTest, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        self.fwm_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            if i != self.backbone_end_level - 1:
                fwm_conv = AlignedModule(out_channels, out_channels)
                self.fwm_convs.append(fwm_conv)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # 新增模块
        self.la_ca = MutiChannelAtt(out_channels, num_feats=self.backbone_end_level - self.start_level)
        self.ca = MutiChannelAtt(out_channels, num_feats=self.num_outs)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals = self.la_ca(laterals)

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] = self.fwm_convs[i-1](laterals[i - 1], laterals[i])

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # outs = [
        #     laterals[i] for i in range(used_backbone_levels)
        # ]


        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            # 未指定额外层的输出方式，则使用maxpool来进行输出
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        # 修改后版本，利用差异增强注意力进行(避免直接pool产生extra layer的情况发生）
        outs = self.ca(outs)

        return tuple(outs)
