# =============================================================================
# Vendored from: https://github.com/princeton-vl/SEA-RAFT
# Original RAFT implementation by Zachary Teed and Jia Deng
# Licensed under BSD-3-Clause
# =============================================================================
#
# BSD 3-Clause License
#
# Copyright (c) 2024, Princeton Vision & Learning Lab
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from types import SimpleNamespace

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = norm_layer(planes)
        self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.bn3 = norm_layer(planes)
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                self.bn3
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.relu(self.bn2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x+y)

class ResNetFPN(nn.Module):
    def __init__(self, args, input_dim=3, output_dim=256, ratio=1.0, norm_layer=nn.BatchNorm2d, init_weight=False):
        super().__init__()
        block = BasicBlock
        block_dims = args.block_dims
        initial_dim = args.initial_dim
        self.init_weight = init_weight
        self.input_dim = input_dim
        
        self.in_planes = initial_dim
        for i in range(len(block_dims)):
            block_dims[i] = int(block_dims[i] * ratio)
            
        self.conv1 = nn.Conv2d(input_dim, initial_dim, kernel_size=7, stride=2, padding=3)
        self.bn1 = norm_layer(initial_dim)
        self.relu = nn.ReLU(inplace=True)
        
        if args.pretrain == 'resnet34':
            n_block = [3, 4, 6]
        elif args.pretrain == 'resnet18':
            n_block = [2, 2, 2]
        else:
            raise NotImplementedError
            
        self.layer1 = self._make_layer(block, block_dims[0], stride=1, norm_layer=norm_layer, num=n_block[0])
        self.layer2 = self._make_layer(block, block_dims[1], stride=2, norm_layer=norm_layer, num=n_block[1])
        self.layer3 = self._make_layer(block, block_dims[2], stride=2, norm_layer=norm_layer, num=n_block[2])
        self.final_conv = conv1x1(block_dims[2], output_dim)

    def _make_layer(self, block, dim, stride=1, norm_layer=nn.BatchNorm2d, num=2):
        layers = []
        layers.append(block(self.in_planes, dim, stride=stride, norm_layer=norm_layer))
        for i in range(num - 1):
            layers.append(block(dim, dim, stride=1, norm_layer=norm_layer))
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
        output = self.final_conv(x)
        return output

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvNextBlock(nn.Module):
    def __init__(self, dim, output_dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * output_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * output_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.final = nn.Conv2d(dim, output_dim, kernel_size=1, padding=0)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = self.final(input + x)
        return x

class BasicMotionEncoder(nn.Module):
    def __init__(self, args, dim=128):
        super().__init__()
        cor_planes = args.corr_channel
        self.convc1 = nn.Conv2d(cor_planes, dim*2, 1, padding=0)
        self.convc2 = nn.Conv2d(dim*2, dim+dim//2, 3, padding=1)
        self.convf1 = nn.Conv2d(2, dim, 7, padding=3)
        self.convf2 = nn.Conv2d(dim, dim//2, 3, padding=1)
        self.conv = nn.Conv2d(dim*2, dim-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hdim=128, cdim=128):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args, dim=cdim)
        self.refine = []
        for i in range(args.num_blocks):
            self.refine.append(ConvNextBlock(2*cdim+hdim, hdim))
        self.refine = nn.ModuleList(self.refine)

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        for blk in self.refine:
            net = blk(torch.cat([net, inp], dim=1))
        return net

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)
    return img

class CorrBlock:
    def __init__(self, fmap1, fmap2, args):
        self.num_levels = args.corr_levels
        self.radius = args.corr_radius
        self.args = args
        self.corr_pyramid = []
        for i in range(self.num_levels):
            corr = CorrBlock.corr(fmap1, fmap2, 1)
            batch, h1, w1, dim, h2, w2 = corr.shape
            corr = corr.reshape(batch*h1*w1, dim, h2, w2)
            fmap2 = F.interpolate(fmap2, scale_factor=0.5, mode='bilinear', align_corners=False)
            self.corr_pyramid.append(corr)

    def __call__(self, coords, dilation=None):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        if dilation is None:
            dilation = torch.ones(batch, 1, h1, w1, device=coords.device)

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            device = coords.device
            dx = torch.linspace(-r, r, 2*r+1, device=device)
            dy = torch.linspace(-r, r, 2*r+1, device=device)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1)
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            delta_lvl = delta_lvl * dilation.view(batch * h1 * w1, 1, 1, 1)
            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            coords_lvl = centroid_lvl + delta_lvl
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        out = out.permute(0, 3, 1, 2).contiguous().float()
        return out

    @staticmethod
    def corr(fmap1, fmap2, num_head):
        batch, dim, h1, w1 = fmap1.shape
        h2, w2 = fmap2.shape[2:]
        fmap1 = fmap1.view(batch, num_head, dim // num_head, h1*w1)
        fmap2 = fmap2.view(batch, num_head, dim // num_head, h2*w2)
        corr = fmap1.transpose(2, 3) @ fmap2
        corr = corr.reshape(batch, num_head, h1, w1, h2, w2).permute(0, 2, 3, 1, 4, 5)
        return corr / torch.sqrt(torch.tensor(dim).float())

def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device), indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

class InputPadder:
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

class SEA_RAFT(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        
        if cfg is not None and hasattr(cfg, 'items'):
            cfg = cfg.items
        else:
            cfg = {}
            
        # Default config values (from SEA-RAFT-M)
        cfg.setdefault('use_var', True)
        cfg.setdefault('var_min', 0)
        cfg.setdefault('var_max', 10)
        cfg.setdefault('pretrain', 'resnet34')
        cfg.setdefault('initial_dim', 64)
        cfg.setdefault('block_dims', [64, 128, 256])
        cfg.setdefault('radius', 4)
        cfg.setdefault('dim', 128)
        cfg.setdefault('num_blocks', 2)
        cfg.setdefault('iters', 4)
        
        args = SimpleNamespace(**cfg)
        args.corr_levels = 4
        args.corr_radius = args.radius
        args.corr_channel = args.corr_levels * (args.radius * 2 + 1) ** 2
        
        self.args = args
        self.output_dim = args.dim * 2

        self.cnet = ResNetFPN(args, input_dim=6, output_dim=2 * self.args.dim, norm_layer=nn.BatchNorm2d, init_weight=False)

        self.init_conv = conv3x3(2 * args.dim, 2 * args.dim)
        self.upsample_weight = nn.Sequential(
            nn.Conv2d(args.dim, args.dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.dim * 2, 64 * 9, 1, padding=0)
        )
        self.flow_head = nn.Sequential(
            nn.Conv2d(args.dim, 2 * args.dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * args.dim, 6, 3, padding=1)
        )
        if args.iters > 0:
            self.fnet = ResNetFPN(args, input_dim=3, output_dim=self.output_dim, norm_layer=nn.BatchNorm2d, init_weight=False)
            self.update_block = BasicUpdateBlock(args, hdim=args.dim, cdim=args.dim)

    def upsample_data(self, flow, info, mask):
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)
        
        return up_flow.reshape(N, 2, 8*H, 8*W), up_info.reshape(N, C, 8*H, 8*W)

    def forward(self, image1, image2, iters=None, flow_gt=None, test_mode=False):
        N, _, H, W = image1.shape
        if iters is None:
            iters = self.args.iters
        if flow_gt is None:
            flow_gt = torch.zeros(N, 2, H, W, device=image1.device)

        image1 = 2 * image1 - 1.0
        image2 = 2 * image2 - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        flow_predictions = []
        info_predictions = []

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        N, _, H, W = image1.shape
        dilation = torch.ones(N, 1, H//8, W//8, device=image1.device)
        
        cnet = self.cnet(torch.cat([image1, image2], dim=1))
        cnet = self.init_conv(cnet)
        net, context = torch.split(cnet, [self.args.dim, self.args.dim], dim=1)

        flow_update = self.flow_head(net)
        weight_update = .25 * self.upsample_weight(net)
        flow_8x = flow_update[:, :2]
        info_8x = flow_update[:, 2:]
        flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
        flow_predictions.append(flow_up)
        info_predictions.append(info_up)
            
        if self.args.iters > 0:
            fmap1_8x = self.fnet(image1)
            fmap2_8x = self.fnet(image2)
            corr_fn = CorrBlock(fmap1_8x, fmap2_8x, self.args)

        for itr in range(iters):
            N, _, H, W = flow_8x.shape
            flow_8x = flow_8x.detach()
            coords2 = (coords_grid(N, H, W, device=image1.device) + flow_8x).detach()
            corr = corr_fn(coords2, dilation=dilation)
            net = self.update_block(net, context, corr, flow_8x)
            flow_update = self.flow_head(net)
            weight_update = .25 * self.upsample_weight(net)
            flow_8x = flow_8x + flow_update[:, :2]
            info_8x = flow_update[:, 2:]
            flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)

        for i in range(len(info_predictions)):
            flow_predictions[i] = padder.unpad(flow_predictions[i])
            info_predictions[i] = padder.unpad(info_predictions[i])

        if test_mode:
            return flow_predictions[-1], None
        else:
            return {'final': flow_predictions[-1], 'flow': flow_predictions, 'info': info_predictions, 'nf': None}
