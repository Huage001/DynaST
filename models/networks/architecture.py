# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from util.util import vgg_preprocess
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d


def PositionalNorm2d(x, epsilon=1e-8):
    # x: B*C*W*H normalize in C dim
    mean = x.mean(dim=1, keepdim=True)
    std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
    output = (x - mean) / std
    return output


class SPADE(nn.Module):

    def __init__(self, dim, ic, ks=3):
        super().__init__()
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(ic, dim, kernel_size=(ks, ks), padding=ks // 2, padding_mode='reflect'),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(dim, dim, kernel_size=(ks, ks), padding=ks // 2, padding_mode='reflect')
        self.mlp_beta = nn.Conv2d(dim, dim, kernel_size=(ks, ks), padding=ks // 2, padding_mode='reflect')

    def forward(self, x, signal):
        if signal.shape[-2:] != x.shape[-2:]:
            signal = F.interpolate(signal, x.shape[-2:])
        hidden = self.mlp_shared(signal)
        gamma = self.mlp_gamma(hidden)
        beta = self.mlp_beta(hidden)
        return (1 + gamma) * PositionalNorm2d(x) + beta


class ResidualBlock(nn.Module):
    def __init__(self, dim, ks=3):
        super(ResidualBlock, self).__init__()
        self.relu = nn.PReLU()
        self.model = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(ks, ks), padding=ks // 2, stride=(1, 1), padding_mode='reflect'),
            SynchronizedBatchNorm2d(dim),
            self.relu,
            nn.Conv2d(dim, dim, kernel_size=(ks, ks), padding=ks // 2, stride=(1, 1), padding_mode='reflect'),
            SynchronizedBatchNorm2d(dim),
        )

    def forward(self, x):
        out = self.relu(x + self.model(x))
        return out


class SPADEResnetBlock(nn.Module):
    def __init__(self, dim, ic, use_spectral_norm=True, ks=3):
        super().__init__()
        # Attributes

        self.conv_0 = nn.Conv2d(dim, dim, kernel_size=(ks, ks), padding=ks // 2, padding_mode='reflect')
        self.conv_1 = nn.Conv2d(dim, dim, kernel_size=(ks, ks), padding=ks // 2, padding_mode='reflect')

        if use_spectral_norm:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)

        self.norm_0 = SPADE(dim, ic)
        self.norm_1 = SPADE(dim, ic)
        self.relu = nn.PReLU()

    def forward(self, x, seg):
        dx = self.conv_0(self.relu(self.norm_0(x, seg)))
        dx = self.conv_1(self.relu(self.norm_1(dx, seg)))
        out = self.relu(x + dx)
        return out


class FeatureGenerator(nn.Module):

    def __init__(self, ic, nf=64, max_multi=4, kw=3, use_spectral_norm=True, norm='batch'):
        # Can be changed to patch projection in the future
        super().__init__()
        conv_1 = nn.Conv2d(ic, nf, (3, 3), (1, 1), padding=1, padding_mode='reflect')
        self.layer1 = nn.Sequential(
            spectral_norm(conv_1) if use_spectral_norm else conv_1,
            SynchronizedBatchNorm2d(nf) if norm == 'batch' else nn.InstanceNorm2d(nf),
            ResidualBlock(nf, ks=kw),
        )
        conv_2 = nn.Conv2d(nf, nf * min(2, max_multi), (3, 3),
                           stride=(2, 2), padding=1, padding_mode='reflect')
        self.layer2 = nn.Sequential(
            spectral_norm(conv_2) if use_spectral_norm else conv_2,
            SynchronizedBatchNorm2d(nf * min(2, max_multi)) if norm == 'batch'
            else nn.InstanceNorm2d(nf * min(2, max_multi)),
            ResidualBlock(nf * min(2, max_multi), ks=kw),
        )
        conv_3 = nn.Conv2d(nf * min(2, max_multi), nf * min(4, max_multi), (3, 3),
                           stride=(2, 2), padding=1, padding_mode='reflect')
        self.layer3 = nn.Sequential(
            spectral_norm(conv_3) if use_spectral_norm else conv_3,
            SynchronizedBatchNorm2d(nf * min(4, max_multi)) if norm == 'batch'
            else nn.InstanceNorm2d(nf * min(4, max_multi)),
            ResidualBlock(nf * min(4, max_multi), ks=kw),
        )
        conv_4 = nn.Conv2d(nf * min(4, max_multi), nf * min(8, max_multi), (3, 3),
                           stride=(2, 2), padding=1, padding_mode='reflect')
        self.layer4 = nn.Sequential(
            spectral_norm(conv_4) if use_spectral_norm else conv_4,
            SynchronizedBatchNorm2d(nf * min(8, max_multi)) if norm == 'batch'
            else nn.InstanceNorm2d(nf * min(8, max_multi)),
            ResidualBlock(nf * min(8, max_multi), ks=kw),
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x4, x3, x2, x1


class EmbeddingLayer(nn.Module):

    def __init__(self, ic, patch_size, dim, prev_dim=0, active=nn.LeakyReLU()):
        super().__init__()
        self.conv = nn.Conv2d(patch_size * patch_size * ic + ic + prev_dim, dim, (1, 1))
        self.patch_size = patch_size
        self.active = active

    def forward(self, x, prev_layer=None):
        b, c, h, w = x.shape
        x_patch = x.view(b, c, h // self.patch_size, self.patch_size, w // self.patch_size, self.patch_size)
        x_patch = x_patch.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, -1, h // self.patch_size, w // self.patch_size)
        x_down = F.avg_pool2d(x, self.patch_size, stride=self.patch_size)
        data = [x_patch, x_down]
        if prev_layer is not None:
            data.append(F.interpolate(prev_layer, (h // self.patch_size, w // self.patch_size), mode='bilinear'))
        return self.active(self.conv(torch.cat(data, dim=1)))


class EmbeddingInverseLayer(nn.Module):

    def __init__(self, patch_size, dim, oc=3, active=nn.Tanh()):
        super().__init__()
        self.conv = nn.Conv2d(dim, patch_size * patch_size * oc, (1, 1))
        self.patch_size = patch_size
        self.active = active

    def forward(self, x):
        b, c, h, w = x.shape
        return self.active(self.conv(x).view(b, self.patch_size, self.patch_size, -1, h, w).permute(
            0, 3, 4, 1, 5, 2).contiguous().view(b, -1, h * self.patch_size, w * self.patch_size))


class VGG19_feature_color_torchversion(nn.Module):
    """
    NOTE: there is no need to pre-process the input 
    input tensor should range in [0,1]
    """
    def __init__(self, pool='max', vgg_normal_correct=False, ic=3):
        super(VGG19_feature_color_torchversion, self).__init__()
        self.vgg_normal_correct = vgg_normal_correct

        self.conv1_1 = nn.Conv2d(ic, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys, preprocess=True):
        ''' 
        NOTE: input tensor should range in [0,1]
        '''
        out = {}
        if preprocess:
            x = vgg_preprocess(x, vgg_normal_correct=self.vgg_normal_correct)
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]
