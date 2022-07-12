import torch
import torch.nn as nn
import torch.nn.functional as F

import util.util as util

from models.networks.base_network import BaseNetwork
from models.networks.architecture import SPADEResnetBlock
from models.networks.architecture import FeatureGenerator
from models.networks.dynast_transformer import DynamicTransformerBlock
from models.networks.dynast_transformer import DynamicSparseTransformerBlock


class DynaSTGenerator(BaseNetwork):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(spectual_norm=True)
        parser.add_argument('--max_multi', type=int, default=8)
        parser.add_argument('--top_k', type=int, default=4)
        parser.add_argument('--n_layers', type=int, default=3)
        parser.add_argument('--smooth', type=float, default=0.01)
        parser.add_argument('--pos_dim', type=int, default=16)
        parser.add_argument('--prune_dim', type=int, default=16)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.encoder_q = FeatureGenerator(opt.semantic_nc, opt.ngf, opt.max_multi, norm='instance')
        self.encoder_kv = FeatureGenerator(opt.semantic_nc + 3, opt.ngf, opt.max_multi, norm='instance')
        pos_embed = nn.Parameter(torch.randn(
            1, opt.pos_dim * min(opt.max_multi, 8), opt.crop_size // 8, opt.crop_size // 8))
        self.register_parameter('pos_embed', pos_embed)
        self.embed_q4 = SPADEResnetBlock(opt.ngf * min(opt.max_multi, 8), opt.semantic_nc)
        self.embed_kv4 = SPADEResnetBlock(opt.ngf * min(opt.max_multi, 8), opt.semantic_nc + 3)
        transformer_4_list = []
        for _ in range(opt.n_layers):
            transformer_4_list.append(DynamicTransformerBlock((opt.ngf + opt.pos_dim) * min(opt.max_multi, 8),
                                                              opt.ngf * min(opt.max_multi, 8),
                                                              opt.prune_dim * min(opt.max_multi, 8),
                                                              opt.semantic_nc, smooth=None))
        self.transformer_4 = nn.ModuleList(transformer_4_list)
        self.decoder_43 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(opt.ngf * min(opt.max_multi, 8), opt.ngf * min(opt.max_multi, 4), (3, 3),
                      (1, 1), 1, padding_mode='reflect'),
            nn.ReLU(),
        )
        self.decoder_q43 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(opt.ngf * min(opt.max_multi, 8), opt.ngf * min(opt.max_multi, 4), (3, 3),
                      (1, 1), 1, padding_mode='reflect'),
            nn.ReLU(),
        )
        self.decoder_kv43 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(opt.ngf * min(opt.max_multi, 8), opt.ngf * min(opt.max_multi, 4), (3, 3),
                      (1, 1), 1, padding_mode='reflect'),
            nn.ReLU(),
        )
        self.pos_43 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(opt.pos_dim * min(8, opt.max_multi),
                      opt.pos_dim * min(4, opt.max_multi),
                      (3, 3), (1, 1), 1, padding_mode='reflect'),
            nn.LeakyReLU(),
        )

        self.embed_q3 = SPADEResnetBlock(opt.ngf * min(opt.max_multi, 4), opt.semantic_nc)
        self.embed_kv3 = SPADEResnetBlock(opt.ngf * min(opt.max_multi, 4), opt.semantic_nc + 3)
        transformer_3_list = []
        for i in range(opt.n_layers):
            transformer_3_list.append(DynamicSparseTransformerBlock((opt.ngf + opt.pos_dim) * min(opt.max_multi, 4),
                                                                    opt.ngf * min(opt.max_multi, 4),
                                                                    opt.prune_dim * min(opt.max_multi, 4),
                                                                    opt.semantic_nc, opt.smooth, inter_scale=i == 0))
        self.transformer_3 = nn.ModuleList(transformer_3_list)
        self.embed_3 = SPADEResnetBlock(opt.ngf * min(opt.max_multi, 4), opt.semantic_nc)
        self.decoder_32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(opt.ngf * min(opt.max_multi, 4), opt.ngf * min(opt.max_multi, 2), (3, 3),
                      (1, 1), 1, padding_mode='reflect'),
            nn.ReLU(),
        )
        self.decoder_q32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(opt.ngf * min(opt.max_multi, 4), opt.ngf * min(opt.max_multi, 2), (3, 3),
                      (1, 1), 1, padding_mode='reflect'),
            nn.ReLU(),
        )
        self.decoder_kv32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(opt.ngf * min(opt.max_multi, 4), opt.ngf * min(opt.max_multi, 2), (3, 3),
                      (1, 1), 1, padding_mode='reflect'),
            nn.ReLU(),
        )
        self.pos_32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(opt.pos_dim * min(4, opt.max_multi),
                      opt.pos_dim * min(2, opt.max_multi),
                      (3, 3), (1, 1), 1, padding_mode='reflect'),
            nn.LeakyReLU(),
        )

        self.embed_q2 = SPADEResnetBlock(opt.ngf * min(opt.max_multi, 2), opt.semantic_nc)
        self.embed_kv2 = SPADEResnetBlock(opt.ngf * min(opt.max_multi, 2), opt.semantic_nc + 3)
        transformer_2_list = []
        for i in range(opt.n_layers):
            transformer_2_list.append(DynamicSparseTransformerBlock((opt.ngf + opt.pos_dim) * min(opt.max_multi, 2),
                                                                    opt.ngf * min(opt.max_multi, 2),
                                                                    opt.prune_dim * min(opt.max_multi, 2),
                                                                    opt.semantic_nc, opt.smooth, inter_scale=i == 0))
        self.transformer_2 = nn.ModuleList(transformer_2_list)
        self.embed_2 = SPADEResnetBlock(opt.ngf * min(opt.max_multi, 2), opt.semantic_nc)
        self.decoder_21 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(opt.ngf * min(opt.max_multi, 2), opt.ngf, (3, 3),
                      (1, 1), 1, padding_mode='reflect'),
            nn.ReLU(),
        )
        self.decoder_q21 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(opt.ngf * min(opt.max_multi, 2), opt.ngf, (3, 3),
                      (1, 1), 1, padding_mode='reflect'),
            nn.ReLU(),
        )
        self.decoder_kv21 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(opt.ngf * min(opt.max_multi, 2), opt.ngf, (3, 3),
                      (1, 1), 1, padding_mode='reflect'),
            nn.ReLU(),
        )
        self.pos_21 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(opt.pos_dim * min(2, opt.max_multi),
                      opt.pos_dim, (3, 3), (1, 1), 1, padding_mode='reflect'),
            nn.LeakyReLU(),
        )

        self.embed_q1 = SPADEResnetBlock(opt.ngf, opt.semantic_nc)
        self.embed_kv1 = SPADEResnetBlock(opt.ngf, opt.semantic_nc + 3)
        transformer_1_list = []
        for i in range(opt.n_layers):
            transformer_1_list.append(DynamicSparseTransformerBlock(opt.ngf + opt.pos_dim, opt.ngf, opt.prune_dim,
                                                                    opt.semantic_nc, opt.smooth, inter_scale=i == 0))
        self.transformer_1 = nn.ModuleList(transformer_1_list)
        self.embed_1 = SPADEResnetBlock(opt.ngf, opt.semantic_nc)
        self.decoder_10 = nn.Sequential(
            nn.Conv2d(opt.ngf, 3, (3, 3), (1, 1), 1, padding_mode='reflect'),
            nn.Tanh()
        )

    def forward(self, ref_img, real_img, seg_map, ref_seg_map):
        out = {}
        ref_input = torch.cat((ref_img, ref_seg_map), dim=1)
        out['warp_out'] = []
        cross_cor_map = None

        q4_prev, q3_prev, q2_prev, q1_prev = self.encoder_q(seg_map)
        kv4_prev, kv3_prev, kv2_prev, kv1_prev = self.encoder_kv(ref_input)
        q4 = self.embed_q4(q4_prev, seg_map)
        kv4 = self.embed_kv4(kv4_prev, ref_input)
        x4 = q4
        pos = self.pos_embed
        for i in range(self.opt.n_layers):
            x4, warped, cross_cor_map = self.transformer_4[i](
                x4, kv4, kv4, pos, seg_map, F.avg_pool2d(ref_img, 8, stride=8) if self.opt.isTrain else None)
            if self.opt.isTrain:
                out['warp_out'].append(warped)
        _, top_k_idx = torch.topk(cross_cor_map, k=self.opt.top_k, dim=-1)
        x = x4

        x = self.decoder_43(x)
        q3 = self.decoder_q43(q4)
        q3 = self.embed_q3(q3 + q3_prev, seg_map)
        kv3 = self.decoder_kv43(kv4)
        kv3 = self.embed_kv3(kv3 + kv3_prev, ref_input)
        x3 = q3
        pos = self.pos_43(pos)
        for i in range(self.opt.n_layers):
            x3, warped, top_k_idx = self.transformer_3[i](
                x3, kv3, kv3, pos, top_k_idx,
                seg_map, F.avg_pool2d(ref_img, 4, stride=4) if self.opt.isTrain else None,
                need_idx=True)
            if self.opt.isTrain:
                out['warp_out'].append(warped)

        x = self.embed_3(x + x3, seg_map)
        x = self.decoder_32(x)
        q2 = self.decoder_q32(q3)
        q2 = self.embed_q2(q2 + q2_prev, seg_map)
        kv2 = self.decoder_kv32(kv3)
        kv2 = self.embed_kv2(kv2 + kv2_prev, ref_input)
        x2 = q2
        pos = self.pos_32(pos)
        for i in range(self.opt.n_layers):
            x2, warped, top_k_idx = self.transformer_2[i](
                x2, kv2, kv2, pos, top_k_idx,
                seg_map, F.avg_pool2d(ref_img, 2, stride=2) if self.opt.isTrain else None,
                need_idx=True)
            if self.opt.isTrain:
                out['warp_out'].append(warped)

        x = self.embed_2(x + x2, seg_map)
        x = self.decoder_21(x)
        q1 = self.decoder_q21(q2)
        q1 = self.embed_q1(q1 + q1_prev, seg_map)
        kv1 = self.decoder_kv21(kv2)
        kv1 = self.embed_kv1(kv1 + kv1_prev, ref_input)
        x1 = q1
        pos = self.pos_21(pos)
        for i in range(self.opt.n_layers):
            x1, warped, top_k_idx = self.transformer_1[i](
                x1, kv1, kv1, pos, top_k_idx,
                seg_map, ref_img if self.opt.isTrain else None, need_idx=i != self.opt.n_layers - 1)
            if self.opt.isTrain:
                out['warp_out'].append(warped)

        x = self.embed_q1(x + x1, seg_map)
        x = self.decoder_10(x)
        out['fake_image'] = x

        return out
