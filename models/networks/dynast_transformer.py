import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.architecture import ResidualBlock
from models.networks.architecture import SPADE
from models.networks.architecture import PositionalNorm2d
import util.util as util


class SignWithSigmoidGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        result = (x > 0).float()
        sigmoid_result = torch.sigmoid(x)
        ctx.save_for_backward(sigmoid_result)
        return result

    @staticmethod
    def backward(ctx, grad_result):
        (sigmoid_result,) = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = grad_result * sigmoid_result * (1 - sigmoid_result)
        else:
            grad_input = None
        return grad_input


def dynamic_attention(q, k, q_prune, k_prune, v, smooth=None, v2=None):
    # q, k, v: b, c, h, w
    b, c_qk, h_q, w_q = q.shape
    h_kv, w_kv = k.shape[2:]
    q = q.view(b, c_qk, h_q * w_q).transpose(-1, -2).contiguous()
    k = k.view(b, c_qk, h_kv * w_kv)
    v = v.view(b, -1, h_kv * w_kv).transpose(-1, -2).contiguous()
    q_prune = q_prune.view(b, -1, h_q * w_q).transpose(-1, -2).contiguous()
    k_prune = k_prune.view(b, -1, h_kv * w_kv)
    mask = SignWithSigmoidGrad.apply(torch.bmm(q_prune, k_prune) / k_prune.shape[1])
    # q: b, N_q, c_qk
    # k: b, c_qk, N_kv
    # v: b, N_kv, c_v
    if smooth is None:
        smooth = c_qk ** 0.5
    cor_map = torch.bmm(q, k) / smooth
    attn = torch.softmax(cor_map, dim=-1)
    # attn: b, N_q, N_kv
    masked_attn = attn * mask
    output = torch.bmm(masked_attn, v)
    # output: b, N_q, c_v
    output = output.transpose(-1, -2).contiguous().view(b, -1, h_q, w_q)
    conf = masked_attn.sum(-1).view(b, 1, h_q, w_q)
    if v2 is not None:
        v2 = v2.view(b, -1, h_kv * w_kv).transpose(-1, -2).contiguous()
        output2 = torch.bmm(torch.softmax(torch.masked_fill(cor_map, mask.bool(), -1e4), dim=-1),
                            v2).transpose(-1, -2).contiguous().view(b, -1, h_q, w_q)
    else:
        output2 = None
    return output, cor_map, conf, output2


class DynamicTransformerBlock(nn.Module):

    def __init__(self, embed_dim_qk, embed_dim_v, dim_prune, ic, smooth=None):
        super().__init__()
        self.f = nn.Conv2d(embed_dim_qk, embed_dim_qk, (1, 1), (1, 1))
        self.g = nn.Conv2d(embed_dim_qk, embed_dim_qk, (1, 1), (1, 1))
        self.h = nn.Conv2d(embed_dim_v, embed_dim_v, (1, 1), (1, 1))
        self.f_prune = nn.Conv2d(embed_dim_qk, dim_prune, (1, 1), (1, 1))
        self.g_prune = nn.Conv2d(embed_dim_qk, dim_prune, (1, 1), (1, 1))
        self.spade = SPADE(embed_dim_v, ic)
        self.res_block = ResidualBlock(embed_dim_v)
        self.smooth = smooth

    def forward(self, q, k, v, pos, seg_map, v2=None):
        pos = pos.repeat(q.shape[0], 1, 1, 1)
        query = torch.cat([util.feature_normalize(q), pos], dim=1)
        key = torch.cat([util.feature_normalize(k), pos], dim=1)
        attn_output, cor_map, conf, output2 = dynamic_attention(
            self.f(query), self.g(key), self.f_prune(query), self.g_prune(key), self.h(v), self.smooth, v2)
        spade_output = self.spade(q, seg_map)
        y = PositionalNorm2d(attn_output + (1 - conf) * spade_output + q)
        return self.res_block(y), output2, cor_map


def inter_scale_dynamic_sparse_attention(q, k, q_prune, k_prune, v, prev_attn_top_k_idx,
                                         need_idx=True, smooth=None, v2=None):
    # qkv: b, c, H, W
    # prev_attn_idx: b, (H_q // 2) * (W_q // 2), top_k
    c_qk, H_q, W_q = q.shape[1:]
    H_k, W_k = k.shape[2:]
    b, _, top_k = prev_attn_top_k_idx.shape

    # Unfold q
    q_win_fmt = F.unfold(q, kernel_size=2, stride=2).view(b, c_qk, 4, (H_q // 2) * (W_q // 2))
    q_win_fmt = q_win_fmt.permute(0, 3, 2, 1).contiguous()
    q_prune_win_fmt = F.unfold(q_prune, kernel_size=2, stride=2).view(b, -1, 4, (H_q // 2) * (W_q // 2))
    q_prune_win_fmt = q_prune_win_fmt.permute(0, 3, 2, 1).contiguous()
    # q_win_fmt: b, (H_q // 2) * (W_q // 2), 4, c

    # Unfold k
    k_win_fmt = F.unfold(k, kernel_size=2, stride=2).view(b, c_qk, 4, (H_k // 2) * (W_k // 2))
    k_win_fmt = k_win_fmt.permute(0, 3, 2, 1).contiguous()
    k_prune_win_fmt = F.unfold(k_prune, kernel_size=2, stride=2).view(b, -1, 4, (H_k // 2) * (W_k // 2))
    k_prune_win_fmt = k_prune_win_fmt.permute(0, 3, 2, 1).contiguous()
    # k_win_fmt: b, (H_k // 2) * (W_k // 2), 4, c

    # Unfold v
    v_win_fmt = F.unfold(v, kernel_size=2, stride=2).view(b, -1, 4, (H_k // 2) * (W_k // 2))
    v_win_fmt = v_win_fmt.permute(0, 3, 2, 1).contiguous()
    # v_win_fmt: b, (H_k // 2) * (W_k // 2), 4, c

    # Get Local Attention
    batch_idx = torch.arange(0, b, device=prev_attn_top_k_idx.device).view(b, 1, 1)
    prev_attn_idx_ = prev_attn_top_k_idx + batch_idx * (H_k // 2) * (W_k // 2)
    k_win_fmt_ = k_win_fmt.view(-1, 4, c_qk)
    k_prune_win_fmt_ = k_prune_win_fmt.view(b * (H_k // 2) * (W_k // 2), 4, -1)
    q_select_k = k_win_fmt_[prev_attn_idx_.view(-1), :, :].view(b, (H_q // 2) * (W_q // 2), top_k, 4, c_qk)
    q_select_k = q_select_k.flatten(-3, -2).contiguous().transpose(-1, -2).contiguous()
    q_select_k_prune = k_prune_win_fmt_[prev_attn_idx_.view(-1), :, :].view(b, (H_q // 2) * (W_q // 2), top_k, 4, -1)
    q_select_k_prune = q_select_k_prune.flatten(-3, -2).contiguous().transpose(-1, -2).contiguous()
    # q_select_k: b, (H_q // 2) * (W_q // 2), c, top_k * 4
    if smooth is None:
        smooth = c_qk ** 0.5
    attn = torch.softmax(torch.matmul(q_win_fmt, q_select_k) / smooth, dim=-1)
    mask = SignWithSigmoidGrad.apply(torch.matmul(q_prune_win_fmt, q_select_k_prune))
    masked_attn = mask * attn
    # attn: b, (H_q // 2) * (W_q // 2), 4, top_k * 4

    # Gather v according to attn
    v_win_fmt_ = v_win_fmt.view(b * (H_k // 2) * (W_k // 2), 4, -1)
    q_select_v = v_win_fmt_[prev_attn_idx_.view(-1), :, :].view(b, (H_q // 2) * (W_q // 2), top_k, 4, -1)
    q_select_v = q_select_v.flatten(-3, -2).contiguous()
    # q_select_v: b, (H_q // 2) * (W_q // 2), top_k * 4, c
    output = torch.matmul(masked_attn, q_select_v)
    # output: b, (H_q // 2) * (W_q // 2), 4, c
    output = output.permute(0, 3, 2, 1).contiguous().view(b, -1, (H_q // 2) * (W_q // 2)).contiguous()
    output = F.fold(output, output_size=(H_q, W_q), kernel_size=2, stride=2)
    # attn_output: b, c, H_q, W_q
    conf = masked_attn.sum(-1).transpose(-2, -1).contiguous()
    conf = F.fold(conf, output_size=(H_q, W_q), kernel_size=2, stride=2)

    if v2 is not None:
        v2_win_fmt = F.unfold(v2, kernel_size=2, stride=2).view(b, -1, 4, (H_k // 2) * (W_k // 2))
        v2_win_fmt = v2_win_fmt.permute(0, 3, 2, 1).contiguous()
        v2_win_fmt_ = v2_win_fmt.view(b * (H_k // 2) * (W_k // 2), 4, -1)
        q_select_v2 = v2_win_fmt_[prev_attn_idx_.view(-1), :, :].view(b, (H_q // 2) * (W_q // 2), top_k, 4, -1)
        q_select_v2 = q_select_v2.flatten(-3, -2).contiguous()
        cor = torch.matmul(q_win_fmt, q_select_k) / smooth
        warp_attn = torch.softmax(torch.masked_fill(cor, mask.bool(), -1e4), dim=-1)
        output2 = torch.matmul(warp_attn, q_select_v2)
        output2 = output2.permute(0, 3, 2, 1).contiguous().view(b, -1, (H_q // 2) * (W_q // 2)).contiguous()
        output2 = F.fold(output2, output_size=(H_q, W_q), kernel_size=2, stride=2)
    else:
        output2 = None

    if not need_idx:
        this_attn_top_k_idx = None
    else:
        # Get index at this scale
        prev_attn_idx_y = prev_attn_top_k_idx // (W_k // 2)
        prev_attn_idx_x = prev_attn_top_k_idx % (W_k // 2)
        this_attn_idx = torch.stack([2 * prev_attn_idx_y * W_k + 2 * prev_attn_idx_x,
                                     2 * prev_attn_idx_y * W_k + 2 * prev_attn_idx_x + 1,
                                     (2 * prev_attn_idx_y + 1) * W_k + 2 * prev_attn_idx_x,
                                     (2 * prev_attn_idx_y + 1) * W_k + 2 * prev_attn_idx_x + 1], dim=-1)
        # this_attn_idx: b, (H_q // 2) * (W_q // 2), top_k, 4
        this_attn_idx = this_attn_idx.flatten(-2, -1).unsqueeze(2).contiguous().repeat(1, 1, 4, 1)
        # this_attn_idx: b, (H_q // 2) * (W_q // 2), 4, top_k * 4
        _, attn_top_k_idx = torch.topk(attn, top_k, -1)
        # attn_top_k_idx: b, (H_q // 2) * (W_q // 2), 4, top_k
        this_attn_top_k_idx = this_attn_idx.gather(dim=-1, index=attn_top_k_idx)
        # this_attn_top_k_idx: b, (H_q // 2) * (W_q // 2), 4, top_k
        this_attn_top_k_idx = this_attn_top_k_idx.view(b, H_q // 2, W_q // 2, 2, 2, top_k).permute(
            0, 1, 3, 2, 4, 5).contiguous().view(b, H_q * W_q, top_k)
        # this_attn_top_k_idx: b, H_q * W_q, top_k

    return output, this_attn_top_k_idx, conf, output2


def inner_scale_dynamic_sparse_attention(q, k, q_prune, k_prune, v, prev_attn_top_k_idx,
                                         need_idx=True, smooth=None, v2=None):
    # qkv: b, c, H, W
    # prev_attn_idx: b, H_q * W_q, top_k
    c_qk, H_q, W_q = q.shape[1:]
    H_k, W_k = k.shape[2:]
    b, _, top_k = prev_attn_top_k_idx.shape

    # Unfold k
    k_win_fmt = k.view(b, c_qk, H_k * W_k).transpose(-2, -1).contiguous()
    k_prune_win_fmt = k_prune.view(b, -1, H_k * W_k).transpose(-2, -1).contiguous()
    # k_win_fmt: b, H_k * W_k, c_qk

    # Unfold q
    q_win_fmt = q.view(b, c_qk, H_q * W_q).transpose(-2, -1).contiguous()
    q_prune_win_fmt = q_prune.view(b, -1, H_q * W_q).transpose(-2, -1).contiguous()
    # q_win_fmt: b, H_q * W_q, c_qk

    # Unfold v
    v_win_fmt = v.view(b, -1, H_k * W_k).transpose(-2, -1).contiguous()
    # v_win_fmt: b, H_k * W_k, c

    # Get Local Attention
    batch_idx = torch.arange(0, b, device=prev_attn_top_k_idx.device).view(b, 1, 1)
    prev_attn_idx_ = F.pad(prev_attn_top_k_idx.view(b, H_q, W_q, top_k).permute(0, 3, 1, 2).contiguous().float(),
                           [1, 1, 1, 1], mode='replicate')
    prev_attn_idx_ = prev_attn_idx_.permute(0, 2, 3, 1).contiguous().long()
    prev_attn_idx_y = prev_attn_idx_ // W_k
    prev_attn_idx_x = prev_attn_idx_ % W_k
    prev_attn_idx_up_y = prev_attn_idx_y[:, :-2, 1:-1, :]
    prev_attn_idx_up_x = prev_attn_idx_x[:, :-2, 1:-1, :]
    prev_attn_idx_up_down = (prev_attn_idx_up_y + 1).clip(max=H_k - 1) * W_k + prev_attn_idx_up_x
    prev_attn_idx_down_y = prev_attn_idx_y[:, 2:, 1:-1, :]
    prev_attn_idx_down_x = prev_attn_idx_x[:, 2:, 1:-1, :]
    prev_attn_idx_down_up = (prev_attn_idx_down_y - 1).clip(min=0) * W_k + prev_attn_idx_down_x
    prev_attn_idx_left_y = prev_attn_idx_y[:, 1:-1, :-2, :]
    prev_attn_idx_left_x = prev_attn_idx_x[:, 1:-1, :-2, :]
    prev_attn_idx_left_right = prev_attn_idx_left_y * W_k + (prev_attn_idx_left_x + 1).clip(max=W_k - 1)
    prev_attn_idx_right_y = prev_attn_idx_y[:, 1:-1, 2:, :]
    prev_attn_idx_right_x = prev_attn_idx_x[:, 1:-1, 2:, :]
    prev_attn_idx_right_left = prev_attn_idx_right_y * W_k + (prev_attn_idx_right_x - 1).clip(min=0)
    # this_attn_idx: b, H_q * W_q, 5 * top_k
    this_attn_idx = torch.cat([prev_attn_top_k_idx.view(b, H_q, W_q, top_k),
                               prev_attn_idx_up_down, prev_attn_idx_down_up,
                               prev_attn_idx_left_right, prev_attn_idx_right_left], dim=-1).flatten(1, 2).contiguous()
    this_attn_idx_ = this_attn_idx + batch_idx * H_k * W_k
    k_win_fmt_ = k_win_fmt.view(-1, c_qk)
    q_select_k = k_win_fmt_[this_attn_idx_.view(-1), :].view(b, H_q * W_q, top_k * 5, c_qk)
    q_select_k = q_select_k.transpose(-1, -2).contiguous()
    k_prune_win_fmt_ = k_prune_win_fmt.view(b * H_k * W_k, -1)
    q_select_k_prune = k_prune_win_fmt_[this_attn_idx_.view(-1), :].view(b, H_q * W_q, top_k * 5, -1)
    q_select_k_prune = q_select_k_prune.transpose(-1, -2).contiguous()
    # q_select_k: b, H_q * W_q, c_qk, top_k * 5
    if smooth is None:
        smooth = c_qk ** 0.5
    attn = torch.softmax(torch.sum(q_win_fmt.unsqueeze(-1) * q_select_k, dim=-2) / smooth, dim=-1)
    mask = SignWithSigmoidGrad.apply(torch.sum(q_prune_win_fmt.unsqueeze(-1) * q_select_k_prune, dim=-2))
    masked_attn = mask * attn
    # attn: b, H_q * W_q, top_k * 5

    # Gather v according to attn
    v_win_fmt_ = v_win_fmt.view(b * H_k * W_k, -1)
    q_select_v = v_win_fmt_[this_attn_idx_.view(-1), :].view(b, H_q * W_q, top_k * 5, -1)
    # q_select_v: b, H_q * W_q, top_k * 5, c
    output = torch.sum(masked_attn.unsqueeze(-1) * q_select_v, dim=-2)
    # output: b, H_q * W_q, c
    output = output.transpose(-1, -2).contiguous().view(b, -1, H_q, W_q).contiguous()
    # output: b, c, H_q, W_q
    conf = masked_attn.sum(dim=-1).view(b, 1, H_q, W_q)

    if v2 is not None:
        v2_win_fmt = v2.view(b, -1, H_k * W_k).transpose(-2, -1).contiguous()
        v2_win_fmt_ = v2_win_fmt.view(b * H_k * W_k, -1)
        q_select_v2 = v2_win_fmt_[this_attn_idx_.view(-1), :].view(b, H_q * W_q, top_k * 5, -1)
        cor = torch.sum(q_win_fmt.unsqueeze(-1) * q_select_k, dim=-2) / smooth
        warp_attn = torch.softmax(torch.masked_fill(cor, mask.bool(), -1e4), dim=-1)
        output2 = torch.sum(warp_attn.unsqueeze(-1) * q_select_v2, dim=-2)
        output2 = output2.transpose(-1, -2).contiguous().view(b, -1, H_q, W_q).contiguous()
    else:
        output2 = None

    if not need_idx:
        this_attn_top_k_idx = None
    else:
        # attn_top_k_idx: b, H_q * W_q, top_k
        _, attn_top_k_idx = torch.topk(attn, top_k, dim=-1)
        this_attn_top_k_idx = this_attn_idx.gather(dim=-1, index=attn_top_k_idx)

    return output, this_attn_top_k_idx, conf, output2


class DynamicSparseTransformerBlock(nn.Module):

    def __init__(self, embed_dim_qk, embed_dim_v, dim_prune, ic, smooth=None, inter_scale=True):
        super().__init__()
        self.f = nn.Conv2d(embed_dim_qk, embed_dim_qk, (1, 1), (1, 1))
        self.g = nn.Conv2d(embed_dim_qk, embed_dim_qk, (1, 1), (1, 1))
        self.h = nn.Conv2d(embed_dim_v, embed_dim_v, (1, 1), (1, 1))
        self.f_prune = nn.Conv2d(embed_dim_qk, dim_prune, (1, 1), (1, 1))
        self.g_prune = nn.Conv2d(embed_dim_qk, dim_prune, (1, 1), (1, 1))
        self.spade = SPADE(embed_dim_v, ic)
        self.res_block = ResidualBlock(embed_dim_v)
        self.smooth = smooth
        self.inter_scale = inter_scale

    def forward(self, q, k, v, pos, prev_attn_top_k_idx, seg_map, v2=None, need_idx=True):
        # qkv: b, c, H, W
        # prev_attn_idx: b, (H_q // 2) * (W_q // 2), top_k
        pos = pos.repeat(q.shape[0], 1, 1, 1)
        query = torch.cat([util.feature_normalize(q), pos], dim=1)
        key = torch.cat([util.feature_normalize(k), pos], dim=1)
        if self.inter_scale:
            attn_output, this_attn_top_k_idx, conf, output2 = inter_scale_dynamic_sparse_attention(
                self.f(query), self.g(key), self.f_prune(query), self.g_prune(key), self.h(v),
                prev_attn_top_k_idx, need_idx, self.smooth, v2)
        else:
            attn_output, this_attn_top_k_idx, conf, output2 = inner_scale_dynamic_sparse_attention(
                self.f(query), self.g(key), self.f_prune(query), self.g_prune(key), self.h(v),
                prev_attn_top_k_idx, need_idx, self.smooth, v2)
        spade_output = self.spade(q, seg_map)
        y = PositionalNorm2d(attn_output + (1 - conf) * spade_output + q)
        return self.res_block(y), output2, this_attn_top_k_idx
