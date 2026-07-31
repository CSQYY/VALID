# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified Model definition: Removed spx, kept only aux (AU) and x, integrated VideoAEDAmplifier
# Adapted for Multi-Task Hybrid Loss: outputs (logits_main, event_set, logits_au, logits_vid)

import torch
import torch.nn as nn
from functools import partial
import math
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from timesformer.models.vit_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timesformer.models.helpers import load_pretrained
from timesformer.models.vit_utils import DropPath, to_2tuple, trunc_normal_

from einops import rearrange


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


class VideoAEDAmplifier(nn.Module):
    def __init__(self, num_classes: int, mu: float = 2.0, lam: float = 3.0,
                 num_events: int = 8, feat_dim: int = 256, spatial_channels: int = 2048,
                 trainable_params: bool = False):
        super().__init__()
        self.num_classes, self.num_events, self.feat_dim = num_classes, num_events, feat_dim
        self.mu = nn.Parameter(torch.tensor(mu), requires_grad=trainable_params)
        self.lam = nn.Parameter(torch.tensor(lam), requires_grad=trainable_params)
        self.event_queries = nn.Parameter(torch.randn(num_events, feat_dim))
        self.attn = nn.MultiheadAttention(feat_dim, num_heads=4, batch_first=True)
        self.proj_in = nn.Linear(num_classes, feat_dim)
        self.proj_out = nn.Linear(feat_dim, num_classes + 3)

        self.spatial_attn = nn.Sequential(
            nn.Linear(spatial_channels, spatial_channels // 4),
            nn.GELU(),
            nn.Linear(spatial_channels // 4, 1)
        )

    def forward(self, logits: torch.Tensor, motion_intensity: torch.Tensor = None,
                spatial_features: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, K = logits.shape
        device = logits.device
        eps = 1e-5
        probs = F.softmax(logits, dim=-1)

        if motion_intensity is not None:
            norm_intensity = motion_intensity / (motion_intensity.mean(dim=1, keepdim=True) + eps)
            bg_prob = torch.exp(-norm_intensity)
        else:
            entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)
            bg_prob = entropy / torch.log(torch.tensor(K + eps, device=device))

        W_att = F.softmax(probs * probs, dim=-1)
        W_bg_suppressed = bg_prob.unsqueeze(-1).clamp(min=eps) ** self.lam
        probs_refined = self.mu * W_att * probs + probs * W_bg_suppressed
        probs_refined = F.normalize(probs_refined, p=1, dim=-1)

        feats = self.proj_in(probs_refined)
        queries = self.event_queries.unsqueeze(0).expand(B, -1, -1)
        event_feats, _ = self.attn(queries, feats, feats)
        event_out = self.proj_out(event_feats)

        event_dict = {
            'class_logits': event_out[:, :, :K],
            'start_norm': torch.sigmoid(event_out[:, :, K]),
            'end_norm': torch.sigmoid(event_out[:, :, K + 1]),
            'confidence': torch.sigmoid(event_out[:, :, K + 2])
        }

        if spatial_features is not None:
            spatial_weights = self.spatial_attn(spatial_features).squeeze(-1)
            spatial_weights = torch.sigmoid(spatial_weights)
            event_dict['spatial_confidence'] = spatial_weights
        else:
            event_dict['spatial_confidence'] = torch.ones(B, T, 49, device=device)

        return probs_refined, event_dict


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x, aux=None, attn_bias=None):
        if aux is None:
            B, N, C = x.shape
            if self.with_qkv:
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
            else:
                qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
                q, k, v = qkv, qkv, qkv

            attn = (q @ k.transpose(-2, -1)) * self.scale
            if attn_bias is not None:
                if attn_bias.dim() == 2:
                    attn = attn + attn_bias[:, None, :, None]
                else:
                    attn = attn + attn_bias.unsqueeze(1).unsqueeze(2)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            if self.with_qkv:
                x = self.proj(x)
                x = self.proj_drop(x)
        else:
            # Cross-attention with aux (AU features)
            bl, sl, cl = aux.shape
            B, N, C = x.shape
            qs = rearrange(x, '(b h w) t m -> b (h w) t m', b=bl, h=14, w=14, t=N)
            qs = qs.reshape(bl, 196, N, self.num_heads, C // self.num_heads).permute(1, 0, 3, 2, 4)
            kv = self.kv(aux).reshape(bl, sl, 2, self.num_heads, cl // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            merged_in = qs[0]
            for q in qs:
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                attn_output = (attn @ v)

                attn = (merged_in @ attn_output.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                merged_out = (attn @ attn_output)

                merged_in = merged_out

            x = merged_out.transpose(1, 2).reshape(bl, N, C)
            x = x.unsqueeze(0).repeat(196, 1, 1, 1).reshape(B, N, C)
            if self.with_qkv:
                x = self.proj(x)
                x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        super().__init__()
        self.attention_type = attention_type
        assert attention_type in ['divided_space_time', 'space_only', 'joint_space_time']

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn1 = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)
            # Gate weight for AED spatial confidence modulation
            self.aed_gate_weight = nn.Parameter(torch.tensor(0.2))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def _aed_temporal_bias(self, event_set, T, H, W, B):
        """Compute temporal attention bias from AED event boundaries and confidence."""
        start_frames = (event_set['start_norm'] * (T - 1)).long().clamp(0, T - 1)
        end_frames = (event_set['end_norm'] * (T - 1)).long().clamp(0, T - 1)
        conf = event_set['confidence']

        t_idx = torch.arange(T, device=start_frames.device).unsqueeze(0).unsqueeze(-1)
        mask = (t_idx >= start_frames.unsqueeze(1)) & (t_idx <= end_frames.unsqueeze(1))
        frame_mask = (mask.float() * conf.unsqueeze(1)).sum(dim=2)  # [B, T]
        frame_mask = torch.tanh(frame_mask * 2 - 1)

        # FIX: Keep 3D shape [B, T, H*W] instead of flattening to 2D
        bias_spatial = frame_mask.unsqueeze(-1).expand(-1, -1, H * W)  # [B, T, H*W]

        # Prepend zero bias for CLS token: [B, T, 1 + H*W]
        cls_bias = torch.zeros(B, T, 1, device=bias_spatial.device, dtype=bias_spatial.dtype)
        bias_with_cls = torch.cat([cls_bias, bias_spatial], dim=2)  # [B, T, 1 + H*W]

        # Reshape to [B*T, 1 + H*W] to match attn Key dimension
        bias_flat = bias_with_cls.reshape(B * T, 1 + H * W)

        return bias_flat

    def forward(self, x, B, T, W, aux, event_set=None):
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, aux

        elif self.attention_type == 'divided_space_time':
            ## Temporal: cross-attention with AU features
            xt = x[:, 1:, :]
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m', b=B, h=H, w=W, t=T)

            aux_normed = self.temporal_norm1(aux)
            xtn = self.temporal_norm1(xt)

            res_temporal = self.drop_path(self.temporal_attn1(xtn, aux_normed))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m', b=B, h=H, w=W, t=T)
            res_temporal = self.temporal_fc(res_temporal)

            xt = x[:, 1:, :] + res_temporal

            ## Spatial with AED gating
            xs = rearrange(xt, 'b (h w t) m -> (b t) (h w) m', b=B, h=H, w=W, t=T)

            # Apply AED spatial confidence gate
            if event_set is not None and 'spatial_confidence' in event_set:
                spatial_conf = event_set['spatial_confidence'].reshape(B * T, H * W, 1)
                event_conf_mean = event_set['confidence'].mean(dim=1, keepdim=True)
                event_conf_expanded = event_conf_mean.repeat_interleave(T, dim=0).unsqueeze(-1)
                gate = 1 + self.aed_gate_weight * (spatial_conf * event_conf_expanded)
                xs = xs * gate

            # Compute AED temporal attention bias
            attn_bias = None
            if event_set is not None:
                attn_bias = self._aed_temporal_bias(event_set, T, H, W, B)

            init_cls_token = x[:, 0, :].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(cls_token, 'b t m -> (b t) m', b=B, t=T).unsqueeze(1)
            xs_cat = torch.cat((cls_token, xs), 1)
            res_spatial = self.drop_path(self.attn(self.norm1(xs_cat), attn_bias=attn_bias))

            cls_token = res_spatial[:, 0, :]
            cls_token = rearrange(cls_token, '(b t) m -> b t m', b=B, t=T)
            cls_token = torch.mean(cls_token, 1, True)
            res_spatial = res_spatial[:, 1:, :]
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m', b=B, h=H, w=W, t=T)

            x = xt
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res_spatial), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))

            return x, aux


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, T, W


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, normalized_time: torch.Tensor):
        indices = (normalized_time * (self.pe.size(0) - 1)).long().clamp(0, self.pe.size(0) - 1)
        return self.pe[indices]


class PerPartEventFusion(nn.Module):
    def __init__(self, input_dim: int = 4, d_model: int = 256, nhead: int = 8, num_layers: int = 2,
                 dropout: float = 0.1, max_events: int = 35, use_output_proj: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.max_events = max_events
        self.use_output_proj = use_output_proj

        self.event_proj = nn.Linear(input_dim, d_model)
        self.norm_input = nn.LayerNorm(d_model)
        self.temp_pe = TemporalPositionalEncoding(d_model, max_len=max_events)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.output_proj = nn.Linear(d_model, input_dim) if use_output_proj else None

    def forward(self, events: torch.Tensor, event_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, n_parts, T, D = events.shape
        x = events.view(B * n_parts, T, D)

        if event_mask is not None:
            mask_flat = event_mask.view(B * n_parts, T)
        else:
            mask_flat = torch.ones(B * n_parts, T, dtype=torch.bool, device=events.device)

        x = self.norm_input(self.event_proj(x))

        time_steps = torch.arange(T, device=events.device, dtype=torch.float)
        norm_time = time_steps / max(T - 1, 1) if T > 1 else torch.zeros_like(time_steps)
        norm_time = norm_time.unsqueeze(0).expand(B * n_parts, -1)

        pos_enc = self.temp_pe(norm_time)
        x = x + pos_enc

        x = self.transformer(x, src_key_padding_mask=None)

        scores = self.attn_pool(x).squeeze(-1)
        if event_mask is not None:
            scores = scores.masked_fill(~mask_flat, -1e4)
        weights = torch.softmax(scores, dim=-1)
        fused_hidden = torch.bmm(weights.unsqueeze(1), x).squeeze(1)

        if self.output_proj is not None:
            fused = self.output_proj(fused_hidden)
        else:
            fused = fused_hidden

        output_dim = self.input_dim if self.output_proj is not None else self.d_model
        fused = fused.view(B, n_parts, output_dim)
        return fused


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=8,
                 attention_type='divided_space_time', dropout=0.,
                 aed_num_classes=52, aed_num_events=8, aed_trainable=False):
        super().__init__()
        self.attention_type = attention_type
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                attention_type=self.attention_type)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

        # ✅ Main classification head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # ✅ Auxiliary heads for multi-task hybrid loss
        self.aux_head_au = nn.Linear(embed_dim, num_classes)
        self.aux_head_vid = nn.Linear(embed_dim, num_classes)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        # AU event fusion
        self.TSembed1 = PerPartEventFusion(input_dim=4, d_model=embed_dim)

        # AED Amplifier and its dependencies
        self.feat_adapter = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.feat_adapter.weight, mode='fan_out', nonlinearity='relu')
        self.aed_pseudo_head = nn.Linear(embed_dim, aed_num_classes)
        nn.init.xavier_uniform_(self.aed_pseudo_head.weight)
        self.aed_amplifier = VideoAEDAmplifier(
            num_classes=aed_num_classes, mu=2.0, lam=3.0, num_events=aed_num_events,
            feat_dim=embed_dim, spatial_channels=embed_dim, trainable_params=aed_trainable
        )

        if self.attention_type == 'divided_space_time':
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                        nn.init.constant_(m.temporal_fc.weight, 0)
                        nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.aux_head_au = nn.Linear(self.embed_dim, num_classes)
        self.aux_head_vid = nn.Linear(self.embed_dim, num_classes)

    def _compute_motion_intensity(self, x):
        """Compute frame-level motion intensity from raw video frames."""
        diff = torch.diff(x, dim=2)  # temporal diff along T dimension
        intensity = torch.norm(diff, dim=1).mean(dim=[2, 3])  # norm over C, avg over H,W
        return F.pad(intensity, (1, 0), value=0)

    def forward_features(self, data):
        x = data[0]       # video frames: (B, C, T, H, W)
        aux = data[1]     # AU features only
        B = x.shape[0]

        # --- AED: compute motion intensity and pseudo logits before patch_embed ---
        motion_intensity = self._compute_motion_intensity(x)

        # Get spatial features for AED after patch_embed but before flattening
        x_patch, T, W = self.patch_embed(x)  # (B*T, num_patches, embed_dim)
        H = x_patch.size(1) // W  # spatial height in tokens

        # Reshape to (B, T, H*W, embed_dim) for AED spatial attention
        x_spatial_for_aed = x_patch.reshape(B, T, H * W, self.embed_dim)

        # Global average pooled features for pseudo classification
        x_global = x_patch.mean(dim=1)  # (B*T, embed_dim)
        x_global = x_global.reshape(B, T, self.embed_dim)
        pseudo_logits = self.aed_pseudo_head(x_global)  # (B, T, aed_num_classes)

        _, event_set = self.aed_amplifier(
            pseudo_logits, motion_intensity, spatial_features=x_spatial_for_aed
        )

        # --- AU encoding ---
        au_mask = (aux != 0).any(dim=-1)
        au_mask[:, :, 0] = True
        aux_enc = self.TSembed1(aux, au_mask)  # (B, n_parts, embed_dim)

        # --- Continue standard ViT forward ---
        cls_tokens = self.cls_token.expand(x_patch.size(0), -1, -1)
        x = torch.cat((cls_tokens, x_patch), dim=1)

        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H_pos = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H_pos, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2).transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        if self.attention_type != 'space_only':
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:, 1:]
            x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=T, mode='nearest').transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)
            x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x, aux_enc = blk(x, B, T, W, aux_enc, event_set=event_set)

        if self.attention_type == 'space_only':
            x = rearrange(x, '(b t) n m -> b t n m', b=B, t=T)
            x = torch.mean(x, 1)

        x = self.norm(x)

        # ✅ Extract features for all branches
        cls_feat = x[:, 0]                                          # (B, embed_dim)
        au_feat_pooled = torch.mean(aux_enc, dim=1) if aux_enc.dim() == 3 else aux_enc  # (B, embed_dim)

        # ✅ Compute logits for all branches
        logits_au = self.aux_head_au(au_feat_pooled)   # (B, num_classes)
        logits_vid = self.aux_head_vid(cls_feat)       # (B, num_classes)

        return cls_feat, event_set, logits_au, logits_vid

    def forward(self, x):
        cls_feat, event_set, logits_au, logits_vid = self.forward_features(x)
        logits_main = self.head(cls_feat)
        # ✅ Return 4-tuple for multi-task hybrid loss
        return logits_main, event_set, logits_au, logits_vid


def _conv_filter(state_dict, patch_size=16):
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            if v.shape[-1] != patch_size:
                patch_size = v.shape[-1]
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


class vit_base_patch16_224(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(vit_base_patch16_224, self).__init__()
        self.pretrained = True
        patch_size = 16
        self.model = VisionTransformer(
            img_size=cfg.DATA.TRAIN_CROP_SIZE, num_classes=cfg.MODEL.NUM_CLASSES,
            patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
            num_frames=cfg.DATA.NUM_FRAMES, attention_type=cfg.TIMESFORMER.ATTENTION_TYPE, **kwargs)

        self.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (cfg.DATA.TRAIN_CROP_SIZE // patch_size) * (cfg.DATA.TRAIN_CROP_SIZE // patch_size)
        pretrained_model = cfg.TIMESFORMER.PRETRAINED_MODEL
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes,
                            in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter,
                            img_size=cfg.DATA.TRAIN_CROP_SIZE, num_patches=self.num_patches,
                            attention_type=self.attention_type, pretrained_model=pretrained_model)

    def forward(self, x):
        return self.model(x)


class TimeSformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=2, num_frames=64,
                 attention_type='divided_space_time', pretrained_model='', **kwargs):
        super(TimeSformer, self).__init__()
        self.pretrained = True
        self.model = VisionTransformer(
            img_size=img_size, num_classes=num_classes, patch_size=patch_size, embed_dim=768,
            depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
            num_frames=num_frames, attention_type=attention_type, **kwargs)

        self.attention_type = attention_type
        self.model.default_cfg = default_cfgs['vit_base_patch' + str(patch_size) + '_224']
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes,
                            in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter,
                            img_size=img_size, num_frames=num_frames, num_patches=self.num_patches,
                            attention_type=self.attention_type, pretrained_model=pretrained_model)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   total_params:{total_params:,}")
        print("=" * 60)

    def forward(self, x):
        return self.model(x)