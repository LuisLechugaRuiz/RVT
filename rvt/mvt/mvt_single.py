# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from math import ceil

import torch
import torch.nn.functional as F

from torch import nn
from einops import rearrange, repeat

from rvt.mvt.attn import (
    Conv2DBlock,
    Conv2DUpsampleBlock,
    PreNorm,
    Attention,
    MemoryAttention,
    cache_fn,
    DenseBlock,
    FeedForward,
)

from general_manipulation.utils.video_recorder import VideoRecorder
from general_manipulation.models.act_cvae import ACTCVAE


class MVT(nn.Module):
    def __init__(
        self,
        depth,
        img_size,
        add_proprio,
        proprio_dim,
        add_lang,
        lang_dim,
        lang_len,
        img_feat_dim,
        feat_dim,
        im_channels,
        attn_dim,
        attn_heads,
        attn_dim_head,
        activation,
        weight_tie_layers,
        attn_dropout,
        decoder_dropout,
        img_patch_size,
        final_dim,
        self_cross_ver,
        add_corr,
        add_pixel_loc,
        add_depth,
        pe_fix,
        act_cfg_dict,
        renderer_device="cuda:0",
        renderer=None,
    ):
        """MultiView Transfomer

        :param depth: depth of the attention network
        :param img_size: number of pixels per side for rendering
        :param renderer_device: device for placing the renderer
        :param add_proprio:
        :param proprio_dim:
        :param add_lang:
        :param lang_dim:
        :param lang_len:
        :param img_feat_dim:
        :param feat_dim:
        :param im_channels: intermediate channel size
        :param attn_dim:
        :param attn_heads:
        :param attn_dim_head:
        :param activation:
        :param weight_tie_layers:
        :param attn_dropout:
        :param decoder_dropout:
        :param img_patch_size: intial patch size
        :param final_dim: final dimensions of features
        :param self_cross_ver:
        :param add_corr:
        :param add_pixel_loc:
        :param add_depth:
        :param pe_fix: matter only when add_lang is True
            Either:
                True: use position embedding only for image tokens
                False: use position embedding for lang and image token
        """

        super().__init__()
        self.depth = depth
        self.img_feat_dim = img_feat_dim
        self.img_size = img_size
        self.add_proprio = add_proprio
        self.proprio_cartesian_dim = proprio_dim
        self.add_lang = add_lang
        self.lang_dim = lang_dim
        self.lang_len = lang_len
        self.im_channels = im_channels
        self.img_patch_size = img_patch_size
        self.final_dim = final_dim
        self.attn_dropout = attn_dropout
        self.decoder_dropout = decoder_dropout
        self.self_cross_ver = self_cross_ver
        self.add_corr = add_corr
        self.add_pixel_loc = add_pixel_loc
        self.add_depth = add_depth
        self.pe_fix = pe_fix
        self.attn_heads = attn_heads

        # ACT
        self.proprio_joint_pos_dim = act_cfg_dict["proprio_dim"]
        self.dim_feedforward = act_cfg_dict["dim_feedforward"]
        self.normalize_before = act_cfg_dict["normalize_before"]
        self.num_decoder_layers = act_cfg_dict["num_decoder_layers"]
        self.num_queries = act_cfg_dict["num_queries"]
        self.state_dim = act_cfg_dict["state_dim"]
        self.debug = act_cfg_dict["debug"]

        print(f"MVT Vars: {vars(self)}")

        assert renderer is not None
        self.renderer = renderer
        self.num_img = self.renderer.num_img

        # patchified input dimensions
        spatial_size = img_size // self.img_patch_size  # 220 / 11 = 20

        if self.add_proprio:
            # 64 img features + 128 proprio features (64 cartesian + 64 joint pos)
            self.input_dim_before_seq = self.im_channels * 3
        else:
            self.input_dim_before_seq = self.im_channels

        # learnable positional encoding
        if add_lang:
            lang_emb_dim, lang_max_seq_len = lang_dim, lang_len
        else:
            lang_emb_dim, lang_max_seq_len = 0, 0
        self.lang_emb_dim = lang_emb_dim
        self.lang_max_seq_len = lang_max_seq_len

        if self.pe_fix:
            num_pe_token = spatial_size**2 * self.num_img
        else:
            num_pe_token = lang_max_seq_len + (spatial_size**2 * self.num_img)
        self.pos_encoding = nn.Parameter(
            torch.randn(
                1,
                num_pe_token,
                self.input_dim_before_seq,
            )
        )

        inp_img_feat_dim = self.img_feat_dim
        if self.add_corr:
            inp_img_feat_dim += 3
        if self.add_pixel_loc:
            inp_img_feat_dim += 3
            self.pixel_loc = torch.zeros(
                (self.num_img, 3, self.img_size, self.img_size)
            )
            self.pixel_loc[:, 0, :, :] = (
                torch.linspace(-1, 1, self.num_img).unsqueeze(-1).unsqueeze(-1)
            )
            self.pixel_loc[:, 1, :, :] = (
                torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(-1)
            )
            self.pixel_loc[:, 2, :, :] = (
                torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(0)
            )
        if self.add_depth:
            inp_img_feat_dim += 1

        # img input preprocessing encoder
        self.input_preprocess = Conv2DBlock(
            inp_img_feat_dim,
            self.im_channels,
            kernel_sizes=1,
            strides=1,
            norm=None,
            activation=activation,
        )
        inp_pre_out_dim = self.im_channels

        if self.add_proprio:
            # proprio preprocessing encoder
            self.proprio_cartesian_preprocess = DenseBlock(
                self.proprio_cartesian_dim,
                self.im_channels,
                norm="group",
                activation=activation,
            )
            # proprio preprocessing encoder
            self.proprio_joint_pos_preprocess = DenseBlock(
                self.proprio_joint_pos_dim,
                self.im_channels,
                norm="group",
                activation=activation,
            )

        self.patchify = Conv2DBlock(
            inp_pre_out_dim,
            self.im_channels,
            kernel_sizes=self.img_patch_size,
            strides=self.img_patch_size,
            norm="group",
            activation=activation,
            padding=0,
        )

        # lang preprocess
        if self.add_lang:
            self.lang_preprocess = DenseBlock(
                lang_emb_dim,
                self.input_dim_before_seq,
                norm="group",
                activation=activation,
            )

        self.fc_bef_attn = DenseBlock(
            self.input_dim_before_seq,
            attn_dim,
            norm=None,
            activation=None,
        )
        self.fc_aft_attn = DenseBlock(
            attn_dim,
            self.input_dim_before_seq,
            norm=None,
            activation=None,
        )

        get_attn_attn = lambda: PreNorm(
            attn_dim,
            MemoryAttention(
                attn_dim,
                heads=attn_heads,
                dim_head=attn_dim_head,
                dropout=attn_dropout,
            ),
        )
        get_attn_ff = lambda: PreNorm(attn_dim, FeedForward(attn_dim))
        get_attn_attn, get_attn_ff = map(cache_fn, (get_attn_attn, get_attn_ff))
        # self-attention layers
        self.layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}
        attn_depth = depth

        for _ in range(attn_depth):
            self.layers.append(
                nn.ModuleList([get_attn_attn(**cache_args), get_attn_ff(**cache_args)])
            )

        self.up0 = Conv2DUpsampleBlock(
            self.input_dim_before_seq,
            self.im_channels,
            kernel_sizes=self.img_patch_size,
            strides=self.img_patch_size,
            norm=None,
            activation=activation,
        )

        final_inp_dim = self.im_channels + inp_pre_out_dim

        # final layers
        self.final = Conv2DBlock(
            final_inp_dim,
            self.im_channels,
            kernel_sizes=3,
            strides=1,
            norm=None,
            activation=activation,
        )

        self.trans_decoder = Conv2DBlock(
            self.final_dim,
            1,
            kernel_sizes=3,
            strides=1,
            norm=None,
            activation=None,
        )

        feat_out_size = feat_dim
        feat_fc_dim = 0
        feat_fc_dim += self.input_dim_before_seq
        feat_fc_dim += self.final_dim

        self.feat_fc = nn.Sequential(
            nn.Linear(self.num_img * feat_fc_dim, feat_fc_dim),
            nn.ReLU(),
            nn.Linear(feat_fc_dim, feat_fc_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_fc_dim // 2, feat_out_size),
        )

        self.pos_embed_decoder = nn.Parameter(
            torch.randn(
                1,
                num_pe_token,
                attn_dim,
            )
        )

        self.query_embed = nn.Embedding(self.num_queries, attn_dim)
        self.decoder = ACTCVAE.build_decoder(
            hidden_dim=attn_dim,
            dropout=attn_dropout,
            nhead=attn_heads,
            dim_feedforward=self.dim_feedforward,
            num_decoder_layers=self.num_decoder_layers,
            normalize_before=self.normalize_before,
        )
        self.action_head = nn.Linear(attn_dim, self.state_dim)
        self.is_pad_head = nn.Linear(attn_dim, 1)

        self.video_recorder = VideoRecorder(num_img=self.num_img)

    def get_pt_loc_on_img(self, pt, dyn_cam_info):
        """
        transform location of points in the local frame to location on the
        image
        :param pt: (bs, np, 3)
        :return: pt_img of size (bs, np, num_img, 2)
        """
        pt_img = self.renderer.get_pt_loc_on_img(
            pt, fix_cam=True, dyn_cam_info=dyn_cam_info
        )
        return pt_img

    def forward(
        self,
        img,
        proprio_cartesian=None,
        proprio_joint_pos=None,
        lang_emb=None,
        terminal=None,
    ):
        """
        :param img: tensor of shape (bs, num_img, img_feat_dim, h, w)
        :param proprio_cartesian: tensor of shape (bs, proprio_cartesian_dim)
        :param proprio_joint_pos: tensor of shape (bs, proprio_joint_pos_dim)
        :param lang_emb: tensor of shape (bs, lang_len, lang_dim)
        :param terminal: tensor of shape (bs, 1)
        """

        original_img = img
        bs, num_img, img_feat_dim, h, w = img.shape
        num_pat_img = h // self.img_patch_size
        assert num_img == self.num_img
        # assert img_feat_dim == self.img_feat_dim
        assert h == w == self.img_size

        img = img.view(bs * num_img, img_feat_dim, h, w)
        # preprocess
        # (bs * num_img, im_channels, h, w)
        d0 = self.input_preprocess(img)

        # (bs * num_img, im_channels, h, w) ->
        # (bs * num_img, im_channels, h / img_patch_strid, w / img_patch_strid) patches
        ins = self.patchify(d0)
        # (bs, im_channels, num_img, h / img_patch_strid, w / img_patch_strid) patches
        ins = (
            ins.view(
                bs,
                num_img,
                self.im_channels,
                num_pat_img,
                num_pat_img,
            )
            .transpose(1, 2)
            .clone()
        )

        # concat proprio
        _, _, _d, _h, _w = ins.shape
        if self.add_proprio:

            def expand_dims(tensor, d, h, w):
                return (
                    tensor.unsqueeze(-1)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                    .repeat(1, 1, d, h, w)
                )

            p_cartesian = self.proprio_cartesian_preprocess(
                proprio_cartesian
            )  # [B,4] -> [B,64]
            p_cartesian = expand_dims(p_cartesian, _d, _h, _w)

            p_joint_pos = self.proprio_joint_pos_preprocess(
                proprio_joint_pos
            )  # [B,7] -> [B,64]
            p_joint_pos = expand_dims(p_joint_pos, _d, _h, _w)

            ins = torch.cat(
                [ins, p_cartesian, p_joint_pos], dim=1
            )  # [B, 192, num_img, np, np]

        # channel last
        ins = rearrange(ins, "b d ... -> b ... d")  # [B, num_img, np, np, 192]

        # save original shape of input for layer
        ins_orig_shape = ins.shape

        # flatten patches into sequence
        ins = rearrange(ins, "b ... d -> b (...) d")  # [B, num_img * np * np, 192]
        # add learable pos encoding
        # only added to image tokens
        if self.pe_fix:
            ins += self.pos_encoding

        # append language features as sequence
        num_lang_tok = 0
        if self.add_lang:
            l = self.lang_preprocess(
                lang_emb.view(bs * self.lang_max_seq_len, self.lang_emb_dim)
            )
            l = l.view(bs, self.lang_max_seq_len, -1)
            num_lang_tok = l.shape[1]
            l_repeated = l.unsqueeze(1).repeat(1, num_img, 1, 1).view(bs, num_img * num_lang_tok, -1)
            ins = torch.cat((l_repeated, ins), dim=1)  # [B, num_img * np * np + num_img * lang_max_seq_len, 192]

        # add learable pos encoding
        if not self.pe_fix:
            ins = ins + self.pos_encoding

        x = self.fc_bef_attn(ins)
        if self.self_cross_ver == 0:
            # self-attention layers
            for self_attn, self_ff in self.layers:
                x = self_attn(x) + x
                x = self_ff(x) + x

        elif self.self_cross_ver == 1:
            # lx, imgx = x[:, :num_lang_tok], x[:, num_lang_tok:] -> Use tokens for each image.

            # within image self attention
            attention_weights = []
            # print("X SHAPE:", x.shape)
            x = x.reshape(bs * num_img, num_pat_img * num_pat_img + num_lang_tok, -1)
            for self_attn, self_ff in self.layers[: len(self.layers) // 2]:
                out, attn_weight = self_attn(x, get_weights=True)
                attention_weights.append(attn_weight.detach())
                x = out + x
                x = self_ff(x) + x

            x = x.view(bs, num_img * (num_pat_img * num_pat_img + num_lang_tok), -1)
            # x = torch.cat((lx, imgx), dim=1)
            # cross attention
            for self_attn, self_ff in self.layers[len(self.layers) // 2 :]:
                x = self_attn(x) + x
                x = self_ff(x) + x

            reset_memory = torch.nonzero(terminal)
            for i in reset_memory:
                for attn, ff in self.layers:
                    attn.fn.reset_memory(i)

        else:
            assert False

        if self.add_lang:
            # throwing away the language embeddings
            x = x[:, num_img * num_lang_tok:]

        # -- Two decoders ---

        # First one for sequence of joint absolute positions
        memory = x.transpose(0, 1)

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)
        pos = self.pos_embed_decoder.transpose(0, 1).repeat(1, bs, 1)

        hs = self.decoder(tgt=tgt, memory=memory, pos=pos, query_pos=query_embed)
        hs = hs.transpose(1, 2)[0]  # Get last layer output

        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)

        # Second one for 3D translation - rotation
        x = self.fc_aft_attn(x)

        # reshape back to orginal size
        x = x.view(bs, *ins_orig_shape[1:-1], x.shape[-1])  # [B, num_img, np, np, 192]
        x = rearrange(x, "b ... d -> b d ...")  # [B, 192, num_img, np, np]

        feat = []
        _feat = torch.max(torch.max(x, dim=-1)[0], dim=-1)[0]
        _feat = _feat.view(bs, -1)
        feat.append(_feat)

        x = (
            x.transpose(1, 2)
            .clone()
            .view(
                bs * self.num_img, self.input_dim_before_seq, num_pat_img, num_pat_img
            )
        )

        u0 = self.up0(x)
        u0 = torch.cat([u0, d0], dim=1)
        u = self.final(u0)

        # translation decoder
        trans = self.trans_decoder(u).view(bs, self.num_img, h, w)

        hm = F.softmax(trans.detach().view(bs, self.num_img, h * w), 2).view(
            bs * self.num_img, 1, h, w
        )

        if self.debug:
            debug_hm = hm.view(bs, self.num_img, h, w)
            self.video_recorder.record(
                img=original_img,
                attn=attention_weights,
                num_pat_img=num_pat_img,
                num_heads=self.attn_heads,
                heatmap=debug_hm,
            )

        _feat = torch.sum(hm * u, dim=[2, 3])
        _feat = _feat.view(bs, -1)
        feat.append(_feat)
        feat = torch.cat(feat, dim=-1)
        feat = self.feat_fc(feat)

        out = {"trans": trans, "feat": feat, "actions": a_hat, "is_pad": is_pad_hat}

        return out

    def get_wpt(self, out, dyn_cam_info, y_q=None):
        """
        Estimate the q-values given output from mvt
        :param out: output from mvt
        """
        nc = self.num_img
        h = w = self.img_size
        bs = out["trans"].shape[0]

        q_trans = out["trans"].view(bs, nc, h * w)
        hm = torch.nn.functional.softmax(q_trans, 2)
        hm = hm.view(bs, nc, h, w)

        if dyn_cam_info is None:
            dyn_cam_info_itr = (None,) * bs
        else:
            dyn_cam_info_itr = dyn_cam_info

        pred_wpt = [
            self.renderer.get_max_3d_frm_hm_cube(
                hm[i : i + 1],
                fix_cam=True,
                dyn_cam_info=dyn_cam_info_itr[i : i + 1]
                if not (dyn_cam_info_itr[i] is None)
                else None,
            )
            for i in range(bs)
        ]
        pred_wpt = torch.cat(pred_wpt, 0)

        assert y_q is None

        return pred_wpt

    def free_mem(self):
        """
        Could be used for freeing up the memory once a batch of testing is done
        """
        print("Freeing up some memory")
        self.renderer.free_mem()
