# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import copy
import torch

from torch import nn

from rvt.mvt.mvt_single import MVT as MVTSingle
from rvt.mvt.config import get_cfg_defaults
from rvt.mvt.renderer import BoxRenderer

from general_manipulation.models.act_model import ACTModel

# TODO: REMOVE ME LATER
import cv2
import numpy as np


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
    ):
        """MultiView Transfomer"""
        super().__init__()

        # creating a dictonary of all the input parameters
        args = copy.deepcopy(locals())
        del args["self"]
        del args["__class__"]
        del args["act_cfg_dict"]  # TODO: a bit hacky, clean it when refactoring code.

        # for verifying the input
        self.img_feat_dim = img_feat_dim
        self.add_proprio = add_proprio
        self.proprio_dim = proprio_dim
        self.add_lang = add_lang
        if add_lang:
            lang_emb_dim, lang_max_seq_len = lang_dim, lang_len
        else:
            lang_emb_dim, lang_max_seq_len = 0, 0
        self.lang_emb_dim = lang_emb_dim
        self.lang_max_seq_len = lang_max_seq_len

        self.renderer = BoxRenderer(
            device=renderer_device,
            img_size=(img_size, img_size),
            with_depth=add_depth,
        )
        self.num_img = self.renderer.num_img
        self.img_size = img_size

        self.mvt1 = MVTSingle(**args, renderer=self.renderer)
        self.act_model = ACTModel(act_cfg_dict, num_img=self.num_img)

        # ONLY TO DEBUG -> REMOVE ME LATER!
        self.frames = {}
        self.img_frames = {}
        self.num_steps = 100
        self.output_filename = 'output.avi'

    def get_pt_loc_on_img(self, pt, dyn_cam_info, out=None):
        """
        :param pt: point for which location on image is to be found. the point
            shoud be in the same reference frame as wpt_local (see forward()),
            even for mvt2
        :param out: output from mvt, when using mvt2, we also need to provide the
            origin location where where the point cloud needs to be shifted
            before estimating the location in the image
        """
        assert len(pt.shape) == 3
        bs, np, x = pt.shape
        assert x == 3
        assert out is None
        out = self.mvt1.get_pt_loc_on_img(pt, dyn_cam_info)

        return out

    def get_wpt(self, out, dyn_cam_info, y_q=None):
        """
        Estimate the q-values given output from mvt
        :param out: output from mvt
        :param y_q: refer to the definition in mvt_single.get_wpt
        """
        wpt = self.mvt1.get_wpt(out, dyn_cam_info, y_q)
        return wpt

    def render(self, pc, img_feat, img_aug, dyn_cam_info):
        mvt = self.mvt1

        with torch.no_grad():
            if dyn_cam_info is None:
                dyn_cam_info_itr = (None,) * len(pc)
            else:
                dyn_cam_info_itr = dyn_cam_info

            if mvt.add_corr:
                img = [
                    self.renderer(
                        _pc,
                        torch.cat((_pc, _img_feat), dim=-1),
                        fix_cam=True,
                        dyn_cam_info=(_dyn_cam_info,)
                        if not (_dyn_cam_info is None)
                        else None,
                    ).unsqueeze(0)
                    for (_pc, _img_feat, _dyn_cam_info) in zip(
                        pc, img_feat, dyn_cam_info_itr
                    )
                ]
            else:
                img = [
                    self.renderer(
                        _pc,
                        _img_feat,
                        fix_cam=True,
                        dyn_cam_info=(_dyn_cam_info,)
                        if not (_dyn_cam_info is None)
                        else None,
                    ).unsqueeze(0)
                    for (_pc, _img_feat, _dyn_cam_info) in zip(
                        pc, img_feat, dyn_cam_info_itr
                    )
                ]

            img = torch.cat(img, 0)
            img = img.permute(0, 1, 4, 2, 3)

            # for visualization purposes
            if mvt.add_corr:
                mvt.img = img[:, :, 3:].clone().detach()
            else:
                mvt.img = img.clone().detach()

            # image augmentation
            if img_aug != 0:
                stdv = img_aug * torch.rand(1, device=img.device)
                # values in [-stdv, stdv]
                noise = stdv * ((2 * torch.rand(*img.shape, device=img.device)) - 1)
                img = torch.clamp(img + noise, -1, 1)

            if mvt.add_pixel_loc:
                bs = img.shape[0]
                pixel_loc = mvt.pixel_loc.to(img.device)
                img = torch.cat(
                    (img, pixel_loc.unsqueeze(0).repeat(bs, 1, 1, 1, 1)), dim=2
                )

        return img

    def verify_inp(
        self,
        pc,
        img_feat,
        proprio,
        lang_emb,
        img_aug,
    ):
        if not self.training:
            # no img_aug when not training
            assert img_aug == 0

        bs = len(pc)
        assert bs == len(img_feat)

        for _pc, _img_feat in zip(pc, img_feat):
            np, x1 = _pc.shape
            np2, x2 = _img_feat.shape

            assert np == np2
            assert x1 == 3
            assert x2 == self.img_feat_dim

        if self.add_proprio:
            bs3, x3 = proprio.shape
            assert bs == bs3
            assert (
                x3 == self.proprio_dim
            ), "Does not support proprio of shape {proprio.shape}"
        else:
            assert proprio is None, "Invalid input for proprio={proprio}"

        if self.add_lang:
            bs4, x4, x5 = lang_emb.shape
            assert bs == bs4
            assert (
                x4 == self.lang_max_seq_len
            ), "Does not support lang_emb of shape {lang_emb.shape}"
            assert (
                x5 == self.lang_emb_dim
            ), "Does not support lang_emb of shape {lang_emb.shape}"
        else:
            assert (lang_emb is None) or (
                torch.all(lang_emb == 0)
            ), f"Invalid input for lang={lang}"

    def forward(
        self,
        pc,
        img_feat,
        proprio=None,
        lang_emb=None,
        img_aug=0,
        proprio_joint_abs=None,
        actions=None,
        is_pad=None
    ):
        """
        :param pc: list of tensors, each tensor of shape (num_points, 3)
        :param img_feat: list tensors, each tensor of shape
            (bs, num_points, img_feat_dim)
        :param proprio: tensor of shape (bs, proprio_dim)
        :param lang_emb: tensor of shape (bs, lang_len, lang_dim)
        :param img_aug: (float) magnitude of augmentation in rgb image
        :param proprio_joint_abs: tensor of shape (bs, proprio_joint_abs_dim)
        :param actions: tensor of shape (batch, seq, action_dim)
        :param is_pad: tensor of shape (batch, seq, 1)
        """

        self.verify_inp(pc, img_feat, proprio, lang_emb, img_aug)
        img = self.render(
            pc,
            img_feat,
            img_aug,
            dyn_cam_info=None,
        )
        mvt_out = self.mvt1(img=img, proprio=proprio, lang_emb=lang_emb)
        img = self.add_hm(img, mvt_out["hm"])
        act_out = self.act_model(img=img, proprio=proprio_joint_abs, actions=actions, is_pad=is_pad)
        return mvt_out, act_out

    def add_hm(self, img, hm):
        bs = img.shape[0]
        num_channels = hm.shape[1]
        height = hm.shape[2]
        width = hm.shape[3]

        hm = hm.view(
            bs, self.num_img, num_channels, height, width
        )  # [bs * self.num_img, 1, h, w] -> [bs, self.num_img, 1, h, w]
        img = torch.cat((img, hm), dim=2)

        # TODO: REMOVE ME LATER!
        for i in range(bs):
            for j in range(self.num_img):
                single_img = img[i, j, 3:6].cpu().numpy()
                single_img = (single_img * 255).astype(np.uint8)
                single_img = np.transpose(single_img, (1, 2, 0))
                single_hm = hm[i, j].cpu().squeeze().numpy()

                single_hm = cv2.normalize(single_hm, None, 0, 255, cv2.NORM_MINMAX)
                single_hm_colored = cv2.applyColorMap(single_hm.astype(np.uint8), cv2.COLORMAP_JET).astype(np.uint8)
                overlay = cv2.addWeighted(single_img, 0.6, single_hm_colored, 0.4, 0)

                if j not in self.frames:
                    self.frames[j] = []
                    self.img_frames[j] = []
                self.frames[j].append(overlay)
                self.img_frames[j].append(single_img)

        # If the number of steps is reached, create the video
        if all(len(frames) >= self.num_steps for frames in self.frames.values()):
            self.create_videos()
            self.frames = {}  # Clear the frames
            self.img_frames = {}

        return img

    # TODO: REMOVE ME LATER
    def create_videos(self):
        for camera_idx, frames in self.frames.items():
            if frames:
                self.create_video(frames, f"video_camera_{camera_idx}.avi")

        for camera_idx, img_frames in self.img_frames.items():
            if img_frames:
                self.create_video(img_frames, f"video_camera_{camera_idx}_img_only.avi")

    def create_video(self, frames, output_filename):
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_out = cv2.VideoWriter(output_filename, fourcc, 20.0, (width, height))

        for frame in frames:
            video_out.write(frame)

        video_out.release()

    def free_mem(self):
        """
        Could be used for freeing up the memory once a batch of testing is done
        """
        print("Freeing up some memory")
        self.renderer.free_mem()


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    mvt = MVT(**cfg)
    breakpoint()
