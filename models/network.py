#!/usr/bin/env python 
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
import argparse
import time
import random
import numpy.ma as ma
import copy
import math
import scipy.misc
import scipy.io as scio
import cv2
import matplotlib.pyplot as plt
import _pickle as cPickle

from models.backbone_2d import PSPNet
from models.backbone_3d import Pointnet2_based
from utils.transformations import quaternion_matrix


def generate_random_poses(n=1, device="cpu"):
    quats = torch.randn(n, 4, device=device)
    quats = quats / quats.norm(dim=1, keepdim=True)

    translation = torch.randn(n, 3, device=device)
    translation = translation / translation.norm(dim=1, keepdim=True)

    sRT = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    sRT = sRT.repeat(n, 1, 1)

    for i in range(0, n - 1):
        sRT[i, :, :] = torch.from_numpy(quaternion_matrix(quats[i, :]))
        sRT[i, :3, 3] = translation[i, :]

    return sRT


def generate_superfibonacci(n=1, device="cpu"):
    """
    Samples n rotations equivolumetrically using a Super-Fibonacci Spiral.

    Reference: Marc Alexa, Super-Fibonacci Spirals. CVPR 22.

    Args:
        n (int): Number of rotations to sample.
        device (str): CUDA Device. Defaults to CPU.

    Returns:
        (tensor): Rotations (n, 3, 3).
    """
    phi = np.sqrt(2.0)
    psi = 1.533751168755204288118041
    ind = torch.arange(n, device=device)
    s = ind + 0.5
    r = torch.sqrt(s / n)
    R = torch.sqrt(1.0 - s / n)
    alpha = 2 * np.pi * s / phi
    beta = 2.0 * np.pi * s / psi
    Q = torch.stack(
        [
            r * torch.sin(alpha),
            r * torch.cos(alpha),
            R * torch.sin(beta),
            R * torch.cos(beta),
        ],
        1,
    )
    return quaternion_matrix(Q).float()


class LVTrackNet(nn.Module):
    def __init__(self, caption_len=768, sample_mode="random", num_hypo=50000, n_pts=2048):
        super(LVTrackNet, self).__init__()

        self.caption_len = caption_len
        self.sample_mode = sample_mode
        self.num_hypo = num_hypo
        self.n_pts = n_pts

        self.equi_grid = {}
        self.queries = None

        self.backbone3d_encode = Pointnet2_based(self.n_pts)
        # #
        self.backbone2d_encode = PSPNet(bins=(1, 2, 3, 6), backend='resnet18')
        # self.backbone3d_encode = nn.Sequential(
        #     nn.Conv1d(3, 64, 1),
        #     nn.ReLU(), )
        # self.backbone2d_encode = nn.Sequential(
        #     nn.Conv1d(3, 64, 1),
        #     nn.ReLU(), )

        self.backbone2d_feat = nn.Sequential(
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 512, 1),
            nn.ReLU(),
        )

        self.backbone3d_feat = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 512, 1),
            nn.ReLU(),
        )

        self.geometry_global = nn.Sequential(
            nn.Conv1d(1024, 1024, 1),
            nn.ReLU(),
            nn.Conv1d(1024, 2048, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.pairwise_geometry = nn.Sequential(
            nn.Conv1d(1024, 1024, 1),
            nn.ReLU(),
            nn.Conv1d(1024, 2048, 1),
            nn.ReLU(),
            nn.Conv1d(2048, 4096, 1),
            nn.ReLU(),
        )

        self.neural_pose_align = nn.Sequential(
            nn.Conv1d(4096, 4096, 1),
            nn.ReLU(),
            nn.Conv1d(4096, 2048, 1),
            nn.ReLU(),
            nn.Conv1d(2048, 1024, 1),
            nn.ReLU(),
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 64, 1),
            nn.ReLU(),
        )
        self.sdf_head = nn.Sequential(
            nn.Conv1d(64, 3, 1),
            nn.ReLU(),
            nn.BatchNorm1d(3),
            nn.ReLU(),
            # nn.AdaptiveAvgPoold(1)
        )

        self.shape_head = nn.Sequential(
            nn.Conv1d(64, 3, 1),
            nn.ReLU(),
        )

        self.caption_head = nn.Sequential(
            nn.Conv1d(1024, self.caption_len, 1),
            nn.ReLU(),
        )

        self.hidden_size = 256
        self.num_layer = 4
        self.embed_feature = nn.Linear(2048 * 2, self.hidden_size)
        self.embed_query = nn.Linear(16, self.hidden_size)

        layers = []
        for _ in range(self.num_layer - 2):
            layers.append(nn.LeakyReLU())
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(self.hidden_size, 1))
        self.layers = nn.Sequential(*layers)

    def sample_pose_hypo(self, sRT_gt):
        if self.queries is None:
            if self.sample_mode == "equivolumetric" and self.num_hypo not in self.equi_grid:
                self.equi_grid[self.num_hypo] = generate_superfibonacci(self.num_hypo, device="cpu")
                self.queries = self.equi_grid[self.num_hypo]
                self.queries = self.queries.cuda()

            elif self.sample_mode == "random":
                self.queries = generate_random_poses(self.num_hypo, device="cpu")
                self.queries = self.queries.cuda()
            else:
                raise Exception(f"Unknown sampling mode {self.sample_mode}.")

            if sRT_gt is not None:
                delta_rot = torch.inverse(self.queries[0]) @ sRT_gt
                # First entry will always be the gt rotation
                self.queries = torch.einsum("aij,bjk->baik", self.queries, delta_rot)

            else:
                if len(self.queries.shape) == 3:
                    self.queries = self.queries.unsqueeze(0)
                # self.num_hypo = self.queries[1]
        else:
            self.num_hypo = self.queries.shape[1]

    def forward(self, pre_points, pre_seg, choose_pre, cur_points, cur_seg, choose_cur, sRT_gt=None):
        """
            pre_points: bs x n_pts x 3
            pre_seg: bs x 3 x H x W
            choose_pre: bs x n_pts
            cur_points: bs x n_pts x 3
            cur_seg: bs x 3 x H x W
            choose_cur: bs x n_pts
            sRT_gt: bs x 4 x 4
        """
        bs, n_pts_pre = pre_points.size()[:2]
        _, n_pts_cur = cur_points.size()[:2]

        pre_img = self.backbone2d_encode(pre_seg)
        # pre_img = torch.randn(1, 32, 240, 240)

        di_pre = pre_img.size()[1]
        emb_pre = pre_img.view(bs, di_pre, -1)
        choose_pre = choose_pre.unsqueeze(1).repeat(1, di_pre, 1)
        emb_pre = torch.gather(emb_pre, 2, choose_pre).contiguous()
        emb_pre = self.backbone2d_feat(emb_pre)

        # pre_points = pre_points.permute(0, 2, 1)  # delect
        pre_points = self.backbone3d_encode(pre_points)
        # pre_points = self.backbone3d_feat(pre_points)

        cur_img = self.backbone2d_encode(cur_seg)
        # cur_img = torch.randn(1, 32, 240, 240)

        di_cur = cur_img.size()[1]
        emb_cur = cur_img.view(bs, di_cur, -1)
        choose_cur = choose_cur.unsqueeze(1).repeat(1, di_pre, 1)
        emb_cur = torch.gather(emb_cur, 2, choose_cur).contiguous()
        emb_cur = self.backbone2d_feat(emb_cur)

        # cur_points = cur_points.permute(0, 2, 1)  # delect
        cur_points = self.backbone3d_encode(cur_points)
        # cur_points = self.backbone3d_feat(cur_points)

        pre_base = torch.cat((pre_points, emb_pre), dim=1)  # bs x 1024 x n_pts_pre
        cur_base = torch.cat((cur_points, emb_cur), dim=1)  # bs x 1024 x n_pts_cur

        # energy-based pose hypothesis
        pre_global = self.geometry_global(pre_base).contiguous()
        cur_global = self.geometry_global(cur_base).contiguous()

        pairwise_global = torch.cat((pre_global, cur_global), dim=1)
        pairwise_global = pairwise_global.reshape(bs, -1)  # bs x4096

        self.sample_pose_hypo(sRT_gt)
        pose_hypo = self.queries.clone()
        queries_pe = self.queries.reshape(-1, self.num_hypo, 16)

        e_f = self.embed_feature(pairwise_global).unsqueeze(1)  # bs x 1 x 256
        e_q = self.embed_query(queries_pe)  # (bs, num_hypo, 256)
        out = self.layers(e_f + e_q)  # (bs, num_hypo, 1)
        logits = out.reshape(bs, self.num_hypo)

        # neural_aligned Field
        feat_hypo = pairwise_global.unsqueeze(2)
        feat_hypo = feat_hypo.repeat(1, 1, n_pts_cur)

        feat_cur = self.pairwise_geometry(cur_base)
        feat_input = torch.add(feat_cur, feat_hypo, alpha=0.7)
        feat_output = self.neural_pose_align(feat_input)

        sdf = self.sdf_head(feat_output)
        # max_pool = torch.nn.MaxPool1d(sdf.size()[1])
        sdf_pred = torch.max(sdf, 1, keepdim=True)[0]  # bs x 1 x n_pts_cur
        shape_query = self.shape_head(feat_output)  # bs x 3 x n_pts_cur
        shape_query = shape_query.permute(0, 2, 1).contiguous()  # bs x n_pts_cur x 3

        pair_caption = torch.cat([torch.nn.MaxPool1d(pre_base.size()[2])(pre_base),
                                  torch.nn.MaxPool1d(cur_base.size()[2])(cur_base)], dim=2)
        pair_caption = (pair_caption)
        pair_caption = pair_caption.permute(0, 2, 1).contiguous()  # bs X 2 x caption_len

        return pose_hypo, shape_query, logits, sdf_pred, pair_caption
