#!/usr/bin/env python 
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
# from models.chamfer_distance.chamfer_distance import ChamferDistance
from models.nn_distance.chamfer_loss import ChamferLoss
from models.nn_distance import nn_distance
from models.sdf import SDFFunction


class LipschitzLoss(nn.Module):
    def __init__(self, k, reduction=None):
        super(LipschitzLoss, self).__init__()
        self.relu = nn.ReLU()
        self.k = k
        self.reduction = reduction

    def forward(self, x1, x2, y1, y2):
        l = self.relu(torch.norm(y1-y2, dim=-1) / (torch.norm(x1-x2, dim=-1)+1e-3) - self.k)
        # l = torch.clamp(l, 0.0, 5.0)    # avoid
        if self.reduction is None or self.reduction == "mean":
            return torch.mean(l)
        else:
            return torch.sum(l)


class Loss(nn.Module):
    """
        Loss for LV-track network training
        sdf is defined as the value between prior and model
    """

    def __init__(self):
        super(Loss, self).__init__()
        self.threshold = 0.1
        self.overall_loss_weights = [0.4, 0.4, 0.1, 0.1]
        self.cap_loss_weights = [0.3, 0.3, 0.4]

        # self.chamfer_loss = ChamferDistance()
        self.crossloss = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        self.lipschitz_loss = LipschitzLoss(22)

    def compute_per_level_cap_loss(self, pair_caption, per_caption_embed):
        all_cap_loss = 0
        batch_size, num_pair, caption_len = pair_caption.size()
        pre_emb, cur_emb = pair_caption.chunk(2, 1)
        pair_emb_list = [pre_emb, cur_emb]
        for i in range(batch_size):
            temp_loss = 0
            for j in range(num_pair):
                cap_loss = self.crossloss(pair_emb_list[1].reshape(-1, caption_len).unsqueeze(0),
                                          caption_embed.reshape(-1))
                print("cap_loss")

        return all_cap_loss

    def forward(self, shape_query, logits, pair_caption, caption_embed, sdf_pred, prior, model):
        """
            shape_query: bs x n_pts_cur x 3
            logits: bs x num_hypo
            pair_caption: bs X 2 x caption_len
            caption_embed: bs x 3 x caption_len
            sdf_pred: bs x 1 x n_pts_cur
            prior: bs x 1024 x 3
            model: bs x 4096 x 3
        """
        log_prob = torch.log_softmax(logits, dim=-1)
        pose_loss = -torch.mean(log_prob[:, 0])

        # shape_loss, _, _ = self.chamfer_loss(shape_query, prior)
        reconstructed_shape = torch.cat([shape_query, prior], dim=1)

        dist1, _, dist2, _ = nn_distance(reconstructed_shape, model)
        # dist1, dist2 = self.chamfer_loss(shape_query, prior)
        shape_loss = (torch.mean(dist1)) + (torch.mean(dist2))

        sdf_loss = shape_loss
        # sdf_loss = SDFFunction(sdf_pred, prior, shape_query)

        caption_loss = shape_loss
        # a = self.crossloss(sdf_pred.permute(0,2,1), sdf_pred.reshape(-1))
        #
        # _, num_level, _ = caption_embed.size()
        # v_cap, s_cap, o_cap = caption_embed.chunk(num_level, 1)
        # multi_cap_list = [v_cap, s_cap, o_cap]
        # caption_loss = 0
        #
        # for i in range(num_level):
        #     each_level_loss = self.compute_per_level_cap_loss(pair_caption, multi_cap_list[i])
        #     caption_loss = caption_loss + self.cap_loss_weights[i] * each_level_loss

        loss = self.overall_loss_weights[0] * pose_loss + self.overall_loss_weights[1] * shape_loss + \
               self.overall_loss_weights[2] * sdf_loss + self.overall_loss_weights[3] * caption_loss

        return loss, pose_loss, shape_loss, sdf_loss, caption_loss
