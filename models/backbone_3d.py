#!/usr/bin/env python 
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2.pointnet2_utils import PointNetSetAbstraction


class Pointnet2_based(nn.Module):
    def __init__(self, num_point=None, normal_channel=False):
        super(Pointnet2_based, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.num_point = num_point
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=self.num_point, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128],
                                          group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=self.num_point, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=self.num_point, radius=0.4, nsample=64, in_channel=256 + 3,
                                          mlp=[256, 256, 512], group_all=False)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        # self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        xyz = xyz.permute(0, 2, 1)
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points
        # x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        # x = self.fc3(x)
        # x = F.log_softmax(x, -1)
        return x


if __name__ == '__main__':
    net = Pointnet2_based(num_point=2048)
    net = net.cuda()
    pts = torch.randn(2, 2048, 3).cuda()

    pre = net(pts)
    print(pre.shape)
