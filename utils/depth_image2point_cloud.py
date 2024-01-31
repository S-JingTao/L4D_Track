# -*- coding: utf-8 -*-
from PIL import Image
# import pandas as pd
import numpy as np
import open3d
import time
import scipy.io as scio


class Depth2PointCloud:
    """
    """

    def __init__(self, rgb_file, depth_file, pc_file, intrinsic_matrix, scalingfactor):
        self.rgb_file = rgb_file
        self.depth_file = depth_file
        self.pc_file = pc_file
        self.camera_intrinsic = intrinsic_matrix
        self.focal_length = np.mean([self.camera_intrinsic[0][0], self.camera_intrinsic[1][1]])  # 为了统一焦距，这里做了平均处理
        self.u0 = self.camera_intrinsic[0][2]
        self.v0 = self.camera_intrinsic[1][2]
        self.scalingfactor = scalingfactor
        self.rgb = Image.open(rgb_file)
        self.depth = Image.open(depth_file).convert('I')
        self.width = self.rgb.size[0]
        self.height = self.rgb.size[1]

    def calculate(self):
        t1 = time.time()
        depth = np.asarray(self.depth).T
        self.Z = depth / self.scalingfactor
        X = np.zeros((self.width, self.height))
        Y = np.zeros((self.width, self.height))
        for i in range(self.width):
            X[i, :] = np.full(X.shape[1], i)

        self.X = ((X - self.u0) * self.Z) / self.focal_length
        for i in range(self.height):
            Y[:, i] = np.full(Y.shape[0], i)
        self.Y = ((Y - self.v0) * self.Z) / self.focal_length

        df = np.zeros((6, self.width * self.height))
        df[0] = self.X.T.reshape(-1)
        df[1] = -self.Y.T.reshape(-1)
        df[2] = -self.Z.T.reshape(-1)
        img = np.array(self.rgb)
        # df[3] = 100*np.ones((1, self.width * self.height))
        # df[4] = 100*np.ones((1, self.width * self.height))
        # df[5] = 100*np.ones((1, self.width * self.height))
        df[3] = img[:, :, 0:1].reshape(-1)
        df[4] = img[:, :, 1:2].reshape(-1)
        df[5] = img[:, :, 2:3].reshape(-1)
        self.df = df
        t2 = time.time()
        print('calcualte 3d point cloud Done.', t2 - t1)

    def write_ply(self):
        t1 = time.time()
        float_formatter = lambda x: "%.4f" % x
        points = []
        for i in self.df.T:
            points.append("{} {} {} {} {} {} 0\n".format
                          (float_formatter(i[0]), float_formatter(i[1]), float_formatter(i[2]),
                           int(i[3]), int(i[4]), int(i[5])))

        file = open(self.pc_file, "w")
        file.write('''ply
        format ascii 1.0
        element vertex %d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        property uchar alpha
        end_header
        %s
        ''' % (len(points), "".join(points)))
        file.close()

        t2 = time.time()
        print("Write into .ply file Done.", t2 - t1)

    def save_npy_file(self):
        df = self.df
        np.save('pc.npy', df)

    def show_point_cloud(self):
        pcd = open3d.io.read_point_cloud(self.pc_file)
        open3d.visualization.draw_geometries([pcd])
