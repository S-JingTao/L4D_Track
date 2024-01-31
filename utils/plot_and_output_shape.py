#!/usr/bin/env python 
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from lib.utils import align_rotation, get_3d_bbox,transform_coordinates_3d,calculate_2d_projections,draw_bboxes

def standardize_bbox(pcl, points_per_object):
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    np.random.shuffle(pt_indices)
    pcl = pcl[pt_indices]  # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = (mins + maxs) / 2.
    scale = np.amax(maxs - mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center) / scale).astype(np.float32)  # [-0.5, 0.5]
    return result


xml_head = \
    """
    <scene version="0.5.0">
        <integrator type="path">
            <integer name="maxDepth" value="-1"/>
        </integrator>
        <sensor type="perspective">
            <float name="farClip" value="100"/>
            <float name="nearClip" value="0.1"/>
            <transform name="toWorld">
                <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
            </transform>
            <float name="fov" value="25"/>
    
            <sampler type="ldsampler">
                <integer name="sampleCount" value="256"/>
            </sampler>
            <film type="ldrfilm">
                <integer name="width" value="1600"/>
                <integer name="height" value="1200"/>
                <rfilter type="gaussian"/>
                <boolean name="banner" value="false"/>
            </film>
        </sensor>
    
        <bsdf type="roughplastic" id="surfaceMaterial">
            <string name="distribution" value="ggx"/>
            <float name="alpha" value="0.05"/>
            <float name="intIOR" value="1.46"/>
            <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
        </bsdf>
    
    """

xml_ball_segment = \
    """
        <shape type="sphere">
            <float name="radius" value="0.02"/>
            <transform name="toWorld">
                <translate x="{}" y="{}" z="{}"/>
                <scale value="0.7"/>
            </transform>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="{},{},{}"/>
            </bsdf>
        </shape>
    """

xml_tail = \
    """
        <shape type="rectangle">
            <ref name="bsdf" id="surfaceMaterial"/>
            <transform name="toWorld">
                <scale x="10" y="10" z="10"/>
                <translate x="0" y="0" z="-0.5"/>
            </transform>
        </shape>
    
        <shape type="rectangle">
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
            </transform>
            <emitter type="area">
                <rgb name="radiance" value="6,6,6"/>
            </emitter>
        </shape>
    </scene>
    """


def colormap(x, y, z):
    vec = np.array([x, y, z])
    vec = np.clip(vec, 0.001, 1.0)
    norm = np.sqrt(np.sum(vec ** 2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]


def mitsuba(pcl, path, clr=None):
    xml_segments = [xml_head]

    # pcl = standardize_bbox(pcl, 2048)
    pcl = pcl[:, [2, 0, 1]]
    pcl[:, 0] *= -1
    h = np.min(pcl[:, 2])

    for i in range(pcl.shape[0]):
        if clr == None:
            color = colormap(pcl[i, 0] + 0.5, pcl[i, 1] + 0.5, pcl[i, 2] + 0.5)
        else:
            color = clr
        if h < -0.25:
            xml_segments.append(xml_ball_segment.format(pcl[i, 0], pcl[i, 1], pcl[i, 2] - h - 0.6875, *color))
        else:
            xml_segments.append(xml_ball_segment.format(pcl[i, 0], pcl[i, 1], pcl[i, 2], *color))
    xml_segments.append(xml_tail)

    xml_content = str.join('', xml_segments)

    with open(path, 'w') as f:
        f.write(xml_content)


def add_noise(sRT):
    def noise_Gaussian(points, std):
        noise = np.random.normal(0, std, points.shape)
        out = points + noise
        return out

    sRT[:3, :3] = noise_Gaussian(sRT[:3, :3],0.01)
    sRT[:3, 3] = noise_Gaussian(sRT[:3, 3],0.01)

    return sRT


def draw_detections(img, out_dir, data_name, img_id, intrinsics, pred_sRT, pred_size, pred_class_ids,
                    gt_sRT, gt_size, gt_class_ids):
    out_path = os.path.join(out_dir, '{}_{}_pred.png'.format(data_name, img_id))
    for i in range(gt_sRT.shape[0]):
        if gt_class_ids[i] in [1, 2, 4]:
            sRT = align_rotation(gt_sRT[i, :, :])
        else:
            sRT = gt_sRT[i, :, :]
        sRT = add_noise(sRT)
        bbox_3d = get_3d_bbox(gt_size[i, :], 0)
        transformed_bbox_3d = transform_coordinates_3d(bbox_3d, sRT)
        projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
        # if gt_class_ids[i] == 5:
        # img = draw_bboxes(img, projected_bbox, (0, 234, 0))
        img = draw_bboxes(img, projected_bbox, (0, 238, 238))

    # for i in range(pred_sRT.shape[0]):
    #     if pred_class_ids[i] in [1, 2, 4]:
    #         sRT = align_rotation(pred_sRT[i, :, :])
    #     else:
    #         sRT = pred_sRT[i, :, :]
    #     bbox_3d = get_3d_bbox(pred_size[i, :], 0)
    #     transformed_bbox_3d = transform_coordinates_3d(bbox_3d, sRT)
    #     projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
    #     img = draw_bboxes(img, projected_bbox, (0, 238, 238))

    cv2.imwrite(out_path, img)


def draw_detections_one(img, out_dir, data_name, img_id, intrinsics, pred_sRT, pred_size):
    out_path = os.path.join(out_dir, '{}_{}_pred.png'.format(data_name, img_id))

    bbox_3d = get_3d_bbox(pred_size, 0)
    transformed_bbox_3d = transform_coordinates_3d(bbox_3d, pred_sRT)
    projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
    # if gt_class_ids[i] == 5:
    # img = draw_bboxes(img, projected_bbox, (0, 234, 0))
    img = draw_bboxes(img, projected_bbox, (0, 238, 238))

    # for i in range(pred_sRT.shape[0]):
    #     if pred_class_ids[i] in [1, 2, 4]:
    #         sRT = align_rotation(pred_sRT[i, :, :])
    #     else:
    #         sRT = pred_sRT[i, :, :]
    #     bbox_3d = get_3d_bbox(pred_size[i, :], 0)
    #     transformed_bbox_3d = transform_coordinates_3d(bbox_3d, sRT)
    #     projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
    #     img = draw_bboxes(img, projected_bbox, (0, 238, 238))

    cv2.imwrite(out_path, img)

img_dir = "/home/amax/document/sjt_project/datasets/YCBInEOAT/sugar_box_yalehand0/rgb/1582918624239672590.png"
img = cv2.imread(img_dir)
out_dir = "results/ycb"
data_name = 'YCBInEOAT'
img_id = 1
intrinsics = [
    [3.195820007324218750e+02, 0, 3.202149847676955687e+02],
    [0, 4.171186828613281250e+02, 2.443486680871046701e+02],
    [0, 0, 1]
]
gt = np.loadtxt("/home/amax/document/sjt_project/datasets/YCBInEOAT/sugar_box_yalehand0/annotated_poses/0000000.txt")

pred_size = [171.19,148.72,367.10, 337.44]
