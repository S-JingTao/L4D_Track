#!/usr/bin/env python 
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import random
import torch
import glob
import csv

import _pickle as cPickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.nocs_eval_utils import compute_mAP, plot_mAP


def load_pred_results(result_dir):
    result_pkl_list = glob.glob(os.path.join(result_dir, 'results_*.pkl'))
    result_pkl_list = sorted(result_pkl_list)
    assert len(result_pkl_list)  # 2754 for real
    pred_results = []
    for pkl_path in result_pkl_list:
        with open(pkl_path, 'rb') as f:
            result = cPickle.load(f)
            if 'gt_handle_visibility' not in result:
                result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])
            else:
                assert len(result['gt_handle_visibility']) == len(result['gt_class_ids']), "{} {}".format(
                    result['gt_handle_visibility'], result['gt_class_ids'])
        if type(result) is list:
            pred_results += result
        elif type(result) is dict:
            pred_results.append(result)
        else:
            assert False
    return pred_results


def evaluate(result_dir):
    degree_thres_list = list(range(0, 61, 1))
    shift_thres_list = [i / 2 for i in range(21)]
    iou_thres_list = [i / 100 for i in range(101)]

    pred_results = load_pred_results(result_dir)

    iou_aps, pose_aps, iou_acc, pose_acc = compute_mAP(pred_results, result_dir, degree_thres_list, shift_thres_list,
                                                       iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True)

    # metric list
    iou_25_idx = iou_thres_list.index(0.25)  # 25
    iou_50_idx = iou_thres_list.index(0.5)  # 50
    iou_75_idx = iou_thres_list.index(0.75)  # 75
    degree_05_idx = degree_thres_list.index(5)  # 5
    degree_10_idx = degree_thres_list.index(10)  # 10
    shift_02_idx = shift_thres_list.index(2)  # 2
    shift_5_idx = shift_thres_list.index(5)  # 20

    messages = []
    messages.append('mAP:')
    messages.append('bottle:')
    messages.append('3D IoU at 25: {:.4f}'.format(iou_aps[1, iou_25_idx]))
    messages.append('3D IoU at 50: {:.4f}'.format(iou_aps[1, iou_50_idx]))
    messages.append('3D IoU at 75: {:.4f}'.format(iou_aps[1, iou_75_idx]))
    messages.append('5 degree, 2cm: {:.4f}'.format(pose_aps[1, degree_05_idx, shift_02_idx]))
    messages.append('5 degree, 5cm: {:.4f}'.format(pose_aps[1, degree_05_idx, shift_5_idx]))
    messages.append('10 degree, 2cm: {:.4f}'.format(pose_aps[1, degree_10_idx, shift_02_idx]))
    messages.append('10 degree, 5cm: {:.4f}'.format(pose_aps[1, degree_10_idx, shift_5_idx]))

    messages.append('bowl:')
    messages.append('3D IoU at 25: {:.4f}'.format(iou_aps[2, iou_25_idx]))
    messages.append('3D IoU at 50: {:.4f}'.format(iou_aps[2, iou_50_idx]))
    messages.append('3D IoU at 75: {:.4f}'.format(iou_aps[2, iou_75_idx]))
    messages.append('5 degree, 2cm: {:.4f}'.format(pose_aps[2, degree_05_idx, shift_02_idx]))
    messages.append('5 degree, 5cm: {:.4f}'.format(pose_aps[2, degree_05_idx, shift_5_idx]))
    messages.append('10 degree, 2cm: {:.4f}'.format(pose_aps[2, degree_10_idx, shift_02_idx]))
    messages.append('10 degree, 5cm: {:.4f}'.format(pose_aps[2, degree_10_idx, shift_5_idx]))

    messages.append('camera:')
    messages.append('3D IoU at 25: {:.4f}'.format(iou_aps[3, iou_25_idx]))
    messages.append('3D IoU at 50: {:.4f}'.format(iou_aps[3, iou_50_idx]))
    messages.append('3D IoU at 75: {:.4f}'.format(iou_aps[3, iou_75_idx]))
    messages.append('5 degree, 2cm: {:.4f}'.format(pose_aps[3, degree_05_idx, shift_02_idx]))
    messages.append('5 degree, 5cm: {:.4f}'.format(pose_aps[3, degree_05_idx, shift_5_idx]))
    messages.append('10 degree, 2cm: {:.4f}'.format(pose_aps[3, degree_10_idx, shift_02_idx]))
    messages.append('10 degree, 5cm: {:.4f}'.format(pose_aps[3, degree_10_idx, shift_5_idx]))

    messages.append('can:')
    messages.append('3D IoU at 25: {:.4f}'.format(iou_aps[4, iou_25_idx]))
    messages.append('3D IoU at 50: {:.4f}'.format(iou_aps[4, iou_50_idx]))
    messages.append('3D IoU at 75: {:.4f}'.format(iou_aps[4, iou_75_idx]))
    messages.append('5 degree, 2cm: {:.4f}'.format(pose_aps[4, degree_05_idx, shift_02_idx]))
    messages.append('5 degree, 5cm: {:.4f}'.format(pose_aps[4, degree_05_idx, shift_5_idx]))
    messages.append('10 degree, 2cm: {:.4f}'.format(pose_aps[4, degree_10_idx, shift_02_idx]))
    messages.append('10 degree, 5cm: {:.4f}'.format(pose_aps[4, degree_10_idx, shift_5_idx]))

    messages.append('laptop:')
    messages.append('3D IoU at 25: {:.4f}'.format(iou_aps[5, iou_25_idx]))
    messages.append('3D IoU at 50: {:.4f}'.format(iou_aps[5, iou_50_idx]))
    messages.append('3D IoU at 75: {:.4f}'.format(iou_aps[5, iou_75_idx]))
    messages.append('5 degree, 2cm: {:.4f}'.format(pose_aps[5, degree_05_idx, shift_02_idx]))
    messages.append('5 degree, 5cm: {:.4f}'.format(pose_aps[5, degree_05_idx, shift_5_idx]))
    messages.append('10 degree, 2cm: {:.4f}'.format(pose_aps[5, degree_10_idx, shift_02_idx]))
    messages.append('10 degree, 5cm: {:.4f}'.format(pose_aps[5, degree_10_idx, shift_5_idx]))

    messages.append('mug:')
    messages.append('3D IoU at 25: {:.4f}'.format(iou_aps[6, iou_25_idx]))
    messages.append('3D IoU at 50: {:.4f}'.format(iou_aps[6, iou_50_idx]))
    messages.append('3D IoU at 75: {:.4f}'.format(iou_aps[6, iou_75_idx]))
    messages.append('5 degree, 2cm: {:.4f}'.format(pose_aps[6, degree_05_idx, shift_02_idx]))
    messages.append('5 degree, 5cm: {:.4f}'.format(pose_aps[6, degree_05_idx, shift_5_idx]))
    messages.append('10 degree, 2cm: {:.4f}'.format(pose_aps[6, degree_10_idx, shift_02_idx]))
    messages.append('10 degree, 5cm: {:.4f}'.format(pose_aps[6, degree_10_idx, shift_5_idx]))

    messages.append('mean:')
    messages.append('3D IoU at 25: {:.4f}'.format(iou_aps[-1, iou_25_idx]))
    messages.append('3D IoU at 50: {:.4f}'.format(iou_aps[-1, iou_50_idx]))
    messages.append('3D IoU at 75: {:.4f}'.format(iou_aps[-1, iou_75_idx]))
    messages.append('5 degree, 2cm: {:.4f}'.format(pose_aps[-1, degree_05_idx, shift_02_idx]))
    messages.append('5 degree, 5cm: {:.4f}'.format(pose_aps[-1, degree_05_idx, shift_5_idx]))
    messages.append('10 degree, 2cm: {:.4f}'.format(pose_aps[-1, degree_10_idx, shift_02_idx]))
    messages.append('10 degree, 5cm: {:.4f}'.format(pose_aps[-1, degree_10_idx, shift_5_idx]))

    with open('{0}/eval_logs.txt'.format(result_dir), 'a') as file:
        for msg in messages:
            file.write(msg + '\n')

    with open('{0}/evaluation-3D-IoU.csv'.format(result_dir), 'a') as file:
        writer = csv.writer(file)
        writer.writerow(np.array(iou_thres_list))
        for i in range(1, iou_aps.shape[0]):
            writer.writerow(iou_aps[i, :])

    with open('{0}/evaluation-rotation.csv'.format(result_dir), 'a') as file:
        writer = csv.writer(file)
        writer.writerow(np.array(degree_thres_list))
        for i in range(1, pose_aps.shape[0]):
            writer.writerow(pose_aps[i, :len(degree_thres_list), -1])

    with open('{0}/evaluation-translation.csv'.format(result_dir), 'a') as file:
        writer = csv.writer(file)
        writer.writerow(np.array(shift_thres_list))
        for i in range(1, pose_aps.shape[0]):
            writer.writerow(pose_aps[i, -1, :len(shift_thres_list)])

    plot_mAP(iou_aps, pose_aps, result_dir, iou_thres_list, degree_thres_list, shift_thres_list)


if __name__ == '__main__':

    result_dir = ""

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    print('Evaluating in test data...')
    evaluate(result_dir)
    print("Evaluation end!")
