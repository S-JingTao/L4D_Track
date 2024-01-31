import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf

from models.network import LVTrackNet
from models.loss_function import Loss

from dataset.nocs_real import PoseDataset

from utils.utils import setup_logger, compute_sRT_errors
from utils.align import estimateSimilarityTransform

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Real', help='CAMERA or CAMERA+Real')
parser.add_argument('--data_dir', type=str, default='/home/amax/document/sjt_project/datasets/MY_NOCS/',
                    help='data directory')
parser.add_argument('--n_pts', type=int, default=2048, help='number of targeted instance points')
parser.add_argument('--n_cat', type=int, default=6, help='number of object categories')
parser.add_argument('--img_size', type=int, default=250, help='cropped image size')
parser.add_argument('--caption_len', type=int, default=768, help='length of language caption embeddings')
parser.add_argument('--sample_mode', type=str, default="random",
                    help='sample mode of pose hypothesis (or equivolumetric)')
parser.add_argument('--hypo_num', type=int, default=50000, help='number of pose potential spaces')
parser.add_argument('--batch_size', type=int, default=10, help='batch size')
parser.add_argument('--num_workers', type=int, default=10, help='number of data loading workers')
parser.add_argument('--gpu', type=str, default='1,2,3,0', help='GPU to use')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--start_epoch', type=int, default=2, help='which epoch to start')
parser.add_argument('--max_epoch', type=int, default=30, help='max number of epochs to train')
parser.add_argument('--resume_model', type=str, default='/home/sjt/LV-Track/results/Real/model_02.pth', help='resume from saved model')
parser.add_argument('--result_dir', type=str, default='results/Real', help='directory to save train results')
opt = parser.parse_args()

# LOSS treashold
opt.decay_epoch = [0, 10, 20]
opt.decay_rate = [1.0, 0.6, 0.3]


def train_net():
    # set result directory
    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)
    tb_writer = tf.summary.FileWriter(opt.result_dir)
    logger = setup_logger('train_log', os.path.join(opt.result_dir, 'log.txt'))
    for key, value in vars(opt).items():
        logger.info(key + ': ' + str(value))
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    # model & loss
    estimator = LVTrackNet(opt.caption_len, opt.sample_mode, opt.hypo_num, opt.n_pts)
    estimator.cuda()
    estimator = nn.DataParallel(estimator)
    if opt.resume_model != '':
        estimator.load_state_dict(torch.load(opt.resume_model), False)

    criterion = Loss()

    # dataset
    train_dataset = PoseDataset(opt.dataset, 'train', opt.data_dir, opt.n_pts, opt.img_size, opt.n_cat, opt.caption_len)
    val_dataset = PoseDataset(opt.dataset, 'test', opt.data_dir, opt.n_pts, opt.img_size, opt.n_cat, opt.caption_len)

    # start training
    st_time = time.time()
    train_steps = 300
    global_step = train_steps * (opt.start_epoch - 1)
    n_decays = len(opt.decay_epoch)
    assert len(opt.decay_rate) == n_decays
    for i in range(n_decays):
        if opt.start_epoch > opt.decay_epoch[i]:
            decay_count = i
    train_size = train_steps * opt.batch_size
    indices = []
    page_start = -train_size
    for epoch in range(opt.start_epoch, opt.max_epoch + 1):
        # train one epoch
        logger.info('Time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + \
                                      ', ' + 'Current Epoch %02d:' % epoch + ', ' + 'Training started'))
        # create optimizer and adjust learning rate if needed
        if decay_count < len(opt.decay_rate):
            if epoch > opt.decay_epoch[decay_count]:
                current_lr = opt.lr * opt.decay_rate[decay_count]
                optimizer = torch.optim.Adam(estimator.parameters(), lr=current_lr)
                decay_count += 1
        # sample train subset
        page_start += train_size
        len_last = len(indices) - page_start
        if len_last < train_size:
            indices = indices[page_start:]
            if opt.dataset == 'CAMERA+Real':
                # CAMERA : Real = 3 : 1
                camera_len = train_dataset.subset_len[0]
                real_len = train_dataset.subset_len[1]
                real_indices = list(range(camera_len, camera_len + real_len))
                camera_indices = list(range(camera_len))
                n_repeat = (train_size - len_last) // (4 * real_len) + 1
                data_list = random.sample(camera_indices, 3 * n_repeat * real_len) + real_indices * n_repeat
                random.shuffle(data_list)
                indices += data_list
            else:
                data_list = list(range(train_dataset.length))
                for i in range((train_size - len_last) // train_dataset.length + 1):
                    random.shuffle(data_list)
                    indices += data_list
            page_start = 0
        train_idx = indices[page_start:(page_start + train_size)]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, sampler=train_sampler,
                                                       num_workers=opt.num_workers, pin_memory=True)
        estimator.train()

        for i, data in enumerate(train_dataloader, 1):
            pre_data, cur_data, caption_embed, model, prior, cat_id = data

            pre_points, pre_rgb, pre_choose = pre_data[0].cuda(), pre_data[1].cuda(), pre_data[2].cuda()
            cur_points, cur_rgb, cur_choose = cur_data[0].cuda(), cur_data[1].cuda(), cur_data[2].cuda()

            # compute relative pose change ground-truth
            cur_sRT = cur_data[3]
            pre_sRT = pre_data[3]
            sRT_gt = cur_sRT @ torch.inverse(pre_sRT)
            sRT_gt = sRT_gt.cuda()

            caption_embed = caption_embed.cuda()
            model = model.cuda()
            prior = prior.cuda()
            cat_id = cat_id.cuda()

            pose_hypo, shape_query, logits, sdf_pred, pair_caption = estimator(pre_points, pre_rgb, pre_choose,
                                                                               cur_points, cur_rgb, cur_choose, sRT_gt)

            loss, pose_loss, shape_loss, sdf_loss, caption_loss = criterion(shape_query, logits, pair_caption,
                                                                            caption_embed, sdf_pred, prior, model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            # write results to tensorboard
            summary = tf.Summary(value=[tf.Summary.Value(tag='learning_rate', simple_value=current_lr),
                                        tf.Summary.Value(tag='train_loss', simple_value=loss),
                                        tf.Summary.Value(tag='pose_loss', simple_value=pose_loss),
                                        tf.Summary.Value(tag='shape_loss', simple_value=shape_loss),
                                        tf.Summary.Value(tag='sdf_loss', simple_value=sdf_loss),
                                        tf.Summary.Value(tag='caption_loss', simple_value=caption_loss)])
            tb_writer.add_summary(summary, global_step)
            # print("end train loop:", i)
            if i % 10 == 0:
                logger.info(
                    'Batch {0} Loss:{1:f}, pose_loss:{2:f}, shape_loss:{3:f}, sdf_loss:{4:f}, caption_loss:{5:f}'.format(
                        i, loss.item(), pose_loss.item(), shape_loss.item(), sdf_loss.item(), caption_loss.item()))

        logger.info('>>>>>>>>----------Epoch {:02d} train finish---------<<<<<<<<'.format(epoch))

        # evaluate one epoch
        logger.info('Time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) +
                                      ', ' + 'Epoch %02d' % epoch + ', ' + 'Testing started'))

        torch.save(estimator.state_dict(), '{0}/model_{1:02d}.pth'.format(opt.result_dir, epoch))
        val_loss = 0.0
        total_count = np.zeros((opt.n_cat,), dtype=int)
        strict_success = np.zeros((opt.n_cat,), dtype=int)  # 5 degree and 5 cm
        easy_success = np.zeros((opt.n_cat,), dtype=int)  # 10 degree and 5 cm
        iou_success = np.zeros((opt.n_cat,), dtype=int)  # relative scale error < 0.1
        # sample validation subset
        val_size = 900
        val_idx = random.sample(list(range(val_dataset.length)), val_size)
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idx)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, sampler=val_sampler,
                                                     num_workers=opt.num_workers, pin_memory=True)
        estimator = estimator.module
        estimator.eval()

        for i, data in enumerate(val_dataloader, 1):
            pre_data, cur_data, caption_embed, model, prior, cat_id = data

            pre_points, pre_rgb, pre_choose = pre_data[0].cuda(), pre_data[1].cuda(), pre_data[2].cuda()
            cur_points, cur_rgb, cur_choose = cur_data[0].cuda(), cur_data[1].cuda(), cur_data[2].cuda()

            # compute relative pose change ground-truth
            cur_sRT = cur_data[3]
            pre_sRT = pre_data[3]
            sRT_gt = cur_sRT @ torch.inverse(pre_sRT)
            sRT_gt = sRT_gt.cuda()

            caption_embed = caption_embed.cuda()
            model = model.cuda()
            prior = prior.cuda()
            cat_id = cat_id.cuda()

            pose_hypo, shape_query, logits, sdf_pred, pair_caption = estimator(pre_points, pre_rgb, pre_choose,
                                                                               cur_points, cur_rgb, cur_choose, sRT_gt=None)

            loss, pose_loss, shape_loss, sdf_loss, caption_loss = criterion(shape_query, logits, pair_caption,
                                                                            caption_embed, sdf_pred, prior, model)
            max_index = torch.argmax(logits, dim=1)
            pred_relative_sRT = pose_hypo[:, max_index, :]
            pred_sRT = pred_relative_sRT @ pre_sRT.cuda()
            pred_sRT = pred_sRT.detach().cpu().numpy()[0]
            pred_sRT = pred_sRT.reshape(4, 4)
            cat_id = cat_id.item()
            # evaluate pose
            if pred_sRT is not None:
                cur_sRT = cur_sRT.detach().cpu().numpy()[0]
                R_error, T_error, IoU = compute_sRT_errors(pred_sRT, cur_sRT)
                if R_error < 5 and T_error < 0.05:
                    strict_success[cat_id] += 1
                if R_error < 10 and T_error < 0.05:
                    easy_success[cat_id] += 1
                if IoU < 0.1:
                    iou_success[cat_id] += 1
            total_count[cat_id] += 1
            val_loss += loss.item()
            if i % 100 == 0:
                logger.info('Batch {0} Loss:{1:f}'.format(i, loss.item()))
        # compute accuracy
        strict_acc = 100 * (strict_success / total_count)
        easy_acc = 100 * (easy_success / total_count)
        iou_acc = 100 * (iou_success / total_count)
        for i in range(opt.n_cat):
            logger.info('{} accuracies:'.format(val_dataset.cat_names[i]))
            logger.info('5^o 5cm: {:4f}'.format(strict_acc[i]))
            logger.info('10^o 5cm: {:4f}'.format(easy_acc[i]))
            logger.info('IoU < 0.1: {:4f}'.format(iou_acc[i]))
        strict_acc = np.mean(strict_acc)
        easy_acc = np.mean(easy_acc)
        iou_acc = np.mean(iou_acc)
        val_loss = val_loss / val_size
        summary = tf.Summary(value=[tf.Summary.Value(tag='val_loss', simple_value=val_loss),
                                    tf.Summary.Value(tag='5^o5cm_acc', simple_value=strict_acc),
                                    tf.Summary.Value(tag='10^o5cm_acc', simple_value=easy_acc),
                                    tf.Summary.Value(tag='iou_acc', simple_value=iou_acc)])
        tb_writer.add_summary(summary, global_step)
        logger.info('Epoch {0:02d} test average loss: {1:06f}'.format(epoch, val_loss))
        logger.info('Overall accuracies:')
        logger.info('5^o 5cm: {:4f} 10^o 5cm: {:4f} IoU: {:4f}'.format(strict_acc, easy_acc, iou_acc))
        logger.info('>>>>>>>>----------Epoch {:02d} test finish---------<<<<<<<<'.format(epoch))
        # save model after each epoch
        torch.save(estimator.state_dict(), '{0}/model_{1:02d}.pth'.format(opt.result_dir, epoch))


if __name__ == '__main__':
    train_net()
