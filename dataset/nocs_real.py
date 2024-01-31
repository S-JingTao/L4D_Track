import os
import cv2
import math
import random
import torch
import numpy as np
import _pickle as cPickle
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from utils.nocs_real_data import load_depth, get_bbox


class PoseDataset(data.Dataset):
    def __init__(self, source, mode, data_dir, n_pts, img_size, cate_idx, caption_len=768):
        """
        Args:
            source: 'CAMERA', 'Real' or 'CAMERA+Real'
            mode: 'train' or 'test'
            data_dir: dataset path
            n_pts: number of selected foreground points
            img_size: square image window
            cate_idx: choose category, if cate_idx=1-6, train one categorical model or train a unified model
        """
        self.source = source
        self.mode = mode
        self.data_dir = data_dir
        self.n_pts = n_pts
        self.img_size = img_size
        self.caption_len = caption_len

        self.selected_cate_idx = cate_idx

        assert source in ['CAMERA', 'Real', 'CAMERA+Real']
        assert mode in ['train', 'test']
        img_list_path = ['CAMERA/train_list.txt', 'Real/train_list.txt',
                         'CAMERA/val_list.txt', 'Real/test_list.txt']
        model_file_path = ['obj_models/camera_train.pkl', 'obj_models/real_train.pkl',
                           'obj_models/camera_val.pkl', 'obj_models/real_test.pkl']

        self.train_caption_list = ["scene_1_caption.pt", "scene_2_caption.pt", "scene_3_caption.pt",
                                   "scene_4_caption.pt",
                                   "scene_5_caption.pt", "scene_6_caption.pt", "scene_7_caption.pt"]

        self.test_caption_list = ["scene_1_caption.pt", "scene_2_caption.pt", "scene_3_caption.pt",
                                  "scene_4_caption.pt", "scene_5_caption.pt", "scene_6_caption.pt"]

        self.caption_save_path = "/home/sjt/LV-Track/models/video_language_model/"

        if mode == 'train':
            del img_list_path[2:]
            del model_file_path[2:]
        else:
            del img_list_path[:2]
            del model_file_path[:2]
        if source == 'CAMERA':
            del img_list_path[-1]
            del model_file_path[-1]
        elif source == 'Real':
            del img_list_path[0]
            del model_file_path[0]
        else:
            # only use Real to test when source is CAMERA+Real
            if mode == 'test':
                del img_list_path[0]
                del model_file_path[0]

        img_list = []
        subset_len = []
        for path in img_list_path:
            img_list += [os.path.join(path.split('/')[0], line.rstrip('\n'))
                         for line in open(os.path.join(self.data_dir, path))]
            subset_len.append(len(img_list))
        if len(subset_len) == 2:
            self.subset_len = [subset_len[0], subset_len[1] - subset_len[0]]
        self.img_list = img_list
        self.length = len(self.img_list)

        models = {}
        for path in model_file_path:
            with open(os.path.join(data_dir, path), 'rb') as f:
                models.update(cPickle.load(f))
        self.models = models

        self.cat_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug', "all"]
        self.class_dict = {"1": "bottle", "2": "bowl", "3": "camera", "4": "can", "5": "laptop", "6": "mug"}
        self.class_num_list = [1, 2, 3, 4, 5, 6]
        self.sym_ids = [0, 1, 3]  # 0-indexed

        # # collect per-class data_list
        # train_per_class_path = ["train_bottle_list.txt", "train_bowl_list.txt", "train_camera_list.txt",
        #                         "train_can_list.txt", "train_laptop_list.txt", "train_mug_list.txt"]
        #
        # test_per_class_path = ["val_bottle_list.txt", "val_bowl_list.txt", "val_camera_list.txt",
        #                        "val_can_list.txt", "val_laptop_list.txt", "val_mug_list.txt"]
        # if self.selected_cate_idx in self.class_num_list:
        #
        #     if mode == 'train':
        #         for item in train_per_class_path:
        #             if self.class_dict[str(self.selected_cate_idx)] in item:
        #                 per_class_list_path = item
        #     else:
        #         for item in test_per_class_path:
        #             if self.class_dict[str(self.selected_cate_idx)] in item:
        #                 per_class_list_path = item
        #
        #     self.per_img_list = []
        #     self.per_img_list += [os.path.join("Real", line.rstrip('\n'))
        #                           for line in open(os.path.join(data_dir, "Real", per_class_list_path))]
        #
        #     self.per_length = len(self.per_img_list)
        #     self.length = self.per_length
        #     print('{0} images found in category: {1}.'.format(self.per_length,
        #                                                       self.class_dict[str(self.selected_cate_idx)]))

        # meta info for re-label mug category
        with open(os.path.join(data_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
            self.mug_meta = cPickle.load(f)

        # self.mean_shapes = np.load('../assets/mean_points_emb.npy')
        self.mean_shapes = np.load('./assets/mean_points_emb.npy')

        self.camera_intrinsics = [577.5, 577.5, 319.5, 239.5]  # [fx, fy, cx, cy]
        self.real_intrinsics = [591.0125, 590.16775, 322.525, 244.11084]

        self.norm_scale = 1000.0  # normalization scale
        self.xmap = np.array([[i for i in range(640)] for j in range(480)])
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])
        self.shift_range = 0.01
        self.colorjitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        print('{} images found in all.'.format(self.length))
        print('{} models loaded in all.'.format(len(self.models)))

    def __len__(self):
        return self.length

    def get_img_info(self, img_path):
        rgb = cv2.imread(img_path + '_color.png')[:, :, :3]
        rgb = rgb[:, :, ::-1]
        depth = load_depth(img_path)
        mask = cv2.imread(img_path + '_mask.png')[:, :, 2]
        coord = cv2.imread(img_path + '_coord.png')[:, :, :3]
        coord = coord[:, :, (2, 1, 0)]
        coord = np.array(coord, dtype=np.float32) / 255
        coord[:, :, 2] = 1 - coord[:, :, 2]

        with open(img_path + '_label.pkl', 'rb') as f:
            gts = cPickle.load(f)

        return rgb, depth, mask, coord, gts

    def sample_points(self, mask, depth, inst_id, rmin, rmax, cmin, cmax):
        mask = np.equal(mask, inst_id)
        mask = np.logical_and(mask, depth > 0)
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

        if len(choose) > self.n_pts:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.n_pts] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.n_pts - len(choose)), 'wrap')

        cam_fx, cam_fy, cam_cx, cam_cy = self.real_intrinsics

        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        pt2 = depth_masked / self.norm_scale
        pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
        points = np.concatenate((pt0, pt1, pt2), axis=1)

        return points, choose

    def resize_cropped_image(self, rgb, rmin, rmax, cmin, cmax, choose):
        # resize cropped image to standard size and adjust 'choose' accordingly
        rgb = rgb[rmin:rmax, cmin:cmax, :]
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        crop_w = rmax - rmin
        ratio = self.img_size / crop_w
        col_idx = choose % crop_w
        row_idx = choose // crop_w
        choose = (np.floor(row_idx * ratio) * self.img_size + np.floor(col_idx * ratio)).astype(np.int64)

        return rgb, choose

    def load_pose_label(self, cat_id, gts, idx, translation):
        scale = gts['scales'][idx]
        rotation = gts['rotations'][idx]
        # translation = gts['translations'][idx]
        if cat_id in self.sym_ids:
            rotation = gts['rotations'][idx]
            # assume continuous axis rotation symmetry
            theta_x = rotation[0, 0] + rotation[2, 2]
            theta_y = rotation[0, 2] - rotation[2, 0]
            r_norm = math.sqrt(theta_x ** 2 + theta_y ** 2)
            s_map = np.array([[theta_x / r_norm, 0.0, -theta_y / r_norm],
                              [0.0, 1.0, 0.0],
                              [theta_y / r_norm, 0.0, theta_x / r_norm]])
            rotation = rotation @ s_map

        sRT = np.identity(4, dtype=np.float32)
        sRT[:3, :3] = scale * rotation
        # sRT[:3, :3] = rotation / scale
        sRT[:3, 3] = translation
        return sRT

    def data_augmentation(self, rgb, points, translation):
        if self.mode == 'train':
            # color jitter
            rgb = self.colorjitter(Image.fromarray(np.uint8(rgb)))
            rgb = np.array(rgb)
            # point shift
            add_t = np.random.uniform(-self.shift_range, self.shift_range, (1, 3))
            translation = translation + add_t[0]
            # point jitter
            add_t = add_t + np.clip(0.001 * np.random.randn(points.shape[0], 3), -0.005, 0.005)
            points = np.add(points, add_t)
        rgb = self.transform(rgb)
        points = points.astype(np.float32)
        return rgb, points, translation

    def reset_pair_frame_path_all(self, index):

        if index == self.length:
            img_path = os.path.join(self.data_dir, self.img_list[index - 2])
            cur_img_path = os.path.join(self.data_dir, self.img_list[index - 1])
        elif index == 0:
            img_path = os.path.join(self.data_dir, self.img_list[index])
            cur_img_path = os.path.join(self.data_dir, self.img_list[index + 1])
        elif os.path.join(self.data_dir, self.img_list[index]).split('/')[-2] != \
                os.path.join(self.data_dir, self.img_list[index - 1]).split('/')[-2]:
            img_path = os.path.join(self.data_dir, self.img_list[index])
            cur_img_path = os.path.join(self.data_dir, self.img_list[index + 1])
        else:
            cur_img_path = os.path.join(self.data_dir, self.img_list[index])
            img_path = os.path.join(self.data_dir, self.img_list[index - 1])
        return cur_img_path, img_path

    def reset_pair_frame_path_per(self, index):

        if index == self.per_length:
            img_path = os.path.join(self.data_dir, self.per_img_list[index - 2])
            cur_img_path = os.path.join(self.data_dir, self.per_img_list[index - 1])
        elif index == 0:
            img_path = os.path.join(self.data_dir, self.per_img_list[index])
            cur_img_path = os.path.join(self.data_dir, self.per_img_list[index + 1])
        elif os.path.join(self.data_dir, self.per_img_list[index]).split('/')[-2] != \
                os.path.join(self.data_dir, self.per_img_list[index - 1]).split('/')[-2]:
            img_path = os.path.join(self.data_dir, self.per_img_list[index])
            cur_img_path = os.path.join(self.data_dir, self.per_img_list[index + 1])
        else:
            cur_img_path = os.path.join(self.data_dir, self.per_img_list[index])
            img_path = os.path.join(self.data_dir, self.per_img_list[index - 1])
        return cur_img_path, img_path

    def random_choice_inter_frame(self, index):
        cur_img_path = os.path.join(self.data_dir, self.img_list[index])
        cur_scene = cur_img_path.split("/")[-2]
        while True:
            pre_img_num = random.randint(0, 100)
            pre_img_num = str(pre_img_num).zfill(4)
            if os.path.join(self.img_list[index].split("/")[0], self.img_list[index].split("/")[1],
                            cur_scene, pre_img_num) in self.img_list:
                break

        img_path = os.path.join(self.data_dir, self.img_list[index].split("/")[0], self.img_list[index].split("/")[1],
                                cur_scene, pre_img_num)

        return cur_img_path, img_path

    def __getitem__(self, index):
        # if self.selected_cate_idx not in self.class_num_list:
        #     cur_img_path, img_path = self.reset_pair_frame_path_all(index)
        # else:
        #     cur_img_path, img_path = self.reset_pair_frame_path_per(index)

        # cur_img_path, img_path = self.reset_pair_frame_path_all(index)

        # random choice a inter_frame
        cur_img_path, img_path = self.random_choice_inter_frame(index)

        # load corresponding caption embeddings
        cur_data_scene = cur_img_path.split('/')[-2]
        caption_embed = torch.empty((3, self.caption_len), dtype=torch.float64)
        if self.mode == "train":
            for file_temp in self.train_caption_list:
                if cur_data_scene == file_temp.split('_caption.pt')[0]:
                    caption_dir = os.path.join(self.caption_save_path, "train_caption", file_temp)
                    caption_embed = torch.load(caption_dir)

        elif self.mode == "test":
            for file_temp in self.test_caption_list:
                if cur_data_scene == file_temp.split('_caption.pt')[0]:
                    caption_dir = os.path.join(self.caption_save_path, "test_caption", file_temp)
                    caption_embed = torch.load(caption_dir)
        else:
            raise ValueError("Train mode error!")

        # print("num:", index)
        # print(self.mode)
        # print("pre:", img_path.split('/')[-2:])
        # print("cur:", cur_img_path.split('/')[-2:])

        pre_rgb, pre_depth, pre_mask, pre_coord, pre_gts = self.get_img_info(img_path)
        cur_rgb, cur_depth, cur_mask, cur_coord, cur_gts = self.get_img_info(cur_img_path)

        while True:
            shot_idx = random.randint(0, len(cur_gts['instance_ids']) - 1)
            shot_model = cur_gts['model_list'][shot_idx]
            shot_class = cur_gts['class_ids'][shot_idx]
            if shot_model in pre_gts['model_list'] and shot_class in pre_gts['class_ids']:
                break

        # if self.class_dict[str(shot_class)] == "camera" or self.class_dict[str(shot_class)] == 'laptop':
        #     print(self.class_dict[str(shot_class)])

        while shot_model in pre_gts['model_list'] and shot_class in pre_gts['class_ids']:
            cur_ins_idx = shot_idx
            pre_ins_idx = pre_gts['model_list'].index(shot_model)
            cur_ins_ids = cur_gts['instance_ids'][shot_idx]
            pre_ins_ids = pre_gts['instance_ids'][pre_ins_idx]
            shot_cate = shot_class

            # if self.selected_cate_idx not in self.class_num_list:
            #     # select one foreground object
            #     shot_idx = random.randint(0, len(cur_gts['instance_ids']) - 1)
            #
            #     shot_cate = cur_gts["class_ids"][shot_idx]
            #
            #     cur_ins_ids = cur_gts['instance_ids'][shot_idx]
            #     cur_ins_idx = shot_idx
            #
            #     # print("random choose")
            #     # print('category:', self.class_dict[str(shot_cate)])
            #     # print('-----------')
            #     try:
            #         pre_ins_idx = pre_gts['model_list'].index(cur_gts['model_list'][shot_idx])
            #     except ValueError:
            #         print('error! %s is not in instance list' % cur_gts['model_list'][shot_idx])
            #         pre_ins_idx = pre_gts['class_ids'].index(cur_gts["class_ids"][shot_idx])
            #
            #     # pre_ins_ids = pre_gts['instance_ids'][pre_ins_idx]
            # else:
            #     shot_cate = self.selected_cate_idx
            #     pre_all_shot_class_ids = [i for i, x in enumerate(list(cur_gts["class_ids"])) if x == shot_cate]
            #     # for multiple same category in same scene, random choose one
            #     shot_idx = random.choice(pre_all_shot_class_ids)
            #     cur_ins_idx = shot_idx
            #     cur_ins_ids = cur_gts['instance_ids'][shot_idx]
            #     try:
            #         pre_ins_idx = pre_gts['model_list'].index(cur_gts['model_list'][shot_idx])
            #     except ValueError:
            #         print('error! %s is not in instance list' % cur_gts['model_list'][shot_idx])
            #         pre_ins_idx = pre_gts['class_ids'].index(cur_gts["class_ids"][shot_idx])
            #
            #     # print("selected :", self.class_dict[str(shot_cate)])
            #     # print('category:', self.class_dict[str(shot_cate)])
            #     # print('-----------')
            #
            # pre_ins_ids = pre_gts['instance_ids'][pre_ins_idx]
            pre_rmin, pre_rmax, pre_cmin, pre_cmax = get_bbox(pre_gts['bboxes'][pre_ins_idx])
            cur_rmin, cur_rmax, cur_cmin, cur_cmax = get_bbox(cur_gts['bboxes'][cur_ins_idx])

            # sample points
            cur_points, cur_choose = self.sample_points(cur_mask, cur_depth, cur_ins_ids,
                                                        cur_rmin, cur_rmax, cur_cmin, cur_cmax)
            pre_points, pre_choose = self.sample_points(pre_mask, pre_depth, pre_ins_ids,
                                                        pre_rmin, pre_rmax, pre_cmin, pre_cmax)

            cur_rgb, cur_choose = self.resize_cropped_image(cur_rgb, cur_rmin, cur_rmax, cur_cmin, cur_cmax, cur_choose)
            pre_rgb, pre_choose = self.resize_cropped_image(pre_rgb, pre_rmin, pre_rmax, pre_cmin, pre_cmax, pre_choose)

            # common label
            cat_id = shot_cate - 1  # convert to 0-indexed
            model = self.models[cur_gts['model_list'][shot_idx]].astype(np.float32)  # 1024 points
            prior = self.mean_shapes[cat_id].astype(np.float32)

            # data augmentation
            cur_rgb, cur_points, cur_adapt_translation = self.data_augmentation(cur_rgb, cur_points,
                                                                                cur_gts['translations'][cur_ins_idx])
            pre_rgb, pre_points, pre_adapt_translation = self.data_augmentation(pre_rgb, pre_points,
                                                                                pre_gts['translations'][pre_ins_idx])

            # pose label
            cur_sRT = self.load_pose_label(cat_id, cur_gts, cur_ins_idx, cur_adapt_translation)
            pre_sRT = self.load_pose_label(cat_id, pre_gts, pre_ins_idx, pre_adapt_translation)

            pre_data = [pre_points, pre_rgb, pre_choose, pre_sRT]
            cur_data = [cur_points, cur_rgb, cur_choose, cur_sRT]

            return pre_data, cur_data, caption_embed, model, prior, cat_id


def main():
    a = PoseDataset
    res = a("Real", "test", "/home/amax/document/sjt_project/datasets/MY_NOCS/", 2048, 240, 6)
    res_1 = res.__getitem__(602)
    all_len = res.__len__()
    for num in range(0, all_len):
        res_1 = res.__getitem__(num)


if __name__ == '__main__':
    main()
