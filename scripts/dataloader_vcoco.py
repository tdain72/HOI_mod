#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import helpers_preprocess as labels
from PIL import Image
import matplotlib.pyplot as plt

(bad_detections_train, bad_detections_val, bad_detections_test) = \
    labels.dry_run()

# bad_detections_train,bad_detections_val,bad_detections_test=[],[],[]

NO_VERB = 29
NO_OBJ_CAT = 80

# def get_ambiguity_score(prior_mat, labels_all):

# def get_ambiguity_score(prior_mat, labels_all, labels_object):
#     labels_all_re = labels_all.reshape((labels_all.shape[0]*labels_all.shape[1], labels_all.shape[2]))
#     verb_prior = np.zeros((NO_VERB, NO_OBJ_CAT))
#     ambiguity_score = np.zeros((labels_object.shape[0], NO_VERB))
#     for c in range(NO_OBJ_CAT):
#         for w in range(NO_VERB):
#             cnt = 0
#             for h in range(NO_VERB):
#                 if prior_mat[w,h,c] > 0 and w!=h:
#                     verb_prior[w,c] += prior_mat[w,h,c]
#                     cnt += 1
#             if cnt > 0:
#                 verb_prior[w,c] = verb_prior[w,c] / cnt
    
#     for i in range(labels_object.shape[0]):
#         if labels_object[i]-1 < 0:
#             continue
#         ambiguity_score[i] = verb_prior[:, labels_object[i]-1] * labels_all_re[i]
#     return ambiguity_score

def get_ambiguity_score(prior_mat, labels_all):
    labels_all_re = labels_all.reshape((labels_all.shape[0]*labels_all.shape[1], labels_all.shape[2]))
    # pos_idx = np.greater(labels_all_re, 0)
    tmp = np.matmul(prior_mat, labels_all_re.T)
    ambiguity_score = tmp*labels_all_re.T
    # ambiguity_score = 9*np.log10(1/(1+np.exp(-ambiguity_score.T)))+1
    # ambiguity_score = (ambiguity_score*pos_idx)
    return ambiguity_score.T

def vcoco_collate(batch):
    image = []
    image_id = []
    pairs_info = []
    labels_all = []
    labels_single = []
    ambiguity_score = []
    class_dist = []
    class_bias = []
    neg_num = []
    pos_num = []
    for (index, item) in enumerate(batch):
        image.append(item['image'])
        image_id.append(torch.tensor(int(item['image_id'])))
        pairs_info.append(torch.tensor(np.shape(item['labels_all'])))
        tot_HOI = int(np.shape(item['labels_single'])[0])
        labels_all.append(torch.tensor(item['labels_all'
                          ].reshape(tot_HOI, NO_VERB)))
        labels_single.append(torch.tensor(item['labels_single']))
        ambiguity_score.append(torch.tensor(item['ambiguity_score']))
        class_dist.append(torch.tensor(item['class_dist']))
        class_bias.append(torch.tensor(item['class_bias']))
        neg_num.append(torch.tensor(item['negative_num']))
        pos_num.append(torch.tensor(item['positive_num']))
    return [torch.stack(image), torch.cat(labels_all),
            torch.cat(labels_single), torch.stack(image_id),
            torch.stack(pairs_info), torch.cat(ambiguity_score), 
            torch.cat(class_dist), torch.cat(class_bias),
            torch.cat(neg_num), torch.cat(pos_num)]


class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):

        (h, w) = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                (new_h, new_w) = (self.output_size * h / w,
                                  self.output_size)
            else:
                (new_h, new_w) = (self.output_size, self.output_size
                                  * w / h)
        else:
            (new_h, new_w) = self.output_size

        (new_h, new_w) = (int(new_h), int(new_w))
        img2 = transform.resize(image, (new_h, new_w))

        return img2


class ToTensor(object):

    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):

        image = image.transpose((2, 0, 1))

        return torch.from_numpy(image).float()


class vcoco_Dataset(Dataset):

    def __init__(
        self,
        json_file_image,
        root_dir,
        transform=None,
        ):
        # Load prior matrix for loss
        with open('/home/d9/Documents/VSGNet/All_data/Data_vcoco/prior/cooccur_mat_human.npy', 'rb') as prior_mat:
            self.prior_mat = np.load(prior_mat)
        with open('/home/d9/Documents/VSGNet/All_data/Data_vcoco/prior/data_dist.npy', 'rb') as data_json:
            self.data_dist = np.load(data_json)
        with open('/home/d9/Documents/VSGNet/All_data/Data_vcoco/prior/data_bias.npy', 'rb') as bias:
            self.data_bias = np.load(bias)
        with open('/home/d9/Documents/VSGNet/All_data/Data_vcoco/prior/negative_num.npy', 'rb') as negative_num:
            self.negative_num = np.load(negative_num) 
        with open('/home/d9/Documents/VSGNet/All_data/Data_vcoco/prior/positive_num.npy', 'rb') as positive_num:
            self.positive_num = np.load(positive_num)
        #---------------------------
        with open(json_file_image) as json_file_:
            self.vcoco_frame_file = json.load(json_file_)
        self.flag = json_file_image.split('/')[-1].split('_')[0]
        if self.flag == 'train':
            self.vcoco_frame = [x for x in list(self.vcoco_frame_file.keys())
                                if x not in str(bad_detections_train)]
        elif self.flag == 'val':
            self.vcoco_frame = [x for x in list(self.vcoco_frame_file.keys())
                                if x not in str(bad_detections_val)]
        elif self.flag == 'test':
            self.vcoco_frame = [x for x in list(self.vcoco_frame_file.keys())
                                if x not in str(bad_detections_test)]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.vcoco_frame)

    def __getitem__(self, idx):
        if self.flag == 'test':
            img_pre_suffix = 'COCO_val2014_' \
                + str(self.vcoco_frame[idx]).zfill(12) + '.jpg'
        else:
            img_pre_suffix = 'COCO_train2014_' \
                + str(self.vcoco_frame[idx]).zfill(12) + '.jpg'
        all_labels = \
            labels.get_compact_label(int(self.vcoco_frame[idx]),
                self.flag)
        labels_all = all_labels['labels_all']
        labels_single = all_labels['labels_single']
        #-New label--------------
        labels_object = all_labels['labels_object']
        labels_ambiguity = get_ambiguity_score(self.prior_mat, labels_all)
        ones = torch.ones(labels_object.shape[0], NO_VERB)
        labels_dist = ones * self.data_dist
        labels_bias = ones * self.data_bias
        labels_negative_num = ones * self.negative_num
        labels_positive_num = ones * self.positive_num
        #-----------------------

        img_name = os.path.join(self.root_dir, img_pre_suffix)
        ids = [int(self.vcoco_frame[idx]), self.flag]
        image = Image.open(img_name).convert('RGB')
        image = np.array(image)

        if self.transform:
            image = self.transform(image)
        sample = {
            'image': image,
            'labels_all': labels_all,
            'labels_single': labels_single,
            'image_id': self.vcoco_frame[idx],
            'ambiguity_score': labels_ambiguity,
            'class_dist': labels_dist,
            'class_bias': labels_bias,
            'negative_num': labels_negative_num,
            'positive_num': labels_positive_num
            }
        return sample
