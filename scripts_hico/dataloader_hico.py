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

(bad_detections_train, bad_detections_test) = labels.dry_run()

# bad_detections_train,bad_detections_val,bad_detections_test=[],[],[]

NO_VERB = 117
NO_OBJ_CAT = 80

def get_ambiguity_score(prior_mat, labels_all, labels_object):
    labels_all_re = labels_all.reshape((labels_all.shape[0]*labels_all.shape[1], labels_all.shape[2]))
    verb_prior = np.zeros((NO_VERB, NO_OBJ_CAT))
    ambiguity_score = np.zeros((labels_object.shape[0], NO_VERB))
    for c in range(NO_OBJ_CAT):
        for w in range(NO_VERB):
            cnt = 0
            for h in range(NO_VERB):
                if prior_mat[w,h,c] > 0 and w!=h:
                    verb_prior[w,c] += prior_mat[w,h,c]
                    cnt += 1
            if cnt > 0:
                verb_prior[w,c] = verb_prior[w,c] / cnt
    for i in range(labels_object.shape[0]):
        if labels_object[i]-1 < 0:
            continue
        ambiguity_score[i,:] = verb_prior[:, labels_object[i]-1] * labels_all_re[i]
    return ambiguity_score

def hico_collate(batch):
    image = []
    image_id = []
    pairs_info = []
    labels_all = []
    labels_single = []
    ambiguity_score = []
    for (index, item) in enumerate(batch):
        image.append(item['image'])
        image_id.append(torch.tensor(int(item['image_id'])))
        pairs_info.append(torch.tensor(np.shape(item['labels_all'])))
        tot_HOI = int(np.shape(item['labels_single'])[0])
        labels_all.append(torch.tensor(item['labels_all'
                          ].reshape(tot_HOI, NO_VERB)))
        labels_single.append(torch.tensor(item['labels_single']))
        ambiguity_score.append(torch.tensor(item['ambiguity_score']))
    return [torch.stack(image), torch.cat(labels_all),
            torch.cat(labels_single), torch.stack(image_id),
            torch.stack(pairs_info), torch.cat(ambiguity_score)]


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


class hico_Dataset(Dataset):

    def __init__(
        self,
        json_file_image,
        root_dir,
        transform=None,
        ):
        # Load prior matrix for loss
        with open('/home/d9/Documents/VSGNet/All_data/Data_hico/prior/cooccur_mat.npy', 'rb') as prior_mat:
            self.prior_mat = np.load(prior_mat)
        #---------------------------
        with open(json_file_image) as json_file_:
            self.hico_frame_file = json.load(json_file_)
        self.flag = json_file_image.split('/')[-1].split('_')[0]
        if self.flag == 'train':
            self.hico_frame = [x for x in list(self.hico_frame_file.keys())
                               if x not in str(bad_detections_train)]
        elif self.flag == 'test':
            self.hico_frame = [x for x in list(self.hico_frame_file.keys())
                               if x not in str(bad_detections_test)]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.hico_frame)

    def __getitem__(self, idx):
        if self.flag == 'test':
            img_pre_suffix = 'HICO_test2015_' \
                + str(self.hico_frame[idx]).zfill(8) + '.jpg'
        else:
            img_pre_suffix = 'HICO_train2015_' \
                + str(self.hico_frame[idx]).zfill(8) + '.jpg'
        all_labels = \
            labels.get_compact_label(int(self.hico_frame[idx]),
                self.flag)
        labels_all = all_labels['labels_all']
        labels_single = all_labels['labels_single']
        # New label
        labels_object = all_labels['labels_object'] 
        labels_ambiguity = get_ambiguity_score(self.prior_mat, labels_all, labels_object)

        img_name = os.path.join(self.root_dir, img_pre_suffix)
        ids = [int(self.hico_frame[idx]), self.flag]
        image = Image.open(img_name).convert('RGB')
        image = np.array(image)

        if self.transform:
            image = self.transform(image)
        sample = {
            'image': image,
            'labels_all': labels_all,
            'labels_single': labels_single,
            'image_id': self.hico_frame[idx],
            'ambiguity_score': labels_ambiguity,
            }
        return sample
