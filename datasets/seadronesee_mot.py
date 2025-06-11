from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import random
import os.path as osp
from .bases import BaseImageDataset
from collections import defaultdict
import pickle

class SeaDroneSee2019MOT(BaseImageDataset):
    """
    SeaDroneSee2019MOT
    """
    dataset_dir = 'bounding_box_train'
    def __init__(self, root='', verbose=True, test_size=800, **kwargs):
        super(SeaDroneSee2019MOT, self).__init__()
        self.dataset_dir = root 
        self.bbox_folder = osp.join(self.dataset_dir, 'bounding_box_train')
        self.img_dir = self.bbox_folder
        self.split_dir = osp.join(self.dataset_dir, 'train_test_split')
        self.train_list = osp.join(self.split_dir, 'train_list.txt')
        self.test_list = osp.join(self.split_dir, 'val_list.txt')
        self.test_size = test_size
        print(self.test_list)

        self.check_before_run()

        train, query, gallery = self.process_split(relabel=True)
        self.train = train
        self.query = query
        self.gallery = gallery

        if verbose:
            print('=> SeaDroneSee2019MOT loaded')
            self.print_dataset_statistics(train, query, gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError('"{}" is not available'.format(self.dataset_dir))
        if not osp.exists(self.split_dir):
            raise RuntimeError('"{}" is not available'.format(self.split_dir))
        if not osp.exists(self.train_list):
            raise RuntimeError('"{}" is not available'.format(self.train_list))
        if not osp.exists(self.test_list):
            raise RuntimeError('"{}" is not available'.format(self.test_list))

    def get_pid2label(self, pids):
        pid_container = set(pids)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        return pid2label


    def parse_img_pids(self, nl_pairs, pid2label=None, cam=0):
        output = []
        for info in nl_pairs:
            name = info[0]
            pid = info[1]
            seq = info[2]
            if pid2label is not None:
                pid = pid2label[pid]
            camid = cam
            img_path = osp.join(self.img_dir, seq, name)
            viewid = 1
            output.append((img_path, pid, camid, viewid))
        return output
        

    def get_imagedata_info(self, data):
        pids, cams, imgs, vids = set(), set(), set(), set()
        for item in data:
            img_path, pid, camid, viewid = item
            pids.add(pid)
            cams.add(camid)
            imgs.add(img_path)
            vids.add(viewid)
        return len(pids), len(imgs), len(cams), len(vids)

    def process_split(self, relabel=False):
        train_pid_dict = defaultdict(list)
        with open(self.train_list) as f_train:
            train_data = f_train.readlines()
            for data in train_data:
                name, pid, seq = data.strip().split(' ')
                pid = int(pid)
                train_pid_dict[pid].append([name, pid, seq])
        train_pids = list(train_pid_dict.keys())

        test_pid_dict = defaultdict(list)
        with open(self.test_list) as f_test:
            test_data = f_test.readlines()
            for data in test_data:
                name, pid, seq = data.strip().split(' ')
                pid = int(pid)
                test_pid_dict[pid].append([name, pid, seq])
        test_pids = list(test_pid_dict.keys())

        train_data = []
        query_data = []
        gallery_data = []
        train_pids = sorted(train_pids)
        for pid in train_pids:
            imginfo = train_pid_dict[pid]
            train_data.extend(imginfo)

        for pid in test_pids:
            imginfo = test_pid_dict[pid]
            sample = random.choice(imginfo)
            imginfo.remove(sample)
            query_data.extend(imginfo)
            gallery_data.append(sample)

        if relabel:
            train_pid2label = self.get_pid2label(train_pids)
        else:
            train_pid2label = None

        train = self.parse_img_pids(train_data, train_pid2label)
        query = self.parse_img_pids(query_data, cam=0)
        gallery = self.parse_img_pids(gallery_data, cam=1)

        return train, query, gallery