import os
import os.path
import numpy as np
import torch
import pandas
import string
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from ..admin import env_settings
from ..data import opencv_loader


class Illumination(BaseVideoDataset):
    def __init__(self, root=None, image_loader=opencv_loader, split=None):
        root = env_settings().illumination_dir_train if root is None else root
        super().__init__('Illumination', root, image_loader)

        self.sequence_list = self._get_sequence_list()

        if split is not None:
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'train':
                file_path = os.path.join(ltr_path, 'data_specs', 'illumination_train.txt')
            elif split == 'val':
                file_path = os.path.join(ltr_path, 'data_specs', 'illumination_val.txt')
            else:
                raise ValueError('Unknown split name.')
            with open(file_path) as f:
                seq_names = [line.strip() for line in f.readlines()]
        else:
            seq_names = self.sequence_list

        self.sequence_list = [i for i in seq_names]
        self.sequence_meta_info = self._load_meta_info()
        self.seq_per_class = self._build_seq_per_class()
        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()

    def _build_seq_per_class(self):
        seq_per_class = {}
        for i, s in enumerate(self.sequence_list):
            object_class = self.sequence_meta_info[s]['object_class_name']
            if object_class in seq_per_class:
                seq_per_class[object_class].append(i)
            else:
                seq_per_class[object_class] = [i]
        return seq_per_class

    def _get_sequence_list(self):
        seq_list = [s for s in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, s))]
        return seq_list

    def _load_meta_info(self):
        sequence_meta_info = {s: self._read_meta(os.path.join(self.root, s)) for s in self.sequence_list}
        return sequence_meta_info

    def get_num_classes(self):
        return len(self.class_list)

    def get_name(self):
        return 'illumination'

    def get_num_sequences(self):
        return len(self.sequence_list)

    def _read_meta(self, seq_path):
        obj_class = self._get_class(seq_path)
        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})
        return object_meta

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _get_class(self, seq_path):
        raw_class = seq_path.split('/')[-1].rstrip(string.digits).rstrip('_').split('_')[0]
        return raw_class

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def _get_event(self, seq_path, frame_id, cfg=None):
        if cfg.MODEL.T == 1:
            event1 = os.path.join(seq_path, 'inter1_stack_3008', '{:04d}_1.png'.format(frame_id + 1))
            return self.image_loader(event1)
        elif cfg.MODEL.T == 2:
            event1 = os.path.join(seq_path, 'inter2_stack_3008', '{:04d}_1.png'.format(frame_id + 1))
            event2 = os.path.join(seq_path, 'inter2_stack_3008', '{:04d}_2.png'.format(frame_id + 1))
            return self.image_loader(event1), self.image_loader(event2)
        elif cfg.MODEL.T == 4:
            event1 = os.path.join(seq_path, 'inter4_stack_3008', '{:04d}_1.png'.format(frame_id + 1))
            event2 = os.path.join(seq_path, 'inter4_stack_3008', '{:04d}_2.png'.format(frame_id + 1))
            event3 = os.path.join(seq_path, 'inter4_stack_3008', '{:04d}_3.png'.format(frame_id + 1))
            event4 = os.path.join(seq_path, 'inter4_stack_3008', '{:04d}_4.png'.format(frame_id + 1))
            return self.image_loader(event1), self.image_loader(event2), self.image_loader(event3), self.image_loader(event4)

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, 'groundtruth_rect.txt')
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def get_frames(self, seq_id=None, frame_ids=None, anno=None, cfg=None):
        seq_path = self._get_sequence_path(seq_id)
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]
        event_list = [self._get_event(seq_path, f_id, cfg) for f_id in frame_ids]
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
        return event_list, anno_frames, obj_meta
