import math

import numpy as np

from lib.models.hdetrack import build_hdetrack_s
from lib.test.tracker.basetracker import BaseTracker
import torch
import copy

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
import torch.nn.functional as F
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
# from .show_CAM import getCAM2

class HDETrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(HDETrack, self).__init__(params)
        network = build_hdetrack_s(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu', weights_only=False)['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

        # Counts how many more post-flash frames should still use sliding-window redetection.
        self.flash_cooldown = 0

    def initialize(self, image, start_frame_idx,  info: dict):
        ## forward the template once
        z_patch_arr, resize_factor, z_amask_arr, crop_coor = sample_target(image, info['init_bbox'],
                                                                           self.params.template_factor,
                                                                           output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = start_frame_idx
        self.flash_cooldown = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1

        is_flash = bool(info.get('flash', False)) if info else False

        # Redetection gate: cooldown=3 set on flash; sliding window runs ONLY on
        # the 3 post-flash frames, not during the flash itself.
        if is_flash:
            self.flash_cooldown = 3
            use_redetect = False
        else:
            use_redetect = self.flash_cooldown > 0

        if use_redetect:
            # ---------- Sliding-window batch ----------
            win = int(self.params.search_factor * max(self.state[2], self.state[3]))
            win = max(1, min(win, H, W))
            step = max(1, int(win * 0.7))  # 30% overlap

            state_cx = self.state[0] + 0.5 * self.state[2]
            state_cy = self.state[1] + 0.5 * self.state[3]
            local_x0 = max(0, min(int(round(state_cx - win / 2)), W - win))
            local_y0 = max(0, min(int(round(state_cy - win / 2)), H - win))
            positions = [(local_x0, local_y0)]

            xs = list(range(0, max(1, W - win), step))
            if not xs or xs[-1] + win < W:
                xs.append(max(0, W - win))
            ys = list(range(0, max(1, H - win), step))
            if not ys or ys[-1] + win < H:
                ys.append(max(0, H - win))
            for y0 in ys:
                for x0 in xs:
                    if (x0, y0) != positions[0]:
                        positions.append((x0, y0))

            empty_mask = np.zeros((self.params.search_size, self.params.search_size), dtype=bool)
            crops = []
            crops_resized = []
            for (x0, y0) in positions:
                crop = image[y0:y0 + win, x0:x0 + win]
                crop_resized = cv2.resize(crop, (self.params.search_size, self.params.search_size))
                processed = self.preprocessor.process(crop_resized, empty_mask)
                crops.append(processed.tensors)
                crops_resized.append(crop_resized)

            batch = torch.cat(crops, dim=0)
            N = batch.shape[0]
            template_imgs = self.z_dict1.tensors.expand(N, *self.z_dict1.tensors.shape[1:])
            box_mask_batch = (
                self.box_mask_z.expand(N, *self.box_mask_z.shape[1:])
                if self.box_mask_z is not None else None
            )

            with torch.no_grad():
                out_dict = self.network.forward(
                    event_template_img=template_imgs,
                    event_search_img=batch,
                    ce_template_mask=box_mask_batch,
                )

            # Pick window with the highest score-map peak.
            score_map = out_dict['s_score_map']
            scores, _ = score_map.flatten(1).max(dim=1)
            best_idx = scores.argmax().item()
            x0_best, y0_best = positions[best_idx]

            best_score_map = score_map[best_idx:best_idx + 1]
            best_size_map = out_dict['s_size_map'][best_idx:best_idx + 1]
            best_offset_map = out_dict['s_offset_map'][best_idx:best_idx + 1]

            pred_boxes = self.network.box_head.cal_bbox(best_score_map, best_size_map, best_offset_map)
            pred_boxes = pred_boxes.view(-1, 4)

            resize_factor = self.params.search_size / win
            pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
            cx, cy, w, h = pred_box
            half_side = 0.5 * self.params.search_size / resize_factor
            window_cx = x0_best + win / 2
            window_cy = y0_best + win / 2
            cx_real = cx + (window_cx - half_side)
            cy_real = cy + (window_cy - half_side)
            self.state = clip_box([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], H, W, margin=10)

            # Kept for the debug block below.
            pred_score_map = best_score_map
            x_patch_arr = crops_resized[best_idx]
        else:
            # ---------- Original local-only path ----------
            x_patch_arr, resize_factor, x_amask_arr, crop_coor = sample_target(image, self.state, self.params.search_factor,
                                                                    output_sz=self.params.search_size)
            search = self.preprocessor.process(x_patch_arr, x_amask_arr)

            with torch.no_grad():
                x_dict = search
                out_dict = self.network.forward(
                    event_template_img=self.z_dict1.tensors,
                    event_search_img=x_dict.tensors,
                    ce_template_mask=self.box_mask_z)

            # add hann windows
            pred_score_map = out_dict['s_score_map']
            response = self.output_window * pred_score_map
            pred_boxes = self.network.box_head.cal_bbox(response, out_dict['s_size_map'], out_dict['s_offset_map'])
            pred_boxes = pred_boxes.view(-1, 4)
            pred_box = (pred_boxes.mean(
                dim=0) * self.params.search_size / resize_factor).tolist()
            self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # Decrement cooldown on non-flash frames (flash frames reset it to 3).
        if not is_flash and self.flash_cooldown > 0:
            self.flash_cooldown -= 1

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return HDETrack
