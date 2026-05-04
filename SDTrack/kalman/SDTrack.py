import math

import numpy as np

from lib.models.SDTrack import build_SDTrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
from lib.utils.misc import NestedTensor


class KalmanCenter2D:
    """Constant-velocity Kalman filter on the bbox center.

    State : [cx, cy, vx, vy]   (4-D)
    Meas  : [cx, cy]           (2-D)
    Used to predict the target center during a flash, when the network
    output is unreliable.
    """

    def __init__(self, cx, cy, q=1.0, r=10.0, p0=100.0):
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=float)
        self.P = np.eye(4) * p0
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)
        self.Q = np.eye(4) * q
        self.R = np.eye(2) * r

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0], self.x[1]

    def update(self, cx, cy):
        z = np.array([cx, cy], dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P


class SDTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(SDTrack, self).__init__(params)
        network = build_SDTrack(params.cfg, training=False)
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
        # Kalman filter on the bbox center (init in `initialize`).
        self.kalman = None
        # Number of consecutive flash frames where Kalman has driven the output.
        # We only use Kalman for the first 2 frames of any flash burst; after
        # that the network output is written even if flash continues.
        self.flash_kalman_count = 0
        # (w, h) snapshot taken at the start of a flash; reused while Kalman
        # drives the output so the size stays constant during the flash.
        self.frozen_wh = None
        # SDTracker's own track of the bbox — always the raw network output.
        # Used as the reference for the next frame's search region so the
        # network is never affected by Kalman, even during flash frames.
        self._sd_state = None

    def initialize(self, image, info: dict):
        if self.cfg.MODEL.T == 1:
            # forward the template once
            z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
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
            self._sd_state = list(info['init_bbox'])  # SDTracker's independent track
            self.frame_id = 0
            # Initialize Kalman with the init-bbox center, zero velocity.
            init_cx = info['init_bbox'][0] + 0.5 * info['init_bbox'][2]
            init_cy = info['init_bbox'][1] + 0.5 * info['init_bbox'][3]
            self.kalman = KalmanCenter2D(init_cx, init_cy)
            self.flash_kalman_count = 0
            self.frozen_wh = None
            if self.save_all_boxes:
                '''save all predicted boxes'''
                all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
                return {"all_boxes": all_boxes_save}

        elif self.cfg.MODEL.T == 2:
            z_patch_arr = [0.0, 0.0]
            z_amask_arr = [0.0, 0.0]
            template = [0.0, 0.0]


            z_patch_arr[0], resize_factor, z_amask_arr[0] = sample_target(image[0], info['init_bbox'], self.params.template_factor,
                                                        output_sz=self.params.template_size)
            z_patch_arr[1], resize_factor, z_amask_arr[1] = sample_target(image[1], info['init_bbox'], self.params.template_factor,
                                                        output_sz=self.params.template_size)
            self.z_patch_arr = z_patch_arr[0]

            template[0] = self.preprocessor.process(z_patch_arr[0], z_amask_arr[0])
            template[1] = self.preprocessor.process(z_patch_arr[1], z_amask_arr[1])

            tensor = torch.cat([template[0].tensors.unsqueeze(0), template[1].tensors.unsqueeze(0)], dim=0)
            mask = torch.cat([template[0].mask.unsqueeze(0), template[1].mask.unsqueeze(0)], dim=0)

            template = NestedTensor(tensor, mask)

            with torch.no_grad():
                self.z_dict1 = template

            self.box_mask_z = None
            if self.cfg.MODEL.BACKBONE.CE_LOC:
                template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                            template.tensors.device).squeeze(1)
                self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

            # save states
            self.state = info['init_bbox']
            self.frame_id = 0
            if self.save_all_boxes:
                '''save all predicted boxes'''
                all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
                return {"all_boxes": all_boxes_save}

        elif self.cfg.MODEL.T == 4:
            z_patch_arr = [0.0, 0.0, 0.0, 0.0]
            z_amask_arr = [0.0, 0.0, 0.0, 0.0]
            template = [0.0, 0.0, 0.0, 0.0]


            z_patch_arr[0], resize_factor, z_amask_arr[0] = sample_target(image[0], info['init_bbox'], self.params.template_factor,
                                                        output_sz=self.params.template_size)
            z_patch_arr[1], resize_factor, z_amask_arr[1] = sample_target(image[1], info['init_bbox'], self.params.template_factor,
                                                        output_sz=self.params.template_size)
            z_patch_arr[2], resize_factor, z_amask_arr[2] = sample_target(image[2], info['init_bbox'], self.params.template_factor,
                                                        output_sz=self.params.template_size)
            z_patch_arr[3], resize_factor, z_amask_arr[3] = sample_target(image[3], info['init_bbox'], self.params.template_factor,
                                                        output_sz=self.params.template_size)

            self.z_patch_arr = z_patch_arr[0]

            template[0] = self.preprocessor.process(z_patch_arr[0], z_amask_arr[0])
            template[1] = self.preprocessor.process(z_patch_arr[1], z_amask_arr[1])
            template[2] = self.preprocessor.process(z_patch_arr[2], z_amask_arr[2])
            template[3] = self.preprocessor.process(z_patch_arr[3], z_amask_arr[3])

            tensor = torch.cat([template[0].tensors.unsqueeze(0), template[1].tensors.unsqueeze(0), template[2].tensors.unsqueeze(0), template[3].tensors.unsqueeze(0)], dim=0)
            mask = torch.cat([template[0].mask.unsqueeze(0), template[1].mask.unsqueeze(0), template[2].mask.unsqueeze(0), template[3].mask.unsqueeze(0)], dim=0)

            template = NestedTensor(tensor, mask)

            with torch.no_grad():
                self.z_dict1 = template

            self.box_mask_z = None
            if self.cfg.MODEL.BACKBONE.CE_LOC:
                template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                            template.tensors.device).squeeze(1)
                self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

            # save states
            self.state = info['init_bbox']
            self.frame_id = 0
            if self.save_all_boxes:
                '''save all predicted boxes'''
                all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
                return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        if self.cfg.MODEL.T == 1:
            H, W, _ = image.shape
            self.frame_id += 1

            is_flash = bool(info.get('flash', False)) if info else False

            # SDTracker is independent of Kalman: search region is always
            # taken around its own previous network bbox (`self._sd_state`),
            # whether the current frame is flash or not.
            x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self._sd_state, self.params.search_factor,
                                                                    output_sz=self.params.search_size)  # (x1, y1, w, h)
            search = self.preprocessor.process(x_patch_arr, x_amask_arr)

            with torch.no_grad():
                x_dict = search
                out_dict = self.network.forward(
                    template=self.z_dict1.tensors, search=x_dict.tensors)

            pred_score_map = out_dict['score_map']
            response = pred_score_map

            pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
            pred_boxes = pred_boxes.view(-1, 4)
            pred_box = (pred_boxes.mean(
                dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
            # Inline map-back relative to `self._sd_state` (the actual search-crop center).
            cx_prev = self._sd_state[0] + 0.5 * self._sd_state[2]
            cy_prev = self._sd_state[1] + 0.5 * self._sd_state[3]
            cx, cy, w, h = pred_box
            half_side = 0.5 * self.params.search_size / resize_factor
            cx_real = cx + (cx_prev - half_side)
            cy_real = cy + (cy_prev - half_side)
            net_state = clip_box([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], H, W, margin=10)
            net_cx = net_state[0] + 0.5 * net_state[2]
            net_cy = net_state[1] + 0.5 * net_state[3]

            # Kalman: predict every frame; only update with the network output
            # on regular (non-flash, non-cooldown) frames. Keep the predicted
            # center BEFORE update so we can log the pure trajectory.
            self.kalman.predict()
            kalman_pred_cx = float(self.kalman.x[0])
            kalman_pred_cy = float(self.kalman.x[1])

            if is_flash:
                if self.flash_kalman_count < 1:
                    # First 2 flash frames: write Kalman bbox to the output.
                    if self.frozen_wh is None:
                        self.frozen_wh = (self._sd_state[2], self._sd_state[3])
                    kx, ky = self.kalman.x[0], self.kalman.x[1]
                    fw, fh = self.frozen_wh
                    self.state = clip_box([kx - 0.5 * fw, ky - 0.5 * fh, fw, fh], H, W, margin=10)
                else:
                    # Flash still on but Kalman window exhausted — write the network bbox.
                    self.state = net_state
                self.flash_kalman_count += 1
                # Do NOT update Kalman during flash frames (measurement is unreliable).
            else:
                # No flash — reset the burst counter and resume normal Kalman+network.
                self.flash_kalman_count = 0
                self.frozen_wh = None
                self.kalman.update(net_cx, net_cy)
                self.state = net_state

            # SDTracker's own track is ALWAYS the raw network output — used
            # as the reference for the next frame's search region.
            self._sd_state = net_state

        if self.cfg.MODEL.T == 2:
            T, H, W, _ = image.shape
            self.frame_id += 1
            x_patch_arr = [0.0, 0.0]
            x_amask_arr = [0.0, 0.0]
            search = [0.0, 0.0]

            x_patch_arr[0], resize_factor, x_amask_arr[0] = sample_target(image[0], self.state, self.params.search_factor,
                                                                    output_sz=self.params.search_size)
            x_patch_arr[1], resize_factor, x_amask_arr[1] = sample_target(image[1], self.state, self.params.search_factor,
                                                                    output_sz=self.params.search_size)

            search[0] = self.preprocessor.process(x_patch_arr[0], x_amask_arr[0])
            search[1] = self.preprocessor.process(x_patch_arr[1], x_amask_arr[1])

            tensor = torch.cat([search[0].tensors.unsqueeze(0), search[1].tensors.unsqueeze(0)], dim=0)
            mask = torch.cat([search[0].mask.unsqueeze(0), search[1].mask.unsqueeze(0)], dim=0)

            search = NestedTensor(tensor, mask)

            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            with torch.no_grad():
                x_dict = search
                # merge the template and the search
                # run the transformer
                out_dict = self.network.forward(
                    template=self.z_dict1.tensors, search=x_dict.tensors)

            # ***********************************************************  Center 头 ***************************************************

            # add hann windows
            pred_score_map = out_dict['score_map']

            # response = self.output_window * pred_score_map
            response = pred_score_map


            pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
            pred_boxes = pred_boxes.view(-1, 4)
            # Baseline: Take the mean of all pred boxes as the final result
            pred_box = (pred_boxes.mean(
                dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
            # get the final box result
            self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
            # ***********************************************************  Center 头 ***************************************************
            # ***********************************************************  Corner 头 ***************************************************
            # pred_boxes = out_dict['pred_boxes']
            # pred_boxes = pred_boxes.view(-1, 4)
            # pred_box = (pred_boxes.mean(
            #     dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
            # # get the final box result
            # self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
            # ***********************************************************  Corner 头 ***************************************************

        if self.cfg.MODEL.T == 4:
            T, H, W, _ = image.shape
            self.frame_id += 1

            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
            x_patch_arr = [0.0, 0.0, 0.0, 0.0]
            x_amask_arr = [0.0, 0.0, 0.0, 0.0]
            search = [0.0, 0.0, 0.0, 0.0]

            x_patch_arr[0], resize_factor, x_amask_arr[0] = sample_target(image[0], self.state, self.params.search_factor,
                                                                    output_sz=self.params.search_size)
            x_patch_arr[1], resize_factor, x_amask_arr[1] = sample_target(image[1], self.state, self.params.search_factor,
                                                                    output_sz=self.params.search_size)
            x_patch_arr[2], resize_factor, x_amask_arr[2] = sample_target(image[2], self.state, self.params.search_factor,
                                                                    output_sz=self.params.search_size)
            x_patch_arr[3], resize_factor, x_amask_arr[3] = sample_target(image[3], self.state, self.params.search_factor,
                                                                    output_sz=self.params.search_size)
            search[0] = self.preprocessor.process(x_patch_arr[0], x_amask_arr[0])
            search[1] = self.preprocessor.process(x_patch_arr[1], x_amask_arr[1])
            search[2] = self.preprocessor.process(x_patch_arr[2], x_amask_arr[2])
            search[3] = self.preprocessor.process(x_patch_arr[3], x_amask_arr[3])



            tensor = torch.cat([search[0].tensors.unsqueeze(0), search[1].tensors.unsqueeze(0), search[2].tensors.unsqueeze(0), search[3].tensors.unsqueeze(0)], dim=0)
            mask = torch.cat([search[0].mask.unsqueeze(0), search[1].mask.unsqueeze(0), search[2].mask.unsqueeze(0), search[3].mask.unsqueeze(0)], dim=0)

            search = NestedTensor(tensor, mask)

            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            with torch.no_grad():
                x_dict = search
                # merge the template and the search
                # run the transformer
                out_dict = self.network.forward(
                    template=self.z_dict1.tensors, search=x_dict.tensors)

            # ***********************************************************  Center 头 ***************************************************

            # add hann windows
            pred_score_map = out_dict['score_map']

            # response = self.output_window * pred_score_map
            response = pred_score_map


            pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
            pred_boxes = pred_boxes.view(-1, 4)
            # Baseline: Take the mean of all pred boxes as the final result
            pred_box = (pred_boxes.mean(
                dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
            # get the final box result
            self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
            # ***********************************************************  Center 头 ***************************************************
            # ***********************************************************  Corner 头 ***************************************************
            # pred_boxes = out_dict['pred_boxes']
            # pred_boxes = pred_boxes.view(-1, 4)
            # pred_box = (pred_boxes.mean(
            #     dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
            # # get the final box result
            # self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
            # ***********************************************************  Corner 头 ***************************************************



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
                    "all_boxes": all_boxes_save,
                    "kalman_center": (kalman_pred_cx, kalman_pred_cy)}
        else:
            return {"target_bbox": self.state,
                    "kalman_center": (kalman_pred_cx, kalman_pred_cy)}





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
    return SDTrack
