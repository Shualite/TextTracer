#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
# encoding: utf-8
import copy
import os
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path
import subprocess
import re
import editdistance
import json
from PIL import Image
import cv2
import math
import mmcv
import numpy as np
import torch
from mmcv.image.misc import tensor2imgs
from mmcv.runner import load_checkpoint
from mmcv.utils.config import Config
import pickle
from tqdm import tqdm
from os.path import join as pjoin

from mmocr.apis import init_detector
from mmocr.apis.inference import model_inference
from mmocr.core.visualize import det_recog_show_result
from mmocr.datasets.kie_dataset import KIEDataset
from mmocr.datasets.pipelines.crop import crop_img
from mmocr.models import build_detector
from mmocr.utils.box_util import stitch_boxes_into_lines
from mmocr.utils.fileio import list_from_file
from mmocr.utils.model import revert_sync_batchnorm

from paddleocr import PaddleOCR, draw_ocr

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--video', type=str, help='Input video file or folder path.')
    parser.add_argument(
        '--txt', type=str, help='Input data url file')
    parser.add_argument(
        '--video_path',
        type=str,
        default='',
        help='Output file/folder name for visualization')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='',
        required=True,
        help='Output file path')
    parser.add_argument(
        '--framework',
        choices=['mmocr', 'paddleocr', 'easyocr'],
        type=str,
        default='paddleocr',
        help='mmocr, paddleocr or easyocr')
    parser.add_argument(
        '--output',
        type=str,
        default='',
        help='Output file/folder name for visualization')
    parser.add_argument(
        '--det',
        type=str,
        default='PANet_IC15',
        help='Pretrained text detection algorithm')
    parser.add_argument(
        '--det-config',
        type=str,
        default='',
        help='Path to the custom config file of the selected det model. It '
        'overrides the settings in det')
    parser.add_argument(
        '--det-ckpt',
        type=str,
        default='',
        help='Path to the custom checkpoint file of the selected det model. '
        'It overrides the settings in det')
    parser.add_argument(
        '--recog',
        type=str,
        default='SEG',
        help='Pretrained text recognition algorithm')
    parser.add_argument(
        '--recog-config',
        type=str,
        default='',
        help='Path to the custom config file of the selected recog model. It'
        'overrides the settings in recog')
    parser.add_argument(
        '--recog-ckpt',
        type=str,
        default='',
        help='Path to the custom checkpoint file of the selected recog model. '
        'It overrides the settings in recog')
    parser.add_argument(
        '--kie',
        type=str,
        default='',
        help='Pretrained key information extraction algorithm')
    parser.add_argument(
        '--kie-config',
        type=str,
        default='',
        help='Path to the custom config file of the selected kie model. It'
        'overrides the settings in kie')
    parser.add_argument(
        '--kie-ckpt',
        type=str,
        default='',
        help='Path to the custom checkpoint file of the selected kie model. '
        'It overrides the settings in kie')
    parser.add_argument(
        '--config-dir',
        type=str,
        default=os.path.join(str(Path.cwd()), 'configs/'),
        help='Path to the config directory where all the config files '
        'are located. Defaults to "configs/"')
    parser.add_argument(
        '--batch-mode',
        action='store_true',
        help='Whether use batch mode for inference')
    parser.add_argument(
        '--recog-batch-size',
        type=int,
        default=0,
        help='Batch size for text recognition')
    parser.add_argument(
        '--det-batch-size',
        type=int,
        default=0,
        help='Batch size for text detection')
    parser.add_argument(
        '--single-batch-size',
        type=int,
        default=0,
        help='Batch size for separate det/recog inference')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference.')
    parser.add_argument(
        '--export',
        type=str,
        default='',
        help='Folder where the results of each image are exported')
    parser.add_argument(
        '--export-format',
        type=str,
        default='json',
        help='Format of the exported result file(s)')
    parser.add_argument(
        '--details',
        action='store_true',
        help='Whether include the text boxes coordinates and confidence values'
    )
    parser.add_argument(
        '--imshow',
        action='store_true',
        help='Whether show image with OpenCV.')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Prints the recognised text')
    parser.add_argument(
        '--merge', action='store_true', help='Merge neighboring boxes')
    parser.add_argument(
        '--merge-xdist',
        type=float,
        default=20,
        help='The maximum x-axis distance to merge boxes')
    args = parser.parse_args()
    if args.det == 'None':
        args.det = None
    if args.recog == 'None':
        args.recog = None
    # Warnings
    if args.merge and not (args.det and args.recog):
        warnings.warn(
            'Box merging will not work if the script is not'
            ' running in detection + recognition mode.', UserWarning)
    if not os.path.samefile(args.config_dir, os.path.join(str(
            Path.cwd()))) and (args.det_config != ''
                               or args.recog_config != ''):
        warnings.warn(
            'config_dir will be overridden by det-config or recog-config.',
            UserWarning)
    return args


def is_contain_chinese(check_str):
    """Check whether string contains Chinese or not.

    Args:
        check_str (str): String to be checked.

    Return True if contains Chinese, else False.
    """
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

def if_need_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        return True
    return False

def file_filter(f):
    if f[-4:] in ['.jpg', '.png', '.bmp']:
        return True
    else:
        return False
    
def file_filter2(f):
    if f[-4:] in ['.mp4', '.avi']:
        return True
    else:
        return False
    
def calc_dis(a, b):
    return math.sqrt(pow(a[0]-b[0], 2) + pow(a[1]-b[1], 2))

def text_normalizer(normalized_str):
    """Unify Chinese and English punctuation marks, filter special characters.

    Args:
        normalized_str (str): String to be normalized.

    Return str after normalized.
    """
    a = re.findall('[\u4e00-\u9fa5a-zA-Z0-9]+', normalized_str, re.S) #只要字符串中的中文，字母，数字
    a = "".join(a)
    return a

def keep_chinese(str1):
    return ''.join(re.findall(r'[\u4e00-\u9fa5]', str1))

class MMOCR:

    def __init__(self,
                 det='PANet_IC15',
                 det_config='',
                 det_ckpt='',
                 recog='SEG',
                 recog_config='',
                 recog_ckpt='',
                 kie='',
                 kie_config='',
                 kie_ckpt='',
                 config_dir=os.path.join(str(Path.cwd()), 'configs/'),
                 device='cuda:0',
                 **kwargs):

        textdet_models = {
            'DB_r18': {
                'config':
                'dbnet/dbnet_r18_fpnc_1200e_icdar2015.py',
                'ckpt':
                'dbnet/'
                'dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth'
            },
            'DB_r50': {
                'config':
                'dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py',
                'ckpt':
                'dbnet/'
                'dbnet_r50dcnv2_fpnc_sbn_1200e_icdar2015_20211025-9fe3b590.pth'
            },
            'DRRG': {
                'config':
                'drrg/drrg_r50_fpn_unet_1200e_ctw1500.py',
                'ckpt':
                'drrg/drrg_r50_fpn_unet_1200e_ctw1500_20211022-fb30b001.pth'
            },
            'FCE_IC15': {
                'config':
                'fcenet/fcenet_r50_fpn_1500e_icdar2015.py',
                'ckpt':
                'fcenet/fcenet_r50_fpn_1500e_icdar2015_20211022-daefb6ed.pth'
            },
            'FCE_CTW_DCNv2': {
                'config':
                'fcenet/fcenet_r50dcnv2_fpn_1500e_ctw1500.py',
                'ckpt':
                'fcenet/' +
                'fcenet_r50dcnv2_fpn_1500e_ctw1500_20211022-e326d7ec.pth'
            },
            'MaskRCNN_CTW': {
                'config':
                'maskrcnn/mask_rcnn_r50_fpn_160e_ctw1500.py',
                'ckpt':
                'maskrcnn/'
                'mask_rcnn_r50_fpn_160e_ctw1500_20210219-96497a76.pth'
            },
            'MaskRCNN_IC15': {
                'config':
                'maskrcnn/mask_rcnn_r50_fpn_160e_icdar2015.py',
                'ckpt':
                'maskrcnn/'
                'mask_rcnn_r50_fpn_160e_icdar2015_20210219-8eb340a3.pth'
            },
            'MaskRCNN_IC17': {
                'config':
                'maskrcnn/mask_rcnn_r50_fpn_160e_icdar2017.py',
                'ckpt':
                'maskrcnn/'
                'mask_rcnn_r50_fpn_160e_icdar2017_20210218-c6ec3ebb.pth'
            },
            'PANet_CTW': {
                'config':
                'panet/panet_r18_fpem_ffm_600e_ctw1500.py',
                'ckpt':
                'panet/'
                'panet_r18_fpem_ffm_sbn_600e_ctw1500_20210219-3b3a9aa3.pth'
            },
            'PANet_IC15': {
                'config':
                'panet/panet_r18_fpem_ffm_600e_icdar2015.py',
                'ckpt':
                'panet/'
                'panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth'
            },
            'PS_CTW': {
                'config': 'psenet/psenet_r50_fpnf_600e_ctw1500.py',
                'ckpt':
                'psenet/psenet_r50_fpnf_600e_ctw1500_20210401-216fed50.pth'
            },
            'PS_IC15': {
                'config':
                'psenet/psenet_r50_fpnf_600e_icdar2015.py',
                'ckpt':
                'psenet/psenet_r50_fpnf_600e_icdar2015_pretrain-eefd8fe6.pth'
            },
            'TextSnake': {
                'config':
                'textsnake/textsnake_r50_fpn_unet_1200e_ctw1500.py',
                'ckpt':
                'textsnake/textsnake_r50_fpn_unet_1200e_ctw1500-27f65b64.pth'
            }
        }

        textrecog_models = {
            'CRNN': {
                'config': 'crnn/crnn_academic_dataset.py',
                'ckpt': 'crnn/crnn_academic-a723a1c5.pth'
            },
            'SAR': {
                'config': 'sar/sar_r31_parallel_decoder_academic.py',
                'ckpt': 'sar/sar_r31_parallel_decoder_academic-dba3a4a3.pth'
            },
            'SAR_CN': {
                'config':
                'sar/sar_r31_parallel_decoder_chinese.py',
                'ckpt':
                'sar/sar_r31_parallel_decoder_chineseocr_20210507-b4be8214.pth'
            },
            'NRTR_1/16-1/8': {
                'config': 'nrtr/nrtr_r31_1by16_1by8_academic.py',
                'ckpt':
                'nrtr/nrtr_r31_1by16_1by8_academic_20211124-f60cebf4.pth'
            },
            'NRTR_1/8-1/4': {
                'config': 'nrtr/nrtr_r31_1by8_1by4_academic.py',
                'ckpt':
                'nrtr/nrtr_r31_1by8_1by4_academic_20211123-e1fdb322.pth'
            },
            'RobustScanner': {
                'config': 'robust_scanner/robustscanner_r31_academic.py',
                'ckpt': 'robustscanner/robustscanner_r31_academic-5f05874f.pth'
            },
            'SATRN': {
                'config': 'satrn/satrn_academic.py',
                'ckpt': 'satrn/satrn_academic_20211009-cb8b1580.pth'
            },
            'SATRN_sm': {
                'config': 'satrn/satrn_small.py',
                'ckpt': 'satrn/satrn_small_20211009-2cf13355.pth'
            },
            'ABINet': {
                'config': 'abinet/abinet_academic.py',
                'ckpt': 'abinet/abinet_academic-f718abf6.pth'
            },
            'SEG': {
                'config': 'seg/seg_r31_1by16_fpnocr_academic.py',
                'ckpt': 'seg/seg_r31_1by16_fpnocr_academic-72235b11.pth'
            },
            'CRNN_TPS': {
                'config': 'tps/crnn_tps_academic_dataset.py',
                'ckpt': 'tps/crnn_tps_academic_dataset_20210510-d221a905.pth'
            }
        }

        kie_models = {
            'SDMGR': {
                'config': 'sdmgr/sdmgr_unet16_60e_wildreceipt.py',
                'ckpt':
                'sdmgr/sdmgr_unet16_60e_wildreceipt_20210520-7489e6de.pth'
            }
        }

        self.td = det
        self.tr = recog
        self.kie = kie
        self.device = device

        # Check if the det/recog model choice is valid
        if self.td and self.td not in textdet_models:
            raise ValueError(self.td,
                             'is not a supported text detection algorthm')
        elif self.tr and self.tr not in textrecog_models:
            raise ValueError(self.tr,
                             'is not a supported text recognition algorithm')
        elif self.kie:
            if self.kie not in kie_models:
                raise ValueError(
                    self.kie, 'is not a supported key information extraction'
                    ' algorithm')
            elif not (self.td and self.tr):
                raise NotImplementedError(
                    self.kie, 'has to run together'
                    ' with text detection and recognition algorithms.')

        self.detect_model = None
        if self.td:
            # Build detection model
            if not det_config:
                det_config = os.path.join(config_dir, 'textdet/',
                                          textdet_models[self.td]['config'])
            if not det_ckpt:
                det_ckpt = 'https://download.openmmlab.com/mmocr/textdet/' + \
                    textdet_models[self.td]['ckpt']

            self.detect_model = init_detector(
                det_config, det_ckpt, device=self.device)
            self.detect_model = revert_sync_batchnorm(self.detect_model)

        self.recog_model = None
        if self.tr:
            # Build recognition model
            if not recog_config:
                recog_config = os.path.join(
                    config_dir, 'textrecog/',
                    textrecog_models[self.tr]['config'])
            if not recog_ckpt:
                recog_ckpt = 'https://download.openmmlab.com/mmocr/' + \
                    'textrecog/' + textrecog_models[self.tr]['ckpt']

            self.recog_model = init_detector(
                recog_config, recog_ckpt, device=self.device)
            self.recog_model = revert_sync_batchnorm(self.recog_model)

        self.kie_model = None
        if self.kie:
            # Build key information extraction model
            if not kie_config:
                kie_config = os.path.join(config_dir, 'kie/',
                                          kie_models[self.kie]['config'])
            if not kie_ckpt:
                kie_ckpt = 'https://download.openmmlab.com/mmocr/' + \
                    'kie/' + kie_models[self.kie]['ckpt']

            kie_cfg = Config.fromfile(kie_config)
            self.kie_model = build_detector(
                kie_cfg.model, test_cfg=kie_cfg.get('test_cfg'))
            self.kie_model = revert_sync_batchnorm(self.kie_model)
            self.kie_model.cfg = kie_cfg
            load_checkpoint(self.kie_model, kie_ckpt, map_location=self.device)

        # Attribute check
        for model in list(filter(None, [self.recog_model, self.detect_model])):
            if hasattr(model, 'module'):
                model = model.module
            if model.cfg.data.test['type'] == 'ConcatDataset':
                model.cfg.data.test.pipeline = \
                    model.cfg.data.test['datasets'][0].pipeline

    def readtext(self,
                 img,
                 output=None,
                 details=False,
                 export=None,
                 export_format='json',
                 batch_mode=False,
                 recog_batch_size=0,
                 det_batch_size=0,
                 single_batch_size=0,
                 imshow=False,
                 print_result=False,
                 merge=False,
                 merge_xdist=20,
                 **kwargs):        
        args = locals().copy()
        [args.pop(x, None) for x in ['kwargs', 'self']]
        args = Namespace(**args)

        # Input and output arguments processing
        self._args_processing(args)
        self.args = args

        pp_result = None

        # Send args and models to the MMOCR model inference API
        # and call post-processing functions for the output
        if self.detect_model and self.recog_model:
            det_recog_result = self.det_recog_kie_inference(
                self.detect_model, self.recog_model, kie_model=self.kie_model)
            pp_result = self.det_recog_pp(det_recog_result)
        else:
            for model in list(
                    filter(None, [self.recog_model, self.detect_model])):
                result = self.single_inference(model, args.arrays,
                                               args.batch_mode,
                                               args.single_batch_size)
                pp_result = self.single_pp(result, model)

        return pp_result

    # Post processing function for end2end ocr
    def det_recog_pp(self, result):
        final_results = []
        args = self.args
        for arr, output, export, det_recog_result in zip(
                args.arrays, args.output, args.export, result):
            if output or args.imshow:
                if self.kie_model:
                    res_img = det_recog_show_result(arr, det_recog_result)
                else:
                    res_img = det_recog_show_result(
                        arr, det_recog_result, out_file=output)
                if args.imshow and not self.kie_model:
                    mmcv.imshow(res_img, 'inference results')
            # if not args.details:
            #     simple_res = {}
            #     simple_res['filename'] = det_recog_result['filename']
            #     simple_res['text'] = [
            #         x['text'] for x in det_recog_result['result']
            #     ]
            #     final_result = simple_res
            # else:
            #     final_result = det_recog_result
            final_result = det_recog_result
            final_result['img_shape'] = arr.shape[:2]
            if export:
                mmcv.dump(final_result, export, indent=4)
            if args.print_result:
                print(final_result, end='\n\n')
            final_results.append(final_result)
        return final_results

    # Post processing function for separate det/recog inference
    def single_pp(self, result, model):
        for arr, output, export, res in zip(self.args.arrays, self.args.output,
                                            self.args.export, result):
            if export:
                mmcv.dump(res, export, indent=4)
            if output or self.args.imshow:
                res_img = model.show_result(arr, res, out_file=output)
                if self.args.imshow:
                    mmcv.imshow(res_img, 'inference results')
            if self.args.print_result:
                print(res, end='\n\n')
        return result

    def generate_kie_labels(self, result, boxes, class_list):
        idx_to_cls = {}
        if class_list is not None:
            for line in list_from_file(class_list):
                class_idx, class_label = line.strip().split()
                idx_to_cls[class_idx] = class_label

        max_value, max_idx = torch.max(result['nodes'].detach().cpu(), -1)
        node_pred_label = max_idx.numpy().tolist()
        node_pred_score = max_value.numpy().tolist()
        labels = []
        for i in range(len(boxes)):
            pred_label = str(node_pred_label[i])
            if pred_label in idx_to_cls:
                pred_label = idx_to_cls[pred_label]
            pred_score = node_pred_score[i]
            labels.append((pred_label, pred_score))
        return labels

    def visualize_kie_output(self,
                             model,
                             data,
                             result,
                             out_file=None,
                             show=False):
        """Visualizes KIE output."""
        img_tensor = data['img'].data
        img_meta = data['img_metas'].data
        gt_bboxes = data['gt_bboxes'].data.numpy().tolist()
        if img_tensor.dtype == torch.uint8:
            # The img tensor is the raw input not being normalized
            # (For SDMGR non-visual)
            img = img_tensor.cpu().numpy().transpose(1, 2, 0)
        else:
            img = tensor2imgs(
                img_tensor.unsqueeze(0), **img_meta.get('img_norm_cfg', {}))[0]
        h, w, _ = img_meta.get('img_shape', img.shape)
        img_show = img[:h, :w, :]
        model.show_result(
            img_show, result, gt_bboxes, show=show, out_file=out_file)

    # End2end ocr inference pipeline
    def det_recog_kie_inference(self, det_model, recog_model, kie_model=None):
        end2end_res = []
        # Find bounding boxes in the images (text detection)
        det_result = self.single_inference(det_model, self.args.arrays,
                                           self.args.batch_mode,
                                           self.args.det_batch_size)
        bboxes_list = [res['boundary_result'] for res in det_result]

        if kie_model:
            kie_dataset = KIEDataset(
                dict_file=kie_model.cfg.data.test.dict_file)

        # For each bounding box, the image is cropped and
        # sent to the recognition model either one by one
        # or all together depending on the batch_mode
        for filename, arr, bboxes, out_file in zip(self.args.filenames,
                                                   self.args.arrays,
                                                   bboxes_list,
                                                   self.args.output):
            img_e2e_res = {}
            img_e2e_res['filename'] = filename
            img_e2e_res['result'] = []
            box_imgs = []
            for bbox in bboxes:
                box_res = {}
                box_res['box'] = [round(x) for x in bbox[:-1]]
                box_res['box_score'] = float(bbox[-1])
                box = bbox[:8]
                if len(bbox) > 9:
                    min_x = min(bbox[0:-1:2])
                    min_y = min(bbox[1:-1:2])
                    max_x = max(bbox[0:-1:2])
                    max_y = max(bbox[1:-1:2])
                    box = [
                        min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y
                    ]
                box_img = crop_img(arr, box)
                if self.args.batch_mode:
                    box_imgs.append(box_img)
                else:
                    recog_result = model_inference(recog_model, box_img)
                    text = recog_result['text']
                    text_score = recog_result['score']
                    if isinstance(text_score, list):
                        text_score = sum(text_score) / max(1, len(text))
                    box_res['text'] = text
                    box_res['text_score'] = text_score
                img_e2e_res['result'].append(box_res)

            if self.args.batch_mode:
                recog_results = self.single_inference(
                    recog_model, box_imgs, True, self.args.recog_batch_size)
                for i, recog_result in enumerate(recog_results):
                    text = recog_result['text']
                    text_score = recog_result['score']
                    if isinstance(text_score, (list, tuple)):
                        text_score = sum(text_score) / max(1, len(text))
                    img_e2e_res['result'][i]['text'] = text
                    img_e2e_res['result'][i]['text_score'] = text_score

            if self.args.merge:
                img_e2e_res['result'] = stitch_boxes_into_lines(
                    img_e2e_res['result'], self.args.merge_xdist, 0.5)

            if kie_model:
                annotations = copy.deepcopy(img_e2e_res['result'])
                # Customized for kie_dataset, which
                # assumes that boxes are represented by only 4 points
                for i, ann in enumerate(annotations):
                    min_x = min(ann['box'][::2])
                    min_y = min(ann['box'][1::2])
                    max_x = max(ann['box'][::2])
                    max_y = max(ann['box'][1::2])
                    annotations[i]['box'] = [
                        min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y
                    ]
                ann_info = kie_dataset._parse_anno_info(annotations)
                ann_info['ori_bboxes'] = ann_info.get('ori_bboxes',
                                                      ann_info['bboxes'])
                ann_info['gt_bboxes'] = ann_info.get('gt_bboxes',
                                                     ann_info['bboxes'])
                kie_result, data = model_inference(
                    kie_model,
                    arr,
                    ann=ann_info,
                    return_data=True,
                    batch_mode=self.args.batch_mode)
                # visualize KIE results
                self.visualize_kie_output(
                    kie_model,
                    data,
                    kie_result,
                    out_file=out_file,
                    show=self.args.imshow)
                gt_bboxes = data['gt_bboxes'].data.numpy().tolist()
                labels = self.generate_kie_labels(kie_result, gt_bboxes,
                                                  kie_model.class_list)
                for i in range(len(gt_bboxes)):
                    img_e2e_res['result'][i]['label'] = labels[i][0]
                    img_e2e_res['result'][i]['label_score'] = labels[i][1]

            end2end_res.append(img_e2e_res)
        return end2end_res

    # Separate det/recog inference pipeline
    def single_inference(self, model, arrays, batch_mode, batch_size=0):
        result = []
        if batch_mode:
            if batch_size == 0:
                result = model_inference(model, arrays, batch_mode=True)
            else:
                n = batch_size
                arr_chunks = [
                    arrays[i:i + n] for i in range(0, len(arrays), n)
                ]
                for chunk in arr_chunks:
                    result.extend(
                        model_inference(model, chunk, batch_mode=True))
        else:
            for arr in arrays:
                result.append(model_inference(model, arr, batch_mode=False))
        return result

    # Arguments pre-processing function
    def _args_processing(self, args):
        # Check if the input is a list/tuple that
        # contains only np arrays or strings
        if isinstance(args.img, (list, tuple)):
            img_list = args.img
            if not all([isinstance(x, (np.ndarray, str)) for x in args.img]):
                raise AssertionError('Images must be strings or numpy arrays')

        # Create a list of the images
        if isinstance(args.img, str):
            img_path = Path(args.img)
            if img_path.is_dir():
                img_list = [str(x) for x in img_path.glob('*')]
            else:
                img_list = [str(img_path)]
        elif isinstance(args.img, np.ndarray):
            img_list = [args.img]

        # Read all image(s) in advance to reduce wasted time
        # re-reading the images for visualization output
        args.arrays = [mmcv.imread(x) for x in img_list]

        # Create a list of filenames (used for output images and result files)
        if isinstance(img_list[0], str):
            args.filenames = [str(Path(x).stem) for x in img_list]
        else:
            args.filenames = [str(x) for x in range(len(img_list))]

        # If given an output argument, create a list of output image filenames
        num_res = len(img_list)
        if args.output:
            output_path = Path(args.output)
            if output_path.is_dir():
                args.output = [
                    str(output_path / f'out_{x}.png') for x in args.filenames
                ]
            else:
                args.output = [str(args.output)]
                if args.batch_mode:
                    raise AssertionError('Output of multiple images inference'
                                         ' must be a directory')
        else:
            args.output = [None] * num_res

        # If given an export argument, create a list of
        # result filenames for each image
        if args.export:
            export_path = Path(args.export)
            args.export = [
                str(export_path / f'out_{x}.{args.export_format}')
                for x in args.filenames
            ]
        else:
            args.export = [None] * num_res

        return args

class VideoOCR:
    def __init__(self, **kwargs):
        # import ipdb;ipdb.set_trace()
        if kwargs['framework']=='mmocr':
            ocrmodel = MMOCR(**kwargs)
        elif kwargs['framework']=='paddleocr':
            ocrmodel = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
        else:
            raise NotImplementedError
        
        self.ocrmodel = ocrmodel
        
        self.final_results = []
        
        # 帧间IOU阈值，用于bbox绑定
        self.frames_iou_thresh = 0.6
        # 用于并行计算IOU
        self.buffer_boxes_pre = []
        self.buffer_boxes_cur = []
        # 用于索引前后帧的信息条目
        self.buffer_results_pre = []
        self.buffer_results_cur = []
        # 文本的编辑距离，容错百分比
        self.ed_error_rate = 0.5
        
        self.bbox_mask = True
        # 位于屏幕底端或顶端
        self.topbottom_mask_ratio = [0.05, 0.95]
        # 用于四个角上的区域过滤掉logo，[H, W]
        self.icon_mask_ratio = [0.1, 0.1]
        # bbox高度
        self.hight_thresh = 0.02
        # bbox宽高比
        self.w_h_ratio = 1
        
        # 将若干绑定的bbox表示为一个集合，否则将聚合为最大外接矩形
        self.box_verbose = True
        
        
        self.text_count_filter = -1
        self.key_text_score = 0.7
        self.normal_text_score = 0.8
        
        self.box_score_recall = 0.96
        self.box_score_pre_thresh = 0.8
        self.cluster_rate_H = 0.1
        self.frame_num = -1
        self.frame_preposition_rate = 0.8
        
    def __clean_buffer(self):
        self.buffer_boxes_pre = []
        self.buffer_boxes_cur = []
        self.buffer_results_pre = []
        self.buffer_results_cur = []
        self.final_results = []
    
    
    def info_extract(self, formated_info_file=None):
        prepos_min = max(5, self.frame_preposition_rate*self.frame_num)
        
        prepos_compressed_info = []
        compressed_info = []
        for item in self.final_results:
            line = {}
            
            box_idx = np.argsort(item['box_score'])[::-1]
            sorted_boxes = np.array(item['box'])[box_idx]
            text_idx = np.argsort(item['text_score'])[::-1]
            sorted_texts, sorted_texts_score = np.array(item['text'])[text_idx], np.array(item['text_score'])[text_idx]
            sorted_frames_by_text = np.array(item['frame_id'])[text_idx]
            sorted_frames_by_first = sorted(item['frame_id'], key=lambda x: int(x))
            
            # if sorted_texts_score[0]<self.text_score_thresh:
            #     continue
            if not is_contain_chinese(''.join(sorted_texts[:3])):
                continue
                
            # line['selected_box'] = sorted_boxes[0].tolist()
            line['box_center'] = [int(i) for i in cv2.minAreaRect(np.reshape(sorted_boxes[0], (4,2)))[0]]
            line['selected_text'] = sorted_texts[0]
            line['selected_text_score'] = str(sorted_texts_score[0])
            line['frame_by_text'] = int(sorted_frames_by_first[0])
            line['text_count'] = len(sorted_texts)
            
            if line['text_count']>=self.text_count_filter:            
                if line['text_count']>prepos_min and float(line['selected_text_score'])>=self.key_text_score:
                    prepos_compressed_info.append(line)
                elif line['text_count']<=prepos_min and float(line['selected_text_score'])>=self.normal_text_score:
                    compressed_info.append(line)
                else:
                    pass
        results_info = {}
        
        prepos_compressed_info = sorted(prepos_compressed_info, key=lambda y: y['frame_by_text'])
        prepos_compressed_info = sorted(prepos_compressed_info, key=lambda y: y['text_count'])[::-1]
        prepos_compressed_info = sorted(prepos_compressed_info, key=lambda y: y['box_center'][1])
        results_info['title'] = prepos_compressed_info
        results_info['content'] = {}

        compressed_info = sorted(compressed_info, key=lambda y: y['box_center'][1])
        compressed_info = sorted(compressed_info, key=lambda y: y['frame_by_text'])
        cluster_thresh = self.cluster_rate_H * self.img_shape[0]
        box_info = [t['box_center'] for t in compressed_info]
        box_selected = [0] * len(box_info)
        
        
        cluster_id = 0
        cluster_results = {}
        while sum(box_selected) != len(box_info):
            for idx, v in enumerate(box_selected):
                if v==0:
                    break
            cur_center = box_info[idx]
            cluster_results[cluster_id] = [idx]
            box_selected[idx] = 1
            for idx, v in enumerate(box_selected):
                if v==0 and calc_dis(box_info[idx], cur_center)<=cluster_thresh:
                    cluster_results[cluster_id].append(idx)
                    # center smmoth shift
                    cur_center = [int(0.7*cur_center[0]+0.3*box_info[idx][0]), int(0.7*cur_center[1]+0.3*box_info[idx][1])]
                    box_selected[idx] = 1
                    
            cluster_id+=1
        
        for k, v in cluster_results.items():
            cluster = np.array(compressed_info)[v].tolist()
            cluster = sorted(cluster, key=lambda y: y['frame_by_text'])
            results_info['content'][k] = cluster
                
        result_dict = self.remove_redundancy(results_info)
        
        with open(formated_info_file, 'w', encoding='utf-8') as f_obj:
            json.dump(result_dict, f_obj, indent=4, ensure_ascii=False)
        
        # # pickle file            
        # with open(formated_info_file, 'wb') as f:
        #     pickle.dump(result_dict, f)
        
        return result_dict, results_info
    
    def remove_redundancy(self, results_info):
        
        result_dict = {}
        
        title_info = results_info['title']
        title_line = '\n'.join([t['selected_text'] for t in title_info])
        result_dict['title'] = title_line
        result_dict['content'] = []
        
        for k, v in results_info['content'].items():
            texts, scores = [], []
            for item in v:
                texts.append(item['selected_text'].strip())
                scores.append(float(item['selected_text_score']))
            
            N = len(texts)
            deleted_idx = [0]*N
            assert N == len(scores)
            for i in range(N):
                if deleted_idx[i]==0:
                    for j in range(i+1, N):
                        tmpa, tmpb = keep_chinese(texts[i]), keep_chinese(texts[j])
                        if editdistance.eval(tmpa, tmpb) <= min(3, (len(tmpa)+len(tmpb))//6):
                            tmp = [1] * N
                            tmp[i], tmp[j] = scores[i], scores[j]
                            min_idx = np.argsort(tmp)
                            deleted_idx[min_idx[0]] = 1

            tmp = []
            for i, text in enumerate(texts):
                if deleted_idx[i]==0:
                    tmp.append(text.strip())
            item_line = '\n'.join(tmp)
            result_dict['content'].append(item_line)
        
        return result_dict
    
        
    def readtext_from_video(self, video, save_dir, **kwargs):
        self.__clean_buffer()
        
        # import ipdb;ipdb.set_trace()
        # video_path = os.path.abspath(os.path.dirname(video))
        json_path = save_dir
        video_name, postfix = os.path.basename(video).split('.')[0], os.path.basename(video).split('.')[1]
        # output_path = os.path.join(json_path, 'frames_'+video_name)
        
        source_path = '/mnt/storage01/fengshuyang/data/BOVText_mine/data/Test/images/Cls21_Vlog/{}/frames/'.format(video_name)
        
        formated_info_file = os.path.join(json_path, '{}.json'.format(video_name))
        
        
        if not os.path.exists(source_path):
            print(source_path + ' NOT FOUND!')
            return []
        # if os.path.exists(saved_info_file):
        #     print('json: '+saved_info_file+'. has been gotten')
        #     return []
        
        
        # if if_need_mkdir(output_path):
        #     cmd = 'ffmpeg -i {video_path} -r {frame_num} -q:v 2 -f image2 {out_dir}/%06d.jpg'
        #     subprocess.run(cmd.format(video_path=video, frame_num=1, out_dir=output_path), encoding="utf-8" , shell=True)
        # else:
        #     print('video: '+video+'. has been excuted')
        #     if os.path.exists(saved_info_file):
        #         print('json: '+saved_info_file+'. has been gotten')
        #         return []
        
        logging.debug('video_id: '+video_name +'\n')
        print('video_id: '+video_name +'\n')
        self.frame_num = len(list(filter(file_filter, os.listdir(source_path))))
        
        import ipdb;ipdb.set_trace()
        for frame_id, file in enumerate(sorted(list(filter(file_filter, os.listdir(source_path))), key=lambda x: int(x.split('.')[0]))):
            img_path = os.path.join(source_path, file)
            logging.debug("#"*10 + " "*2 + "frame_id_"+str(frame_id+1) + " "*2 + "#"*10)
            
            image = Image.open(img_path).convert('RGB')
            img_shape = image.size[:2][::-1]
            self.img_shape = img_shape
            
            if isinstance(self.ocrmodel, MMOCR):
                # 
                #     {'filename':xx, 'result':[{'box', 'box_score', 'text', 'text_score'}], 'img_shape':xx}
                # 
                # import ipdb;ipdb.set_trace()
                result = self.ocrmodel.readtext(img_path, **kwargs)[0]
            elif isinstance(self.ocrmodel, PaddleOCR):
                # [[[[x1,y1],...,[x4,y4]], (text, text_score)]]
                result = {}
                paddleocr_result = self.ocrmodel.ocr(img_path, cls=True)
                # logging.debug(np.array(paddleocr_result))
                
                filename = os.path.basename(img_path).split('.')[0]
                result['filename'] = filename
                result['result'] = []
                for pr in paddleocr_result:
                    temp = {}
                    box, text, text_score = np.array(pr[0], dtype=np.int).reshape(-1).tolist(), pr[1][0], pr[1][1]
                    temp['box'], temp['box_score'], temp['text'], temp['text_score'] = box, 1.0, text, text_score
                    result['result'].append(temp)
                result['img_shape'] = img_shape
            else:
                raise NotImplementedError
            self.det_recog_bind(result)
            
            
            # 将buffer_results_cur 相比 同步 将buffer_results_pre 新增的内容同步到 final_results
            new_items = []
            for i in self.buffer_results_cur:
                if i not in self.buffer_results_pre:
                    new_items.append(i)
                    
                    
            self.final_results.extend(new_items)
            # 完成一帧的匹配，交替缓存区
            self.buffer_boxes_pre = np.array(self.buffer_boxes_cur)
            self.buffer_results_pre = np.array(self.buffer_results_cur)
            self.buffer_boxes_cur = []            
            self.buffer_results_cur = []
            
            assert len(self.buffer_boxes_pre) == len(self.buffer_results_pre)
            logging.debug('Current frame bind results:')
            logging.debug(self.buffer_results_pre)
            logging.debug(self.buffer_boxes_pre)
            
        
        logging.debug("Final logging - self.final_results:\n {} \n".format(np.array(self.final_results)))
        
        # import ipdb;ipdb.set_trace()
        line, results_info = self.info_extract(formated_info_file)
        
        logging.debug("After strategy execute: "+str(line))
        logging.debug('\n\n')
        return line
        
    def det_recog_bind(self, cur_frame_result):
        import ipdb;ipdb.set_trace()
        frame_id = cur_frame_result['filename']
        
        img_shape = cur_frame_result['img_shape']
        H, W = img_shape

        for box_id, item in enumerate(cur_frame_result['result']):
            item['frame_id'] = [frame_id]
            item['box'] = [item['box']]
            item['box_score'] = [item['box_score']]
            item['text'] = [item['text']]
            item['text_score'] = [item['text_score']]

            # logging.debug("---------frame:{}, box_id:{}--------".format(frame_id, box_id+1))
            # logging.debug("item info: {}".format(item))

            # step 1. 过滤掉位于屏幕底端且高较小bbox
            query_box, query_box_score = np.array(item['box'][0]), item['box_score'][0]
            img_h, img_w = img_shape
            # self.box_score_pre_thresh
            query_left, query_right, query_top, query_bottom = min(query_box[::2]), max(query_box[::2]), min(query_box[1::2]), max(query_box[1::2])
            wh_ratio = float(query_right-query_left)/(query_bottom-query_top)            
            if (query_bottom-query_top) <= self.hight_thresh*img_h or query_box_score<self.box_score_pre_thresh or wh_ratio<=self.w_h_ratio:
                # logging.debug('@@ droped by bbox height or score illegal !')
                continue
            
            if self.bbox_mask:
                top = np.ceil(H*self.topbottom_mask_ratio[0])
                bottom = np.ceil(H*self.topbottom_mask_ratio[-1])
                if min(query_box[1::2])<top or max(query_box[1::2])>bottom:
                    # logging.debug('@@ droped by bbox top or bottom position illegal !')
                    continue
                lt_x, lt_y = W*self.icon_mask_ratio[-1], H*self.icon_mask_ratio[0]
                rt_x, rt_y = W*(1-self.icon_mask_ratio[-1]), H*self.icon_mask_ratio[0]
                lb_x, lb_y = W*self.icon_mask_ratio[-1], H*(1-self.icon_mask_ratio[0])
                rb_x, rb_y = W*(1-self.icon_mask_ratio[-1]), H*(1-self.icon_mask_ratio[0])
                query_lt, query_rt, query_lb, query_rb = (min(query_box[::2]),min(query_box[1::2])),\
                                                        (max(query_box[::2]),min(query_box[1::2])),\
                                                        (min(query_box[::2]),max(query_box[1::2])),\
                                                        (max(query_box[::2]),max(query_box[1::2]))
                
                if (query_lt[0]<lt_x and query_lt[1]<lt_y) or (query_rt[0]>rt_x and query_rt[1]<rt_y) \
                or (query_lb[0]<lb_x and query_lb[1]>lb_y) or (query_rb[0]>rb_x and query_rb[1]>rb_y):
                    # logging.debug('@@ droped by bbox icon mask position illegal !')
                    continue
            
            # step 2. 确保有效字符长度以及内容合规
            query_text = item['text'][0]
            normalized_text = text_normalizer(query_text)
            if len(normalized_text) > 1 and is_contain_chinese(normalized_text):
                # step 3. 计算和前一帧的匹配情况，分为两种情况：1）bbox匹配上，则进行内容的比较选择最合适的merge进去，或者都不合适单独插入一项。2）没有匹配上, 如果是一个新出现的文本内容则应当满足box_score较高且与前帧IOU较小
                candidate_idx = self.candidatefinder_for_idx(query_box)
                if not bool(candidate_idx.size):
                    if query_box_score >= self.box_score_recall:
                        self.buffer_results_cur.append(item)
                        self.buffer_boxes_cur.append(item['box'][0])
                        # logging.debug('@@ inserted not match previous bbox!')
                    else:
                        # logging.debug('@@ droped by not match and box_score low!')
                        pass
                    continue


                # TODO: 当存在多个candidate的时候  应该依次比较编辑距离，选择IOU和编辑距离综合最小的一项插入，而不是每一项都插入或者融合，会造成大量重复
                candidate_eds = []
                for idx in candidate_idx.tolist():
                    temp_item = self.buffer_results_pre[idx]

                    # step 3. compare text edit distance
                    optimal_text_idx = np.argmax(temp_item['text_score'])
                    temp_text = temp_item['text'][optimal_text_idx]
                    temp_ed = editdistance.eval(text_normalizer(temp_text), normalized_text)
                    candidate_eds.append(temp_ed)

                best_ed_idx = np.argsort(candidate_eds)[0]
                text_ed = candidate_eds[best_ed_idx]

                # step 4. merge or create
                max_ed = max(1, int(len(normalized_text)*self.ed_error_rate))
                if text_ed <= max(1, max_ed):
                    # match & merge item into canditate_item (be equivalent to add item into self.buffer_results_cur)
                    candidate_item = self.buffer_results_pre[candidate_idx[best_ed_idx]]
                    candidate_item['box'].append(item['box'][0])
                    candidate_item['box_score'].append(item['box_score'][0])
                    candidate_item['text'].append(item['text'][0])
                    candidate_item['text_score'].append(item['text_score'][0])
                    candidate_item['frame_id'].append(item['frame_id'][0])

                    self.buffer_results_cur.append(candidate_item)
                    self.buffer_boxes_cur.append(candidate_item['box'][0])
                    # logging.debug('@@ merged!')
                else:
                    # not match & insert item into self.buffer_results_cur as a new one

                    # logging.debug('** filter rule 3: ed:{} > {}; thus, add new one: {};'.format(text_ed, max_ed, item))
                    # insert item into self.buffer_results_cur and self.buffer_boxes_cur
                    self.buffer_results_cur.append(item)
                    self.buffer_boxes_cur.append(item['box'][0])
                    # logging.debug('@@ inserted after ed!')

            else:
                # logging.debug('@@ droped by text content illegal !')
                pass
                        
                
    def candidatefinder_for_idx(self, query_box):
        if len(self.buffer_boxes_pre) == 0:
            return np.array([], dtype=np.int64)
        
        query_left, query_right, query_top, query_bottom = min(query_box[::2]), max(query_box[::2]), min(query_box[1::2]), max(query_box[1::2])
        # 获取前一帧的bbox信息
        left, right, top, bottom = np.min(self.buffer_boxes_pre[:, ::2], axis=1), np.max(self.buffer_boxes_pre[:, ::2], axis=1), np.min(self.buffer_boxes_pre[:, 1::2], axis=1), np.max(self.buffer_boxes_pre[:, 1::2], axis=1)
        
        query_h, query_w = (query_bottom-query_top), (query_right-query_left)
        query_area = query_h * query_w
        
        areas = (right-left)*(bottom-top)
        
        union_lt_x, union_lt_y, union_rb_x, union_rb_y = np.maximum(query_left, left), np.maximum(query_top, top), np.minimum(query_right, right), np.minimum(query_bottom, bottom)
        union_w, union_h = np.maximum(0, union_rb_x-union_lt_x), np.maximum(0, union_rb_y-union_lt_y)
        union_areas = union_w * union_h
        # normal iou
        ious = union_areas/(query_area+areas-union_areas)
        # single iou
        # iou_v1 = union_areas/np.maximum(query_area, union_areas)
        
        idx = np.where(ious>self.frames_iou_thresh)[0]
        
        return idx
        

def main():
    args = parse_args()
    ocr_handle = VideoOCR(**vars(args))
    
    videos_path = '/mnt/storage01/fengshuyang/data/BOVText_chendi/data/Test/Video/Cls21_Vlog/'
    for vname in list(filter(file_filter2, os.listdir(videos_path))):
        v_path = pjoin(videos_path, vname)
        args.video = v_path
        results = ocr_handle.readtext_from_video(**vars(args))
    
    # if args.video:
    #     results = ocr_handle.readtext_from_video(**vars(args))
    # if args.txt:
    #     f = open(args.txt,'r')
    #     next(f)
    #     for idx, item in tqdm(enumerate(f)):
    #         video_id = item.strip()
    #         v_file = '/data/home/v_shualfeng/datasets/douyin_share4/video/{}.mp4'.format(video_id)
    #         args.video = v_file

    #         try:
                
    #             results = ocr_handle.readtext_from_video(**vars(args))
                
    #         except:
    #             pass
            
    #     f.close()

if __name__ == '__main__':
    import logging
    import datetime
    ts = datetime.datetime.now()
    log_file = "./logs/log_{}.{}_{}:{},{}s.log".format(str(ts.month).zfill(2), str(ts.day).zfill(2), str(ts.hour).zfill(2), str(ts.minute).zfill(2), str(ts.second).zfill(2))
    logging.basicConfig(filename = log_file, level = logging.DEBUG) 
    main()