# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
from util import box_ops
import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from datasets import get_coco_api_from_dataset
import json
import matplotlib.pyplot as plt

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()
    i = 0
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        loss_scaled_output = ["loss_ce", "loss_bbox", "loss_giou"]
        loss_unscaled_output = ["loss_ce_unscaled", "class_error_unscaled", "loss_bbox_unscaled", "loss_giou_unscaled", "cardinality_error_unscaled"]
        
        loss_dict_reduced_scaled_out = {x: loss_dict_reduced_scaled.get(x) for x in loss_scaled_output}
        loss_dict_reduced_unscaled_out = {x: loss_dict_reduced_unscaled.get(x) for x in loss_unscaled_output}

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled_out, **loss_dict_reduced_unscaled_out)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
    i = 0
    for samples, targets in metric_logger.log_every(data_loader, 100, header):
        if i == 800:
            break
        i += 1
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}

        loss_scaled_output = ["loss_ce", "loss_bbox", "loss_giou"]
        loss_unscaled_output = ["loss_ce_unscaled", "class_error_unscaled", "loss_bbox_unscaled", "loss_giou_unscaled", "cardinality_error_unscaled"]
        
        loss_dict_reduced_scaled_out = {x: loss_dict_reduced_scaled.get(x) for x in loss_scaled_output}
        loss_dict_reduced_unscaled_out = {x: loss_dict_reduced_unscaled.get(x) for x in loss_unscaled_output}
                             
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled_out,
                             **loss_dict_reduced_unscaled_out)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator


@torch.no_grad()
def evaluate_caltech_map(model, criterion, postprocessors, data_loader, device, output_dir_caltech):
    model.eval()
    criterion.eval()

    # base_ds = get_coco_api_from_dataset(train_datasets)


    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())

    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    predictions,groundtruths = [],[]
    i = 0
    for samples, targets in metric_logger.log_every(data_loader, 100, header):
        if i == 1000:
            break
        i += 1
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}

        loss_scaled_output = ["loss_ce", "loss_bbox", "loss_giou"]
        loss_unscaled_output = ["loss_ce_unscaled", "class_error_unscaled", "loss_bbox_unscaled", "loss_giou_unscaled", "cardinality_error_unscaled"]
        
        loss_dict_reduced_scaled_out = {x: loss_dict_reduced_scaled.get(x) for x in loss_scaled_output}
        loss_dict_reduced_unscaled_out = {x: loss_dict_reduced_unscaled.get(x) for x in loss_unscaled_output}
                             
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled_out,
                             **loss_dict_reduced_unscaled_out)                              
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)  

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        for target, result in zip(targets,results):
            predictions.append(result)
            groundtruths.append(target)

    fauxcoco = COCO()
    cocolike_predictions = []
    for image_id, prediction in enumerate(predictions):
        box = box_ops.xyxy_to_xywh(prediction['boxes']).tolist()  # xywh

        score = prediction['scores'].tolist()  # (#objs,)
        label = torch.zeros_like(prediction['labels']).tolist()  # (#objs,)
        # for predcls, we set label and score to groundtruth

        image_id = np.asarray([image_id] * len(box))
        cocolike_predictions.append(
            np.column_stack((image_id, box, score, label))
        )

    # with open("input/data.json",'r') as load_f:
    #     fauxcoco.dataset = json.load(load_f)

    anns = []
    for image_id, gt in enumerate(groundtruths):
        labels = torch.zeros_like(gt['labels']).tolist()  # integer
        orig_size = gt['orig_size']
        boxes = box_ops.box_cxcywh_to_xywh(gt['boxes']*torch.tensor([orig_size[1],orig_size[0],orig_size[1],orig_size[0]],device=device)).tolist()  # xywh
        for cls, box in zip(labels, boxes):
            anns.append({
                'area': box[3] * box[2],
                'bbox': [box[0], box[1], box[2], box[3]],  # xywh
                'category_id': cls,
                'id': len(anns),
                'image_id': image_id,
                'iscrowd': 0,
            })

    fauxcoco.dataset = {
        'info': {'description': 'use coco script for vg detection evaluation'},
        'images': [{'id': i} for i in range(len(groundtruths))],
        'categories': [
            {'supercategory': 'person', 'id': i, 'name': i}
            for i in range(5)
        ],
        'annotations': anns,
    }

    fauxcoco.createIndex()
    cocolike_predictions = np.concatenate(cocolike_predictions, 0)
    # evaluate via coco API
    res = fauxcoco.loadRes(cocolike_predictions)
    coco_evaluator = COCOeval(fauxcoco, res, 'bbox')
    coco_evaluator.params.imgIds = list(range(len(predictions)))
    coco_evaluator.evaluate()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    # if coco_evaluator is not None:
    #     coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.stats.tolist()


    
    return stats, coco_evaluator



@torch.no_grad()
def evaluate_caltech_mr(model, criterion, postprocessors, data_loader, device, output_dir_caltech):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for samples, targets in metric_logger.log_every(data_loader, 100, header):

        samples = samples.to(device)
        #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"].to(device) for t in targets], dim=0)
        
        results = postprocessors['bbox'](outputs, orig_target_sizes)  

        for result, target in zip(results, targets):
            #predictions.append(result)
            #targets.append(target)

            pre_boxes = box_ops.xyxy_to_xywh(result['boxes'])
            image_id = target['image_id']
            scores = result['scores']
            
            image_path_id = image_id.tolist()
            image_number = int(''.join([str(d) for d in str(image_path_id)][-5:]))

            if (image_number + 1) % 30 == 0:
                
                image_path_txt = 'I'+''.join([str(d) for d in str(image_path_id)][-5:])+'.txt'
                image_path_v = 'V'+''.join([str(d) for d in str(image_path_id)][-8:-5])
                image_path_set = ''.join([str(d) for d in str(image_path_id)][:-8])
                if len(image_path_set) == 1:
                    image_path_set = '0' + image_path_set
                image_path_set = 'set' + image_path_set

            
                if output_dir_caltech:
                    if not os.path.exists(os.path.join(output_dir_caltech, image_path_set)):
                        os.mkdir(os.path.join(output_dir_caltech,image_path_set))
                    if not os.path.exists(os.path.join(output_dir_caltech, image_path_set, image_path_v)):
                        os.mkdir(os.path.join(output_dir_caltech,image_path_set, image_path_v))  
                    with open(os.path.join(output_dir_caltech, image_path_set, image_path_v, image_path_txt), "w") as f:
                        values, indices = torch.topk(scores, 10)
                        
                        for i in indices:
                            pre_box = pre_boxes[i]
                            score = scores[i]
                            pre_box = pre_box.tolist()

                            for box in pre_box:
                                f.write(str(box)+",")
                            f.write(str(score.item())+"\n")
                else:
                    print('No valid output path for caltech evaluation.')


    
