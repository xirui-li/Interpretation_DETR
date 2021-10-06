# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate
from models import build_model
from visualization import AttentionVisualizer, output_visualizer
from main import get_args_parser

def main(args):
    device = torch.device(args.device)
    model, _, postprocessors = build_model(args)
    model.to(device)
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print("Start visualization")
    img_path = 'input/coco/val2017/000000039769.jpg'

    ann_path = 'input/coco/annotations/instances_val2017.json'

    img_id = 39769

    data = args.dataset_file

    # visual = output_visualizer(model, postprocessors, img_path, ann_path, img_id, data, threshold=0.5)

    # visual.standard_visualization()

    # visual.attention_plot()

    w = AttentionVisualizer(model)
    w.run(img_path)





if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)