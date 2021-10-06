import os
import sys

import matplotlib
import cv2
import json
import numpy as np
from skimage import io,data
from typing import Iterable
import torch
from torch import device, nn
import torchvision.transforms as TT 
import datasets.transforms as T
import math
from PIL import Image
from matplotlib import cm
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#%config InlineBackend.figure_format = 'retina'

import ipywidgets as widgets
from IPython.display import display, clear_output

from torchvision.models import resnet50

from util.misc import nested_tensor_from_tensor_list


# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
        [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = TT.Compose([
    TT.Resize(800),
    TT.ToTensor(),
    TT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()


@torch.no_grad()
class AttentionVisualizer:
    def __init__(self, model):
        self.model = model
        self.transform =  TT.Compose([
                                    TT.Resize(800),
                                    TT.ToTensor(),
                                    TT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
        self.path = ''
        self.cur_url = None
        self.pil_img = None
        self.tensor_img = None

        self.conv_features = None
        self.enc_attn_weights = None
        self.dec_attn_weights = None

        self.dec_cross_attn_weights = None
        self.dec_cross_attn_sampling_offsets = None
        self.dec_cross_attn_attention_weights = None
        self.dec_query_5 = None
        self.transformer_output_hs = None
        self.transformer_output_init_reference = None
        self.transformer_output_inter_references = None
        self.model_output = None

        self.device = torch.device('cuda')

    def compute_features(self, img):
        model = self.model
        # use lists to store the outputs via up-values
        conv_features, enc_attn_weights, dec_self_attn_weights, dec_cross_attn_weights, dec_cross_attn_sampling_offsets, dec_cross_attn_attention_weights, dec_query_5 = [], [], [], [], [], [], []

        input_feature, transformer_output, model_output = [], [], []

        hooks = [
            model.backbone[-2].register_forward_hook(
                lambda self, input, output: conv_features.append(output)),
            model.transformer.register_forward_hook(
                lambda self, input, output: transformer_output.append(output)),
            model.register_forward_hook(
                lambda self, input, output: model_output.append(output)),
            model.transformer.encoder.layers[0].self_attn.register_forward_hook(
                lambda self, input, output: input_feature.append(input[0])),
            model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[0])),
            model.transformer.decoder.layers[-1].self_attn.register_forward_hook(
                lambda self, input, output: dec_self_attn_weights.append(output[0])),
            model.transformer.decoder.layers[-1].cross_attn.register_forward_hook(
                lambda self, input, output: dec_cross_attn_weights.append(output[0])),
            model.transformer.decoder.layers[-1].cross_attn.sampling_offsets.register_forward_hook(
                lambda self, input, output: dec_cross_attn_sampling_offsets.append(output[0])),
            model.transformer.decoder.layers[-1].cross_attn.attention_weights.register_forward_hook(
                lambda self, input, output: dec_cross_attn_attention_weights.append(output[0])),
            model.transformer.decoder.layers[-1].cross_attn.register_forward_hook(
                lambda self, input, output: dec_query_5.append(input[0])),
        ]
        # propagate through the model
        img = img.to(self.device)
        outputs = model(img)

        for hook in hooks:
            hook.remove()

        # don't need the list anymore
        self.conv_features = conv_features[0]
        self.enc_attn_weights = enc_attn_weights[0]
        self.dec_self_attn_weights = dec_self_attn_weights[0]
        self.dec_cross_attn_weights = dec_cross_attn_weights[0]
        self.dec_cross_attn_sampling_offsets = dec_cross_attn_sampling_offsets[0]
        self.dec_cross_attn_attention_weights = dec_cross_attn_attention_weights[0]
        self.dec_query_5 = dec_query_5[0]
        self.transformer_output_hs, self.transformer_output_init_reference, self.transformer_output_inter_references, _ , _ = transformer_output[0]
        self.model_output = model_output[0]
    
    def compute_on_image(self, path):
        if path != self.path:
            self.path = path
            self.pil_img = Image.open(path)
            # mean-std normalize the input image (batch-size: 1)
            self.tensor_img = self.transform(self.pil_img).unsqueeze(0)
            self.compute_features(self.tensor_img)
    
    def features_visualizaion(self):
        
        enc_attn_weights_1, enc_attn_weights_2, enc_attn_weights_3, enc_attn_weights_4 = torch.split(self.enc_attn_weights,
                                            [100*134, 50*67, 25*34, 13*17], dim=0)
        conv_features = self.conv_features 
        enc_attn_weights = self.enc_attn_weights
        dec_self_attn_weights = self.dec_self_attn_weights
        dec_cross_attn_weights = self.dec_cross_attn_weights
        dec_cross_attn_sampling_offsets =self.dec_cross_attn_sampling_offsets
        dec_cross_attn_attention_weights =self.dec_cross_attn_attention_weights
        dec_query_5 = self.dec_query_5
    
    def reference_point_position_visualization(self, path):
        transformer_output_init_reference = self.transformer_output_init_reference.cpu().detach().numpy()[0]
        
        self.pil_img = Image.open(path)

        dpi = 80
        scale = 2

        width, height = self.pil_img.width, self.pil_img.height

        figsize = scale * width / float(dpi), scale * height / float(dpi)

        M_rescale = np.ones((2,2))
        M_rescale[0][0], M_rescale[1][1] = width, height

        transformer_output_init_reference_rescaled = np.matmul(transformer_output_init_reference, M_rescale)

        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.imshow(self.pil_img)
        ax.axis('off')
        ax.axis('tight')

        ax = plt.gca()
        plt.scatter(transformer_output_init_reference_rescaled[:, 0], transformer_output_init_reference_rescaled[:, 1], s = 20, c ='white')

        plt.savefig('output/visualization_reference_point/reference_point_initial.jpg')

        plt.close()
        
        
        #interposition
        transformer_output_inter_references = self.transformer_output_inter_references.cpu().detach().numpy()
    
        for i in range(transformer_output_inter_references.shape[0]):

            transformer_output_inter_reference = transformer_output_inter_references[i]

            self.pil_img = Image.open(path)

            dpi = 80
            scale = 2

            width, height = self.pil_img.width, self.pil_img.height

            figsize = scale * width / float(dpi), scale * height / float(dpi)

            M_rescale = np.ones((2,2))
            M_rescale[0][0], M_rescale[1][1] = width, height

            transformer_output_inter_reference_rescaled = np.matmul(transformer_output_inter_reference[0, : ,:], M_rescale)

            fig, ax = plt.subplots(figsize=figsize)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.imshow(self.pil_img)
            ax.axis('off')
            ax.axis('tight')

            ax = plt.gca()
            plt.scatter(transformer_output_inter_reference_rescaled[:, 0], transformer_output_inter_reference_rescaled[:, 1], s = 20, c ='white')

            file_name = 'reference_point_' + str(i)

            plt.savefig('output/visualization_reference_point/' + file_name)

            plt.close()
        

    def query_position_visualization(self, path):

        query_position_end = self.model_output['pred_boxes']
        query_position_end = query_position_end.cpu().detach().numpy()[0, :, 0:2]

        query_class_end = torch.sigmoid(self.model_output['pred_logits'])
        query_class_end = query_class_end.cpu().detach().numpy()[0]

        query_class_end_index = np.argmax(query_class_end, axis=1)

        query_end_size = np.ones((300, 1))

        for i in range(query_class_end.shape[1]):
            query_end_size[i] = query_class_end[i][query_class_end_index[i]]
        
        viridis = cm.get_cmap('rainbow', 91)

        #color map
        class_color = np.linspace(1, 91, 91).reshape((1, 91))
        #import pdb;pdb.set_trace()

        fig, ax = plt.subplots(1, 1, figsize=(20, 2))

        psm = ax.pcolormesh(class_color, cmap=viridis, rasterized=True, vmin=1, vmax=91)
        fig.colorbar(psm, ax=ax)

        ax = plt.gca()
        plt.savefig('output/visualization_query_size/colormap.jpg')
        plt.close()
        
        #the last object query
        self.pil_img = Image.open(path)

        dpi = 80
        scale = 2

        width, height = self.pil_img.width, self.pil_img.height

        figsize = scale * width / float(dpi), scale * height / float(dpi)

        M_rescale = np.ones((2,2))
        M_rescale[0][0], M_rescale[1][1] = width, height

        query_position_rescaled = np.matmul(query_position_end, M_rescale)

        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.imshow(self.pil_img)
        ax.axis('off')
        ax.axis('tight')

        ax = plt.gca()
        plt.scatter(query_position_rescaled[:, 0], query_position_rescaled[:, 1], s = 50 * query_end_size, c =viridis(query_class_end_index))
        #a = plt.scatter(query_position_rescaled[:, 0], query_position_rescaled[:, 1], s = 20, c =viridis(query_class_end_index))
        #fig.colorbar(a, ax=ax)
        print(query_class_end_index)
        plt.savefig('output/visualization_query_size/object_query_6.jpg')

        plt.close()
        
        #interposition
        query_inter = self.model_output['aux_outputs']

        for i in range(len(query_inter)):

            query_position_inter_ith = query_inter[i]['pred_boxes']
            query_position_inter_ith = query_position_inter_ith.cpu().detach().numpy()[0, :, 0:2]

            query_class_inter_ith = query_inter[i]['pred_logits']
            query_class_inter_ith = query_class_inter_ith.cpu().detach().numpy()[0]

            query_class_inter_ith_index = np.argmax(query_class_inter_ith, axis=1)

            query_inter_ith_size = np.ones((300, 1))

            for j in range(query_class_inter_ith.shape[1]):
                query_inter_ith_size[j] = query_class_inter_ith[j][query_class_inter_ith_index[j]]
            

            self.pil_img = Image.open(path)

            dpi = 80
            scale = 2

            width, height = self.pil_img.width, self.pil_img.height

            figsize = scale * width / float(dpi), scale * height / float(dpi)

            M_rescale = np.ones((2,2))
            M_rescale[0][0], M_rescale[1][1] = width, height

            query_position_inter_ith_rescaled = np.matmul(query_position_inter_ith, M_rescale)

            fig, ax = plt.subplots(figsize=figsize)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.imshow(self.pil_img)
            ax.axis('off')
            ax.axis('tight')

            ax = plt.gca()
            plt.scatter(query_position_inter_ith_rescaled[:, 0], query_position_inter_ith_rescaled[:, 1], s = 50 * query_inter_ith_size, c =viridis(query_class_inter_ith_index))

            file_name = 'object_query_' + str(i)

            plt.savefig('output/visualization_query_size/' + file_name)

            plt.close()


    def reference_query_delta_visualization(self, path):

        query_position_end = self.model_output['pred_boxes']
        query_position_end = query_position_end.cpu().detach().numpy()[0, :, 0:2]

        query_class_end = torch.sigmoid(self.model_output['pred_logits'])
        query_class_end = query_class_end.cpu().detach().numpy()[0]

        query_class_end_index = np.argmax(query_class_end, axis=1)

        viridis = cm.get_cmap('rainbow', 91)

        transformer_output_init_reference = self.transformer_output_init_reference.cpu().detach().numpy()[0]
        
        #the last object query
        self.pil_img = Image.open(path)

        dpi = 80
        scale = 2

        width, height = self.pil_img.width, self.pil_img.height

        figsize = scale * width / float(dpi), scale * height / float(dpi)

        M_rescale = np.ones((2,2))
        M_rescale[0][0], M_rescale[1][1] = width, height

        query_position_rescaled = np.matmul(query_position_end, M_rescale)
        transformer_output_init_reference_rescaled = np.matmul(transformer_output_init_reference, M_rescale)

        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.imshow(self.pil_img)
        ax.axis('off')
        ax.axis('tight')

        ax = plt.gca()
        for i in range(query_position_rescaled.shape[0]):
            plt.arrow(transformer_output_init_reference_rescaled[i, 0], transformer_output_init_reference_rescaled[i, 1], 
                query_position_rescaled[i, 0] - transformer_output_init_reference_rescaled[i, 0], query_position_rescaled[i, 1] - transformer_output_init_reference_rescaled[i, 1],
                width = 1, color=viridis(query_class_end_index[i]))
        #a = plt.scatter(query_position_rescaled[:, 0], query_position_rescaled[:, 1], s = 20, c =viridis(query_class_end_index))
        #fig.colorbar(a, ax=ax)

        plt.savefig('output/visualization_delta_reference_query/reference_point_query_prediction_delta.jpg')

        plt.close()
        

    def generate_prediction_header(self):
        
        checkpoint=torch.load('output/pretrained/r50_deformable_detr-checkpoint.pth')
        a_key=checkpoint['model'].keys()
        header_dict = {k:vfir}
        model.load_state_dict(checkpoint['model_state_dict'])

        import pdb; pdb.set_trace()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)


    def run(self, path):

        self.compute_on_image(path)
        #self.generate_prediction_header()
        #self.features_visualizaion()
        #self.reference_point_position_visualization(path)
        #self.query_position_visualization(path)
        self.reference_query_delta_visualization(path)












@torch.no_grad()
class output_visualizer:
    def __init__(self, model, postprocessors, img_path, ann_path, img_id, dataset_file, threshold):
        self.model = model
        self.threshold = threshold
        self.img_path = img_path #path = '../deformable_detr_mainstream/input/caltech/set00/V007/images/I00559.jpg'
        self.ann_path = ann_path
        self.img_id = img_id
        self.pil_img = None
        self.ann = None
        self.attn = None
        self.tensor_img = None
        self.device = torch.device('cuda')
        self.dataset = dataset_file


    def compute_on_image(self, im):
        # mean-std normalize the input image (batch-size: 1)
        img = transform(im).unsqueeze(0)
        img = img.to(self.device)

        convert_tensor = TT.ToTensor()

        size = im.size

        model = self.model
        # propagate through the model
        outputs = model(img)

        # keep only predictions with 0.7+ confidence
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1].cpu()
        keep = probas.max(-1).values > self.threshold
        # convert boxes from [0; 1] to image scales
        
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep].cpu(), size)
        return outputs, probas, bboxes_scaled, keep

    
    def standard_visualization(self):
        self.pil_img = Image.open(self.img_path)

        COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
        [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
        colors = COLORS * 100

        from pycocotools.coco import COCO
        coco = COCO(self.ann_path)
        ann_ids = coco.getAnnIds(imgIds=self.img_id)
        targets = coco.loadAnns(ann_ids)

        #outputs, attention = self.compute_on_image(self.pil_img)
        outputs, probas, bboxes_scaled, keep = self.compute_on_image(self.pil_img)

        dpi = 80
        scale = 2

        width, height = self.pil_img.width, self.pil_img.height

        figsize = scale * width / float(dpi), scale * height / float(dpi)

        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.imshow(self.pil_img)
        ax.axis('off')
        ax.axis('tight')

        #heatmap = np.uint8(255 * attention)
        #ax.imshow(heatmap, alpha=0.35)

        # Configure axis
        ax = plt.gca()
        label_list = ['background','person','people','person-fa','person?']
        CLASSES = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
        'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush']

        dpi = 80
        scale = 2

        width, height = self.pil_img.width, self.pil_img.height

        figsize = scale * width / float(dpi), scale * height / float(dpi)

        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.imshow(self.pil_img)
        ax.axis('off')
        ax.axis('tight')

        ax = plt.gca()
        
        prob = probas[keep]
        boxes = bboxes_scaled

        for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=3))
            cl = p.argmax()
            text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))

        if targets is not None:
            
            for i in range(len(targets)):
                groundtruth = targets[i]['bbox']

                #groundtruth[0], groundtruth[2] = groundtruth[0]/640, groundtruth[2]
                
                rect = plt.Rectangle((groundtruth[0], groundtruth[1]), groundtruth[2], groundtruth[3],
                                        color='blue',
                                        fill=False, linewidth=3)

                # Add the patch to the Axes
                ax.add_patch(rect)

        if self.dataset == "caltech":
            plt.savefig('output/visualization/test_caltech_9.jpg')
        else:
            plt.savefig('output/visualization/test_coco.jpg')
    
    def attention_plot(self):
        self.pil_img = Image.open(self.img_path)
        save_size = (80, 60)
        outputs, attention = self.compute_on_image(self.pil_img)
        
        #attention = cv2.resize(attention, self.pil_img.size) / attention.max()
        attention = attention / attention.max()
        # plot 2D prob-distribution of attention
        X = np.arange(0, save_size[0])
        Y = np.arange(0, save_size[1])
        
        X, Y = np.meshgrid(X, Y)
        
        self.plt_plot_bivariate_normal_pdf(X, Y, np.flipud(attention))


    def plt_plot_bivariate_normal_pdf(self, x, y, z):
        fig = plt.figure(figsize=(12, 6))
        ax = fig.gca(projection='3d')
        ax.plot_surface(x, y, z,
                    cmap=cm.coolwarm,
                    linewidth=0,
                    antialiased=True)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.savefig('output/visualization/attention_9.jpg')

