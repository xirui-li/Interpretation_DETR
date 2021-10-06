import torch
from torch.utils.data import DataLoader
import datasets
from datasets.caltech import Caltech_Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torchvision
from PIL import Image
from matplotlib import cm
import torchvision.transforms as transforms
from util.box_ops import *
from scipy.stats import multivariate_normal

def gt_imshow(img, gt):
    img = 150*(img)     # unnormalize
    npimg = img.numpy()
    img = np.transpose(npimg, (1, 2, 0))
    img = Image.fromarray(np.uint8(img), 'RGB')
    

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.imshow(img)
    ax.axis('off')
    ax.axis('tight')

    boxes = gt['boxes']
    labels = gt['labels']
    image_id = gt['image_id']
    aug_h, aug_w = gt['size'][0], gt['size'][1]

    ax = plt.gca()

    for i in range(boxes.shape[0]):
        box = boxes[i]
        label = labels[i]
        box[0] = box[0] * aug_w
        box[2] = box[2] * aug_w
        box[1] = box[1] * aug_h
        box[3] = box[3] * aug_h

        box = box_cxcywh_to_xyxy(box)
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], color='g', fill=False, linewidth=5)

        ax.add_patch(rect)
    
    print(image_id)
   
    plt.savefig('output/dataset_visual.png')


    targets_att = []


    #import pdb; pdb.set_trace()
        
    target_att_one_image = torch.zeros((npimg.shape[1], npimg.shape[2]))
    
    for i in range(boxes.shape[0]):
        box = boxes[i]
        gaussian_mean = torch.ones(2)
        gaussian_mean[0] = box[0]/npimg.shape[2]
        gaussian_mean[1] = box[1]/npimg.shape[1]
        gaussian_std = torch.eye(2,2)
        gaussian_std[0][0] = box[2] / (npimg.shape[2] * 2)
        gaussian_std[1][1] = box[3] / (npimg.shape[1] * 2)
            
        x = np.linspace(0, 1, npimg.shape[2])
        y = np.linspace(0, 1, npimg.shape[1])
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))
                
        gaussian = multivariate_normal(gaussian_mean, gaussian_std)
        a = gaussian.pdf(pos)

        target_att_one_image = torch.from_numpy(a) + target_att_one_image
    
    print(i)
    print(target_att_one_image.max())
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.imshow(target_att_one_image)
    plt.savefig('output/dataset_visual_1.png')


def collate_fn(batch):
    batch = list(zip(*batch))
    # batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)




def main():

    dataset = Caltech_Dataset('input/caltech', 'train', 'mr')

    trainloader = DataLoader(dataset, batch_size=2, num_workers=1, collate_fn=collate_fn, shuffle=True)

    for images, labels in trainloader:

        gt_imshow(images[0], labels[0])
        print('End')

if __name__ == '__main__':
    main()