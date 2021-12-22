PyTorch implementation of Visualization of Deformable Attention in Deformable DETR based on the following methods:

*DEFORMABLE DETR: DEFORMABLE TRANSFORMERS FOR END-TO-END OBJECT DETECTION [link](https://arxiv.org/pdf/2010.04159.pdf),
*End-to-End Object Detection with Transformers [link](https://arxiv.org/pdf/2005.12872.pdf),
*Visualization demo [link](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb)

This repository is an official implementation of the thesis: Visualization of Deformable Attention in Deformable DETR.


## Introduction

**TL; DR.** Deformable DETR is an efficient and fast-converging end-to-end object detector. It mitigates the high complexity and slow convergence issues of DETR via a novel sampling-based efficient attention mechanism.  

**Abstract.** Deformable DEtection TRansformer (Deformable DETR) algorithm, which utilizes a transformer structure and deformable attention mechanisms, achieves high performance in object detection tasks. 
In certain scenarios with high safety standards, such as autonomous driving, it is crucial to examine the decision-making procedure of detection algorithms.
This thesis works on the interpretability of Deformable DETR algorithm through visualizing deformable attention mechanisms, sampling procedures that weight selected points from image feature spaces.
Firstly, Deformable DETR is a more intuitive detection approach to humans because of the similarity between the deformable attention mechanism and human visual mechanism.
Then this thesis investigates the observability of sampling methods at a module level, a layer level, and an inner-layer level in terms of network architecture.
This kind of observability demonstrates panoramic interpretability from macro to micro.
In addition, the observability of multi-head and multi-scale mechanisms also enriches the interpretability, which enables task-oriented interventions with regard to the human-in-the-loop concept for better task results.
Through a large number of experiments, this thesis demonstrates that Deformable DETR is an interpretable approach, which has the potential for explainable artificial intelligence.


## License

This project is released under the [Apache 2.0 license](./LICENSE).

## Installation

### Requirements

* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n deformable_detr python=3.7 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate deformable_detr
    ```
  
* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/))

    For example, if your CUDA version is 9.2, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

### Compiling CUDA operators
```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

## Usage

### Dataset preparation

Please download [COCO 2017 dataset](https://cocodataset.org/) and organize them as following:

```
code_root/
└── data/
    └── coco/
        ├── train2017/
        ├── val2017/
        └── annotations/
        	├── instances_train2017.json
        	└── instances_val2017.json
```

### Training

#### Training on single node

For example, the command for training Deformable DETR on 8 GPUs is as following:

```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/r50_deformable_detr.sh
```

#### Training on multiple nodes

For example, the command for training Deformable DETR on 2 nodes of each with 8 GPUs is as following:

On node 1:

```bash
MASTER_ADDR=<IP address of node 1> NODE_RANK=0 GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 16 ./configs/r50_deformable_detr.sh
```

On node 2:

```bash
MASTER_ADDR=<IP address of node 1> NODE_RANK=1 GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 16 ./configs/r50_deformable_detr.sh
```

#### Training on slurm cluster

If you are using slurm cluster, you can simply run the following command to train on 1 node with 8 GPUs:

```bash
GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh <partition> deformable_detr 8 configs/r50_deformable_detr.sh
```

Or 2 nodes of  each with 8 GPUs:

```bash
GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh <partition> deformable_detr 16 configs/r50_deformable_detr.sh
```
#### Some tips to speed-up training
* If your file system is slow to read images, you may consider enabling '--cache_mode' option to load whole dataset into memory at the beginning of training.
* You may increase the batch size to maximize the GPU utilization, according to GPU memory of yours, e.g., set '--batch_size 3' or '--batch_size 4'.

### Evaluation

You can get the config file and pretrained model of Deformable DETR (the link is in "Main Results" session), then run following command to evaluate it on COCO 2017 validation set:

```bash
<path to config file> --resume <path to pre-trained model> --eval
```

You can also run distributed evaluation by using ```./tools/run_dist_launch.sh``` or ```./tools/run_dist_slurm.sh```.

### Visualization

### Visualization of images

You can plot input image from the dataset with its bounding boxes and labels by running commands:

```bash
python main_training_data_vis.py
```

### Visualization of evolvement of AP

You can plot a loss file of models' output to visualize loss or AP evolvement in epochs by running commands:

```bash
python main_plot.py
```


### Visualization of deformable attention

You can plot sampling strategy of a query (encoder, decoder, head, level, layer) by running commands:

```bash
python main_visual.py
```