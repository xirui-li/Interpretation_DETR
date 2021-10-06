from copy import deepcopy
from typing import Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils import data
import datasets.transforms_caltech as T
# from software.datasets.preprocess import preprocess


class BaseDataset(data.Dataset):
    """
    This dataset class serves as a base class for any dataset that should be fed
    into the SSD model. You simply need to implement the following functions to
    load the dataset in a format that is compatible with our training, inference
    and evaluation scripts:

    Unimplemented methods:
        - _get_dataset_root_path
        - _get_class_names
        - _load_image_ids
        - _load_image_sizes
        - _load_image_paths
        - _load_annotations
    """

    def __init__(self, image_set, mode, eval_mode) -> None:
        super().__init__()
        self._mode = mode
        self._eval_mode = eval_mode
        self._image_set = image_set
        # self.augmentation = augmentation
        # self.class_names = self._get_class_names()
        # self._root_path = self._get_dataset_root_path()
        self.image_ids = self._load_image_ids()
        self._image_paths = self._load_image_paths()
        # self.img_widths, self.img_heights = self._load_image_sizes()
        self.targets = self._load_annotations()
        self.transforms = self._make_caltech_transforms(mode)


    def _get_image_set(self):
        """
        Retrieves the string name of the current image set.
        """
        raise NotImplementedError

    def _get_dataset_root_path(self) -> str:
        """
        Returns the path to the root folder for this dataset. For PascalVOC this
        would be the path to the VOCdevkit folder.
        """
        raise NotImplementedError

    def _get_class_names(self) -> List[str]:
        """
        Returns the list of class names for the given dataset.
        """
        raise NotImplementedError

    def _load_image_ids(self) -> List:
        """
        Returns a list of strings with image ids that are part of the dataset.
        The image ids usually indicated the image file name.
        """
        raise NotImplementedError

    def _load_image_sizes(self) -> Tuple[List[int], List[int]]:
        """
        Retrieves the width and height of all images in the given dataset and
        returns two lists with all integer widths and all integer heights.
        """
        raise NotImplementedError

    def _load_image_paths(self) -> List[str]:
        """
        Returns a list of file paths for each of the images.
        """
        raise NotImplementedError

    def _load_annotations(self) -> Tuple[List, List]:
        """
        Loads the categories for each detection for each image and loads the
        bounding box information for each detection for each image.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """
        Returns the number of images within this dataset.
        """
        return len(self.image_ids)

    def __getitem__(self, index):
        """
        Loads an image (input) and its class and coordinates (targets) based on
        the index within the list of image ids. That means that the image
        information that will be returned belongs to the image at given index.

        :param index: The index of the image within the list of image ids that
                you want to get the information for.

        :return: A quadruple of an image, its class, its coordinates and the
                path to the image itself.
        """
        image_path = self._image_paths[index]
        image = Image.open(image_path).convert('RGB')

        targets = deepcopy(self.targets[index])
        image,targets= self.transforms(image,targets)
        # Preprocesses the given image and its coordinates based on the mode

        # images, classes, coordinates = preprocess(image, classes, coordinates,
        #                                           self._config,
        #                                           self.augmentation)
        return image, targets

    def _make_caltech_transforms(self, image_set):
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

        if image_set == 'train':
            # return normalize
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333),
                    ])
                ),
                normalize,
            ])

        if image_set == 'val':
            return normalize

        raise ValueError(f'unknown {image_set}')
    # Responsible for batching and padding inputs
    @staticmethod
    def collate_fn(samples_in_batch):
        """
        Helps to create a real batch of inputs and targets. The function
        receives a list of single samples and combines the inputs as well as the
        targets to single lists.

        :param samples_in_batch: A list of quadruples of an image, its class,
            its coordinates and file path.

        :return: A batch of samples.
        """
        images = [sample[0] for sample in samples_in_batch]
        paths = [sample[3] for sample in samples_in_batch]
        max_classes = max([sample[1].size()[0] for sample in samples_in_batch])

        classes = [BaseDataset.add_classes_padding(sample[1], max_classes) for
                   sample in samples_in_batch]
        coords = [BaseDataset.add_coords_padding(sample[2], max_classes) for
                  sample in samples_in_batch]

        images = torch.stack(images, dim=0)
        classes = torch.stack(classes, dim=0)
        coords = torch.stack(coords, dim=0)

        return images, classes, coords, paths

    @staticmethod
    def add_classes_padding(tensor, required_size):
        """
        Adds a zero padding to the classes tensor for each image. We need to do
        this to use DataParallel. Right now DataParallel requires the network to
        receive Tensors. Since each image has a different amount of detections
        we need to convert the different sizes to one size.

        :param tensor: The tensor that should be padded.
        :param required_size: The number of detections this tensor
            needs tohave.

        :return: The padded tensor.
        """
        target = torch.zeros([required_size], dtype=torch.long)
        target[:tensor.size()[0]] = tensor
        return target

    @staticmethod
    def add_coords_padding(tensor, required_size):
        """
        Adds a zero padding to the coords tensor for each image. We need to do
        this to use DataParallel. Right now DataParallel requires the network to
        receive Tensors. Since each image has a different amount of detections
        we need to convert the different sizes to one size.

        :param tensor: The coordinates tensor that should be padded.
        :param required_size: The number of detections this tensor
            needs to have.

        :return: The padded tensor.
        """
        target = torch.zeros(required_size, 4, dtype=torch.float)
        target[:tensor.size()[0], :] = tensor
        return target

    def remove_classes(self, remove):
        """
        Removes the classes wih the given names from this dataset. This function
        will adjust the classes and coordinates tensors of the dataset. If an
        image has zero detections after removing a specific class from the
        dataset, the image will be removed from the dataset.

        :param remove: A list of class names that should be removed from
            the dataset.
        """
        remove_classes = [self.class_names.index(x) for x in remove]
        keep = [x for x in self.class_names if x not in remove]
        reindex = {self.class_names.index(x): i for i, x in enumerate(keep)}
        self.class_names = keep
        for index in reversed(range(self.__len__())):
            class_array = self.classes[index]
            delete_mask = np.isin(class_array, remove_classes)
            class_array = np.array(class_array)[~delete_mask]
            if class_array.size == 0:
                del self.classes[index]
                del self.coordinates[index]
                del self._image_paths[index]
                del self.image_ids[index]
                del self.img_widths[index]
                del self.img_heights[index]
            else:
                class_array = [reindex[id] for id in class_array]
                self.classes[index] = class_array
                coords_array = self.coordinates[index][~delete_mask]
                self.coordinates[index] = coords_array

    def keep_classes(self, keep_names):
        """
        Removes all classes from this dataset except for the class with the
        given names.

        :param keep_names: A list of class names that should be kept.
        """
        keep_names.append("background")
        remove_names = [x for x in self.class_names if x not in keep_names]
        self.remove_classes(remove_names)

    def merge_classes(self, merge_dict):
        """
        Takes in a dictionary with a key class name and a list of class names as
        value that should be merged together. If the key is "car" and the value
        is ["truck", "van"] the result of this function will be that
        there is no more class truck and van but the classes have been merged to
        only car including the car class.

        **NOTE:** Please make sure that the key is not part of the list and
        that no class is mentioned twice.

        :param merge_dict: The dictionary which allows you to specify the merge
            instructions.
        """
        reindex = {}
        remove = [x for sublist in merge_dict.values() for x in sublist]
        keep = [x for x in self.class_names if x not in remove]

        for i, name in enumerate(self.class_names):
            if name in remove:
                new_index = self.find_parent_merger(name, merge_dict, keep)
            else:
                new_index = keep.index(name)
            reindex[i] = new_index

        for index in range(self.__len__()):
            class_array = self.classes[index]
            class_array = np.array([reindex[id] for id in class_array])
            self.classes[index] = class_array

        self.class_names = keep

    def find_parent_merger(self, name, merge_dict, keep):
        """
        Returns the index of key within the dictionary whose value contains the
        name.
        """
        for i, x in enumerate(merge_dict.values()):
            if name in x:
                return keep.index(list(merge_dict.keys())[i])
        return 0

    def rename_class(self, old_name, new_name):
        """
        Renames the class with the given old name to the given new name.
        """
        if old_name not in self.class_names:
            raise AttributeError("The old name does not exist.")
        if new_name in self.class_names:
            raise AttributeError("The new name already exists.")
        self.class_names = [new_name if old_name == name else name for name in self.class_names ]