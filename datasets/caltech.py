import json
import os
from typing import Tuple
import torch
from datasets.base_dataset import BaseDataset

import torch.utils.data



class Caltech_Dataset(BaseDataset):

    def __init__(self, image_set, mode, eval) -> None:
        super().__init__(image_set, mode, eval)
        if eval == 'mr':
            if mode == 'train':
                self.targets,self._image_paths,self.image_ids = self._remove_bad_images()
        else:
            self.targets,self._image_paths,self.image_ids = self._remove_bad_images()

    def _get_image_set(self):
        return self._image_set

    def _load_image_ids(self) -> list:
        sequences, ids = [], []
        images_root_path = os.path.join(self._image_set)

        splits = self._get_sequence_splits()

        for seq in splits[self._mode]:
            V_path = os.path.join(images_root_path, seq)
            v_numbers = os.listdir(V_path)
            for v_num in v_numbers:
                sequence_path = os.path.join(images_root_path, seq, v_num,
                                         "images/")
                files = os.listdir(sequence_path)
                ids += [seq + "/" + v_num + "/" + name[:-4] for name in files if
                        name.endswith('.jpg')]
        return ids

    def _load_image_paths(self) -> list:
        image_paths = []
        images_root_path = os.path.join(self._image_set)
        for single_id in self.image_ids:
            seq = single_id.split("/")[0]
            v_name = single_id.split("/")[1]
            file = single_id.split("/")[2] + ".jpg"
            path = os.path.join(images_root_path, seq,v_name,
                                "images/", file)
            image_paths.append(path)
        return image_paths

    def _load_annotations(self) -> Tuple[list, list]:
        targets = []
        label_list = {'person':1,'people':2,'person-fa':3,'person?':4}
        images_root_path = os.path.join(self._image_set)

        for index, single_id in enumerate(self.image_ids):
            seq = single_id.split("/")[0]
            v_name = single_id.split("/")[1]
            file = single_id.split("/")[2] + ".json"
            ann_path = os.path.join(images_root_path, seq, v_name, "annotations/", file)
            target = {}
            id = torch.tensor(int(seq[-2:] + v_name[-3:] + single_id[-5:]))
            with open(ann_path) as json_file:
                data = json.load(json_file)
                if len(data) > 0:
                    boxes = [tar["pos"] for tar in data]
                    box = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
                    # box = box/torch.tensor([640,480,640,480])
                    target['boxes'] = box
                    target['boxes'][:,2] = box[:,0] + box[:,2]
                    target['boxes'][:,3] = box[:,1] + box[:,3]
                    target['image_id'] = id
                    target['labels'] = torch.as_tensor([label_list[tar["lbl"]] for tar in data], dtype=int)
                    target["orig_size"] = torch.tensor([480,640])
                else:
                    target['boxes'] = None
                    target['image_id'] = id
                    target["orig_size"] = torch.tensor([480,640])

            targets.append(target)
        return targets
            # with open(ann_path) as json_file:
            #     data = json.load(json_file)
            #     if len(data) > 0:
            #         boxes = []
            #         for tar in data:
            #             num_person = 0
            #             if tar["lbl"] == "person":
            #                 num_person += 1
            #                 boxes.append(tar["pos"])
            #         box = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            #                 # box = box/torch.tensor([640,480,640,480])
            #         target['boxes'] = box
            #         target['boxes'][:,2] = box[:,0] + box[:,2]
            #         target['boxes'][:,3] = box[:,1] + box[:,3]
            #         target['image_id'] = id
            #         target['labels'] = torch.zeros(len(box),dtype=int)
            #         target["orig_size"] = torch.tensor([480,640])
            #         if len(boxes) == 0:
            #             target['boxes'] = None
            #             target['image_id'] = id
                        
            #     else:
            #         target['boxes'] = None
            #         target['image_id'] = id
            #         target["orig_size"] = torch.tensor([480,640])
            
            # targets.append(target)

    def _get_sequence_splits(self):
        return {
        "train": [
            "set00", "set01", "set02", "set03", "set04", "set05"
            ],
        "val": [
            "set06", "set07", "set08", "set09", "set10"
            ]
    }

    def _remove_bad_images(self):
        targets,image_paths,image_ids = [],[],[]
        for target,image_path,image_id in zip(self.targets,self._image_paths,self.image_ids):
            if target['boxes'] is not None:
                targets.append(target)
                image_paths.append(image_path)
                image_ids.append(image_id)
        return targets,image_paths,image_ids

    def _delete_sample_at(self, index):
        del self.targets[index]
        del self._image_paths[index]
        del self.image_ids[index]

def build_caltech(image_set, args):
    mode = image_set
    dataset = Caltech_Dataset(args.caltech_path, mode, args.eval_mode)
    return dataset
