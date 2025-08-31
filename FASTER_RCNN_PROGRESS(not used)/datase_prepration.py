

import os
import json
import cv2
import torch
from torchvision.transforms import functional as F




class TextDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_file):
        self.image_dir = image_dir
        with open(annotation_file) as f:
            self.coco = json.load(f)
        self.image_info = {img['id']: img for img in self.coco['images']}
        self.annotations = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            self.annotations.setdefault(img_id, []).append(ann)

    def __getitem__(self, idx):
        img_id = list(self.image_info.keys())[idx]
        img_data = self.image_info[img_id]
        img_path = os.path.join(self.image_dir, img_data['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = []
        for ann in self.annotations.get(img_id, []):
            boxes.append(ann['bbox'])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id])
        }

        return F.to_tensor(image), target

    def __len__(self):
        return len(self.image_info)
