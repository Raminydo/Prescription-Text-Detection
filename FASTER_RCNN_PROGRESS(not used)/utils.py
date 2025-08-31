

import numpy as np
from sklearn.cluster import DBSCAN

def merge_boxes(boxes, eps=60, min_samples=1):
    """
    Merge nearby bounding boxes using DBSCAN clustering.
    Args:
        boxes: List of [x, y, w, h]
        eps: Distance threshold for clustering
        min_samples: Minimum samples per cluster
    Returns:
        List of merged boxes [x, y, w, h]
    """
    if not boxes:
        return []

    centers = np.array([[x + w / 2, y + h / 2] for x, y, w, h in boxes])
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)

    merged = []
    for label in set(clustering.labels_):
        group = [boxes[i] for i in range(len(boxes)) if clustering.labels_[i] == label]
        if not group:
            continue
        x_min = min([b[0] for b in group])
        y_min = min([b[1] for b in group])
        x_max = max([b[0] + b[2] for b in group])
        y_max = max([b[1] + b[3] for b in group])
        merged.append([x_min, y_min, x_max - x_min, y_max - y_min])
    return merged

def clean_annotations(coco_dict, eps=60):
    """
    Clean and merge bounding boxes from COCO-style dict.
    Args:
        coco_dict: Dictionary with 'images', 'annotations', 'categories'
        eps: Clustering distance threshold
    Returns:
        New COCO-style dict with merged boxes
    """
    from collections import defaultdict

    image_boxes = defaultdict(list)
    for ann in coco_dict['annotations']:
        image_id = ann['image_id']
        bbox = ann['bbox']
        image_boxes[image_id].append(bbox)

    cleaned_annotations = []
    ann_id = 1
    for image in coco_dict['images']:
        image_id = image['id']
        boxes = image_boxes.get(image_id, [])
        merged_boxes = merge_boxes(boxes, eps=eps)

        for bbox in merged_boxes:
            cleaned_annotations.append({
                'id': ann_id,
                'image_id': image_id,
                'category_id': 1,  # assuming text
                'bbox': bbox,
                'area': bbox[2] * bbox[3],
                'iscrowd': 0
            })
            ann_id += 1

    return {
        'images': coco_dict['images'],
        'annotations': cleaned_annotations,
        'categories': coco_dict['categories']
    }