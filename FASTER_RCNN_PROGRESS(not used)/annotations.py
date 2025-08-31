

import os
import json
import cv2
import easyocr
from tqdm import tqdm
from utils import clean_annotations

# initialization
IMAGE_DIR = 'data/test'
ANNOTATION_FILE = 'FASTER_RCNN_PROGRESS/annotations.json'
CONFIDENCE_THRESHOLD = 0.7


# generate COCO annotations with EasyOCR
reader = easyocr.Reader(['en', 'fa'])
coco = {
    'images': [],
    'annotations': [],
    'categories': [{'id': 1, 'name': 'text'}]
}
annotation_id = 1
image_id = 1

for filename in tqdm(os.listdir(IMAGE_DIR)):
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    filepath = os.path.join(IMAGE_DIR, filename)
    image = cv2.imread(filepath)
    height, width = image.shape[:2]

    coco['images'].append({
        'id': image_id,
        'file_name': filename,
        'height': height,
        'width': width
    })

    results = reader.readtext(filepath)
    for result in results:
        (bbox, text, confidence) = result
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        x_min = int(min(x_coords))
        y_min = int(min(y_coords))
        box_width = int(max(x_coords) - x_min)
        box_height = int(max(y_coords) - y_min)

        coco['annotations'].append({
            'id': annotation_id,
            'image_id': image_id,
            'category_id': 1,
            'bbox': [x_min, y_min, box_width, box_height],
            'area': box_width * box_height,
            'iscrowd': 0
        })
        annotation_id += 1

    image_id += 1

# clean and merge annotations
cleaned_coco = clean_annotations(coco)

# save
with open(ANNOTATION_FILE, 'w') as file:
    json.dump(cleaned_coco, file, indent=4)



