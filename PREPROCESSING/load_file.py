

import os
import numpy as np
import cv2 as cv
from feature_extraction import extract_features




# load and prepare dataset
def load_dataset(folder_path):
    """Loading files from the dataset for training the model

    Args:
        folder_path (str): dataset folder path

    Returns:
        tuple: X, y
    """
    X, y = [], []

    for label in ['printed', 'handwritten']:
        class_folder = os.path.join(folder_path, label)

        for filename in os.listdir(class_folder):
            img_path = os.path.join(class_folder, filename)
            image = cv.imread(img_path)

            if image is not None:
                features = extract_features(image)
                X.append(features)
                y.append(label)

    return np.array(X), np.array(y)
