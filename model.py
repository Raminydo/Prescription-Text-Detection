

import cv2 as cv
import os
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib


# feature extraction
def extract_features(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    resized = cv.resize(gray, (64, 64))

    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9

    # HOG features
    hog = cv.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog_features = hog.compute(resized).flatten()

    # edge density
    edges = cv.Canny(resized, 100, 200)
    edge_density = np.sum(edges) / (64 * 64)

    # histogram
    hist = cv.calcHist([resized], [0], None, [32], [0, 256]).flatten()
    hist /= np.sum(hist)

    features = np.concatenate([hog_features, [edge_density], hist])

    return features


# Load Dataset
def load_dataset(folder_path):
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


# train model
def train_model(train_folder, val_folder, model_path=None):

    X_train, y_train = load_dataset(train_folder)
    y_train = np.array([1 if label == 'handwritten' else 0 for label in y_train])
    
    X_val, y_val = load_dataset(val_folder)
    y_val = np.array([1 if label == 'handwritten' else 0 for label in y_val])

    clf = XGBClassifier(n_estimators=100, max_depth=5)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print(classification_report(y_val, y_pred))

    joblib.dump(clf, model_path)

    return clf



# ----------------------------------------------------------
# ----------------------------------------------------------
train_model('data/train', 'data/validation', 'boosted_model.pkl')



