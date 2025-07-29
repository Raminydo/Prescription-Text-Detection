import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Feature Extraction
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))  # Normalize size

    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9

    # 1. Histogram of Oriented Gradients (HOG)
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog_features = hog.compute(resized).flatten()

    # 2. Edge Density
    edges = cv2.Canny(resized, 100, 200)
    edge_density = np.sum(edges) / (64 * 64)

    # 3. Intensity Histogram
    hist = cv2.calcHist([resized], [0], None, [32], [0, 256]).flatten()
    hist /= np.sum(hist)  # Normalize

    # Combine features
    features = np.concatenate([hog_features, [edge_density], hist])
    return features

# Load Dataset
def load_dataset(folder_path):
    X, y = [], []
    for label in ['printed', 'handwritten']:
        class_folder = os.path.join(folder_path, label)
        for filename in os.listdir(class_folder):
            img_path = os.path.join(class_folder, filename)
            image = cv2.imread(img_path)
            if image is not None:
                features = extract_features(image)
                X.append(features)
                y.append(label)
    return np.array(X), np.array(y)

# Train Model
def train_model(data_folder, model_path='text_classifier.pkl'):
    X, y = load_dataset(data_folder)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # clf = SVC(kernel='linear', probability=True)
    # clf.fit(X_train, y_train)

    # clf = KNeighborsClassifier(n_neighbors=3, weights='distance')
    # clf.fit(X, y)

    # y_pred = clf.predict(X)
    # print(classification_report(y, y_pred))

    # joblib.dump(clf, model_path)
    # print(f"Model saved to {model_path}")

# Run Training
train_model("training_data")  # Folder structure: training_data/printed/, training_data/handwritten/






