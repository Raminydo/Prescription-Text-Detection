

import cv2 as cv
import numpy as np



# feature extraction --------------------------
def extract_features(image):
    """Feature extraction function based on HOG features and edge density

    Args:
        image (object): numpy.ndarray

    Returns:
        list: features
    """
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
