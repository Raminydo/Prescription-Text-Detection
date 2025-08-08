

import cv2 as cv
import matplotlib.pyplot as plt
import joblib

from PREPROCESSING.feature_extraction import extract_features



# classification --------------------------
def classify(image_path:str, model_path:str, detection_method:object):
    """A function for detecting text areas in an image and classify them into
        printed or handwritten.

    Args:
        image_path (str): image path
        model_path (str): modle path as a pickle file
        detection_method (object): one of two possible methods for detection(text_detection_ocr or text_detection)

    Returns:
        matrix: result image
    """
    image = cv.imread(image_path)
    original = image.copy()
    clf = joblib.load(model_path)

    contours = detection_method(image)

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)

        if w*h < 500:
            continue

        roi = image[y:y+h, x:x+w]
        features = extract_features(roi)
        label = clf.predict([features])[0]

        if label == 1:
            cv.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(original, 'handwritten', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        else:
            cv.rectangle(original, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv.putText(original, 'printed', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # plt.imshow(original)      # uncomment if app is run withou UI

    return original
