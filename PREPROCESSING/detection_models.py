

import numpy as np
import cv2 as cv
import easyocr


# custom method ------------------------
def text_detection(image:object):
    """Text region detection based on contours, adaptive thresholding
        and morphological view(a custom method without a model)

    Args:
        image (object): numpy.ndarray

    Returns:
        list: contours
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 10)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 5))
    morph = cv.dilate(thresh, kernel, iterations=2)
    contours, _ = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    return contours




# easyocr method -------------------------
reader = easyocr.Reader(['en', 'fa'])

def text_detection_ocr(image):
    """Text region detection based on contours and EasyOCR

    Args:
        image (object): numpy.ndarray

    Returns:
        list: contours
    """

    img_rgb = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    result = reader.readtext(img_rgb, decoder='greedy', paragraph=True)

    contours = []
    for detection in result:
        box = detection[0]
        contour = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
        contours.append(contour)
    
    return contours
