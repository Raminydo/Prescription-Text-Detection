

import numpy as np
import cv2 as cv
import easyocr


# custom method ------------------------
def text_detection(image:object):
    """Text region detection based on contours, adaptive thresholding
        and morphological view(a classic method without a model)

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

def text_detection_ocr(image:object):
    """Text region detection based on contours and EasyOCR

    Args:
        image (object): numpy.ndarray

    Returns:
        list: contours
    """
    
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    result = reader.readtext(image_rgb, decoder='greedy', paragraph=True)

    contours = []
    for detection in result:
        box = detection[0]
        contour = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
        contours.append(contour)
    
    return contours




# merged method -------------------------
def merged_detection(image:object):
    """Text region detection; merged and optimized answers of text_detection()
        and text_detection_ocr()

    Args:
        image (object): numpy.ndarray

    Returns:
        list: contours
    """
    # region inner function
    # filter overlap borders of text_detection with text_detection_ocr
    def filter_custom_by_ocr(custom_contours, ocr_contours):

        def contour_inside(contour_small, contour_large, threshold=0.05):
            inside_count = 0

            for point in contour_small:
                x, y = float(point[0][0]), float(point[0][1])

                if cv.pointPolygonTest(contour_large, (x, y), False) >= 0:
                    inside_count += 1
                    
            return inside_count / len(contour_small) >= threshold

        filtered_custom = []
        for c in custom_contours:
            keep = True
            for o in ocr_contours:
                if contour_inside(c, o):
                    keep = False
                    break
            if keep:
                filtered_custom.append(c)

        return filtered_custom
        # endregion


    con1 = text_detection(image)
    con2 = text_detection_ocr(image)
    filtered_con1 = filter_custom_by_ocr(con1, con2)

    all_contours = list(filtered_con1) + list(con2)
    boxes = [cv.boundingRect(c) for c in all_contours]

    indices = cv.dnn.NMSBoxes(boxes, [1.0]*len(boxes), score_threshold=0.5, nms_threshold=0.3)
    indices = np.array(indices).flatten()
    merged_contours = [all_contours[i] for i in indices]

    return merged_contours





# Selective Search method -------------------------
def text_detection_ss(image:object, min_size:int=500, aspect_ratio_range:tuple=(0.2, 5.0)):
    """Text region detection based on Selective Search

    Args:
        image (object): numpy.ndarray
        min_size (int, optional): minimum area(in pixels) of a region to be considered valid. Defaults to 500.
        aspect_ratio_range (tuple, optional): acceptable range of width/height fore a region. Defaults to (0.2, 5.0).

    Returns:
        list: contours
    """
    ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchQuality()

    boxes = ss.process()
    contours_ss = []

    for (x, y, w, h) in boxes:
        if w * h < min_size:
            continue
        aspect_ratio = w / float(h)
        
        if not (aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]):
            continue
        
        contour = np.array([
            [[x, y]],
            [[x + w, y]],
            [[x + w, y + h]],
            [[x, y + h]]
        ], dtype=np.int32)

        contours_ss.append(contour)

    return contours_ss[:10]




