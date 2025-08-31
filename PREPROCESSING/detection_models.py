

import numpy as np
import cv2 as cv
import easyocr
# import torch
# import torchvision
# from torchvision.transforms import functional as F
# from PIL import Image
from MERGE.utils import *
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# from doctr.io import DocumentFile
# from doctr.models import ocr_predictor
# from craft_text_detector import Craft



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



# new new merge
def merged_detection2(img: object):
    """Text region detection; a new merged and optimized answers of text_detection()
        and text_detection_ocr()

    Args:
        image (object): numpy.ndarray

    Returns:
        list: contours
    """
    det1 = text_detection_ocr(img)
    det2 = text_detection(img)
    merged = merge_contours(det1, det2)

    return merged






###=========================######===============#####=========================######==========

# region tested methods

# # Selective Search method -------------------------
# def text_detection_ss(image:object, min_size:int=500, aspect_ratio_range:tuple=(0.2, 5.0)):
#     """Text region detection based on Selective Search

#     Args:
#         image (object): numpy.ndarray
#         min_size (int, optional): minimum area(in pixels) of a region to be considered valid. Defaults to 500.
#         aspect_ratio_range (tuple, optional): acceptable range of width/height fore a region. Defaults to (0.2, 5.0).

#     Returns:
#         list: contours
#     """
#     ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
#     ss.setBaseImage(image)
#     ss.switchToSelectiveSearchQuality()

#     boxes = ss.process()
#     contours_ss = []

#     for (x, y, w, h) in boxes:
#         if w * h < min_size:
#             continue
#         aspect_ratio = w / float(h)
        
#         if not (aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]):
#             continue
        
#         contour = np.array([
#             [[x, y]],
#             [[x + w, y]],
#             [[x + w, y + h]],
#             [[x, y + h]]
#         ], dtype=np.int32)

#         contours_ss.append(contour)

#     return contours_ss[:10]




# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# model.eval()
# def text_detection_frcnn(image:object, threshold:float=0.1):

#     img_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

#     pil_img = Image.fromarray(img_rgb)
#     img_tensor = F.to_tensor(pil_img)

#     input_tensor = img_tensor.unsqueeze(0)

#     with torch.no_grad():
#         outputs = model(input_tensor)

#     boxes = outputs[0]['boxes']
#     scores = outputs[0]['scores']

#     contours = []
#     for box, score in zip(boxes, scores):
#         if score >= threshold:
#             x1, y1, x2, y2 = box.int().tolist()
#             contour = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2],])
#             contours.append(contour)
#     if contours:
#         print('\n\n yesssssssssssssssssssssssssssssssssssssssss\n\n')
#     return contours



#---------------------------


# # Step 1: Preprocess image to suppress background and enhance text
# def preprocess_image(img):
#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Apply CLAHE to enhance contrast (especially for faint handwriting)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced = clahe.apply(gray)

#     # Apply Gaussian blur to reduce patterned or colored background noise
#     blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

#     # Use adaptive thresholding for better handling of uneven lighting
#     thresh = cv2.adaptiveThreshold(
#         blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY_INV, 11, 2
#     )

#     return thresh

# # Step 2: Morphological operations to connect text components
# def extract_text_regions(thresh):
#     # Use horizontal kernel to connect letters in words
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
#     morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

#     # Optional dilation to help connect handwritten strokes
#     morph = cv2.dilate(morph, np.ones((3, 3), np.uint8), iterations=1)

#     return morph

# # Step 3: Find contours and filter by area
# def find_contours(morph, min_area=100):
#     contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     boxes = []
#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         if w * h > min_area:
#             boxes.append((x, y, w, h))
#     return boxes

# # Step 4: Filter out EasyOCR-detected boxes (optional)
# def filter_easyocr_regions(opencv_boxes, easyocr_boxes, iou_threshold=0.5):
#     def iou(boxA, boxB):
#         xA = max(boxA[0], boxB[0])
#         yA = max(boxA[1], boxB[1])
#         xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
#         yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
#         interArea = max(0, xB - xA) * max(0, yB - yA)
#         boxAArea = boxA[2] * boxA[3]
#         boxBArea = boxB[2] * boxB[3]
#         return interArea / float(boxAArea + boxBArea - interArea)

#     filtered = []
#     for box in opencv_boxes:
#         if all(iou(box, ebox) < iou_threshold for ebox in easyocr_boxes):
#             filtered.append(box)
#     return filtered

# # Step 5: Main function to run detection
# def detect2_text_regions(img, easyocr_boxes=None, mode="all"):
#     """
#     mode = "all" → detect all regions
#     mode = "missing_only" → detect only regions EasyOCR missed
#     """
#     thresh = preprocess_image(img)
#     morph = extract_text_regions(thresh)
#     opencv_boxes = find_contours(morph)

#     if mode == "missing_only" and easyocr_boxes:
#         opencv_boxes = filter_easyocr_regions(opencv_boxes, easyocr_boxes)

#     return opencv_boxes


# =====================================================================
# def text2_detection(image:object):
#     """Text region detection based on contours, adaptive thresholding
#         and morphological view(a classic method without a model)

#     Args:
#         image (object): numpy.ndarray

#     Returns:
#         list: contours
#     """


#     gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced = clahe.apply(gray)

#     blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)


#     thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 10)
#     kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 5))
#     morph = cv.dilate(thresh, kernel, iterations=2)
#     contours, _ = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


#     return contours


############################

# craft = Craft(output_dir=None, crop_type='poly', cuda=False)

# def detect_craft(img):

#     def preprocessing(img):
#         gray = cv.cvtColor(img, cv.COLOR_BAYER_BG2GRAY)

#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         enhanced = clahe.apply(gray)

#         blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
#         processed_img = cv.cvtColor(blurred, cv.COLOR_GRAY2BGR)

#         return processed_img
    
#     processed_img = processed_img(img)
#     result = craft.detect_text(processed_img)
#     polys = result['polygons']

#     contours = []
#     for poly in polys:
#         contour = np.array(poly).astype(np.int32)
#         contours.append(contour)

#     return contours



####################################################++++++++++++++++++++++++++++++++++++++++

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
# model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)

# def detect_text_contours_trocr(image):

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (3, 3), 0)
#     thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 10)

#     kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 5))
#     dilated = cv.dilate(thresh, kernel, iterations=2)
#     # morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=2)

#     contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

#     filtered_contours = [cnt for cnt in contours if cv.contourArea(cnt) > 700]

#     return filtered_contours


# ================================================================

# def detect_text_contours_doctr(image):
#     # Load image
#     doc = DocumentFile.from_images(image)

#     # Load model
#     model = ocr_predictor(pretrained=True)
#     result = model(doc)

#     # Extract bounding boxes
#     contours = []
#     for page in result.pages:
#         for block in page.blocks:
#             for line in block.lines:
#                 for word in line.words:
#                     box = word.geometry
#                     # Convert normalized box to pixel coordinates
#                     x_min, y_min = int(box[0][0] * doc[0].shape[1]), int(box[0][1] * doc[0].shape[0])
#                     x_max, y_max = int(box[1][0] * doc[0].shape[1]), int(box[1][1] * doc[0].shape[0])
#                     contours.append((x_min, y_min, x_max - x_min, y_max - y_min))

#     return contours


#endregion

###=========================######===============#####=========================######==========
