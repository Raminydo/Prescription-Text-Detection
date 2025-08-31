import cv2 as cv
import numpy as np

'''
Functions needed for the new method of merging contours.
'''


# region old new merge
# # ---------- Utility Functions ----------

# def box_area(box):
#     x, y, w, h = box
#     return w * h

# def aspect_ratio(box):
#     x, y, w, h = box
#     return w / h if h != 0 else 0

# def iou(boxA, boxB):
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
#     yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

#     interW = max(0, xB - xA)
#     interH = max(0, yB - yA)
#     interArea = interW * interH

#     boxAArea = boxA[2] * boxA[3]
#     boxBArea = boxB[2] * boxB[3]

#     unionArea = boxAArea + boxBArea - interArea
#     return interArea / unionArea if unionArea != 0 else 0

# def overlaps_with_any(box, box_list, threshold=0.3):
#     return any(iou(box, other) > threshold for other in box_list)

# def is_reliable(box, min_area=100, max_area=5000, min_ar=0.2, max_ar=5.0):
#     area = box_area(box)
#     ar = aspect_ratio(box)
#     return min_area < area < max_area and min_ar < ar < max_ar

# # ---------- Merge Function ----------

# # def merge_contours(easyocr_boxes, custom_boxes):
# #     merged = easyocr_boxes.copy()
# #     for c_box in custom_boxes:
# #         if is_reliable(c_box) and not overlaps_with_any(c_box, easyocr_boxes):
# #             merged.append(c_box)
# #     return merged

# endregion



def contour_to_bbox(contour):
    x, y, w, h = cv.boundingRect(contour)
    return x, y, w, h


def bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou


def easyocr_box_to_contour(box):
    return np.array(box, dtype=np.int32).reshape((-1,1,2))


def filter_custom_contour(contour, min_area=100, max_area=10000, min_aspect=0.2, max_aspect=5):
    x, y, w, h = cv.boundingRect(contour)
    area = w*h
    if area < min_area or area > max_area:
        return False
    aspect_ratio = w / float(h+1e-5)
    if aspect_ratio < min_aspect or aspect_ratio > max_aspect:
        return False
    return True


def is_bbox_inside(inner, outer):
    x_in, y_in, w_in, h_in = inner
    x_out, y_out, w_out, h_out = outer
    return (x_in >= x_out and
            y_in >= y_out and
            x_in + w_in <= x_out + w_out and
            y_in + h_in <= y_out + h_out)


def remove_nested_contours(contours):
    bboxes = [cv.boundingRect(c) for c in contours]
    to_remove = set()

    for i, bbox_i in enumerate(bboxes):
        for j, bbox_j in enumerate(bboxes):
            if i != j and is_bbox_inside(bbox_i, bbox_j):
                # If contour i is inside contour j, mark i for removal
                # Optional: Only remove if contour j is significantly bigger
                area_i = bbox_i[2] * bbox_i[3]
                area_j = bbox_j[2] * bbox_j[3]
                if area_j > area_i * 1.2:  # bigger by 20%
                    to_remove.add(i)
                    break

    filtered = [c for idx, c in enumerate(contours) if idx not in to_remove]
    return filtered


def remove_small_contours(contours, min_area=500):
    filtered = []
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        area = w * h
        if area >= min_area:
            filtered.append(cnt)
    return filtered


def merge_contours(easyocr_boxes, custom_contours, iou_threshold=0.5, min_area=200):
    # Convert EasyOCR boxes to contours and bboxes
    easyocr_contours = [easyocr_box_to_contour(box) for box in easyocr_boxes]
    easyocr_bboxes = [cv.boundingRect(cnt) for cnt in easyocr_contours]

    # Start with all easyocr contours
    merged_contours = easyocr_contours.copy()

    # For each custom contour, decide whether to add
    for cust_contour in custom_contours:
        cust_bbox = contour_to_bbox(cust_contour)

        # Compute IoU with all EasyOCR bounding boxes
        ious = [bbox_iou(cust_bbox, e_bbox) for e_bbox in easyocr_bboxes]
        max_iou = max(ious) if ious else 0

        # If overlaps significantly with EasyOCR contour, skip
        if max_iou > iou_threshold:
            continue

        # Else, filter based on contour shape (area, aspect ratio)
        if filter_custom_contour(cust_contour):
            merged_contours.append(cust_contour)

    # Remove nested contours inside bigger ones
    cleaned_contours = remove_nested_contours(merged_contours)

    # Remove very small contours
    final_contours = remove_small_contours(cleaned_contours, min_area=min_area)

    return final_contours
