'''
Run this file to use the application
if a pretrained model is already saved as pickle.
'''

from CLASSIFICATION_MODEL.processing import classify
from PREPROCESSING.detection_models import text_detection_ocr



classify('data/test/37.jpg', 'boosted_model.pkl', text_detection_ocr)
