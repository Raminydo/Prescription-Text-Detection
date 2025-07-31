import cv2 as cv
import joblib
from model import extract_features


# detection
def detect_text_regions(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 10)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 5))
    morph = cv.dilate(thresh, kernel, iterations=2)
    contours, _ = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    return contours


# classification
def classify(image_path, model_path=None):
    image = cv.imread(image_path)
    original = image.copy()
    clf = joblib.load(model_path)

    contours = detect_text_regions(image)

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


    cv.imshow('Detected text regions', original)
    cv.waitKey(0)
    cv.destroyAllWindows()



# ----------------------------------------------------------
# ----------------------------------------------------------
classify('test.jpg', 'boosted_model.pkl')


