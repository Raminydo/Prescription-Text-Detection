import cv2
import numpy as np
import joblib

# Feature Extraction (same as training)
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))

    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9

    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog_features = hog.compute(resized).flatten()

    edges = cv2.Canny(resized, 100, 200)
    edge_density = np.sum(edges) / (64 * 64)

    hist = cv2.calcHist([resized], [0], None, [32], [0, 256]).flatten()
    hist /= np.sum(hist)

    features = np.concatenate([hog_features, [edge_density], hist])
    return features

# Detect Text Regions
def detect_text_regions(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)

    # Morphological operations to group text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    morph = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Classify and Visualize
def classify_and_draw(image_path, model_path='text_classifier.pkl'):
    image = cv2.imread(image_path)
    original = image.copy()
    clf = joblib.load(model_path)

    contours = detect_text_regions(image)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w*h < 500:  # Filter small regions
            continue

        roi = image[y:y+h, x:x+w]
        features = extract_features(roi)
        label = clf.predict([features])[0]

        color = (0, 255, 0) if label == 'printed' else (0, 0, 255)
        # color = (0, 0, 255)
        cv2.rectangle(original, (x, y), (x+w, y+h), color, 2)
        cv2.putText(original, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Detected Text Regions", original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run Testing
classify_and_draw("test7.jpg")