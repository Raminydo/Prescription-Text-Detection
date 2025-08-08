

import numpy as np
from PREPROCESSING.load_file import load_dataset
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib


# buld and train model -------------------------
def train_model(train_folder:str, val_folder:str, model_path:str=None):
    """This funtion build the main classification model
        to classify if a text region is printed or handwritten.

    Args:
        train_folder (str): path for train folder
        val_folder (str): path for validation folder
        model_path (str, optional): path and the name of the trained model. Defaults to None.

    Returns:
        object: a trained classifier
    """

    X_train, y_train = load_dataset(train_folder)
    y_train = np.array([1 if label == 'handwritten' else 0 for label in y_train])
    
    X_val, y_val = load_dataset(val_folder)
    y_val = np.array([1 if label == 'handwritten' else 0 for label in y_val])

    clf = XGBClassifier(n_estimators=100, max_depth=5)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print(classification_report(y_val, y_pred))

    joblib.dump(clf, model_path)

    return clf
