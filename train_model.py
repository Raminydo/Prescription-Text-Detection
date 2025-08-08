'''
If a model is not previously trained and saved,
run this file to build and train a classification model.
'''

from CLASSIFICATION_MODEL.classification_model import train_model


train_model('data/train', 'data/validation', 'boosted_model.pkl')
