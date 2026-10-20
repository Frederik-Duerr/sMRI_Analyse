import pandas as pd

from sklearn.model_selection import train_test_split
import numpy as np
from pipeline import pipeline
from test_train_Split import test_train_split
axial = pd.read_csv('C:/Users/Johan/OneDrive - ucp.pt/Desktop/Privat/Portugal/UCP/Semester 3/Biomedical Project/MRI Project/Features/axial_features_organized_with_Target.csv')
sagittal = pd.read_csv('C:/Users/Johan/OneDrive - ucp.pt/Desktop/Privat/Portugal/UCP/Semester 3/Biomedical Project/MRI Project/Features/sagittal_features_organized_with_Target.csv')
coronal = pd.read_csv('C:/Users/Johan/OneDrive - ucp.pt/Desktop/Privat/Portugal/UCP/Semester 3/Biomedical Project/MRI Project/Features/coronal_features_organized_with_Target.csv')

print(axial.columns)
feature_dictionary = {
    'Axial': axial,
    'Coronal' : coronal,
    'Sagittal' : sagittal
}

test_Train_Seperation = test_train_split(feature_dictionary)
print(test_Train_Seperation['Axial'].keys())

reports = pipeline(test_Train_Seperation,axial['Group'].unique())
print(reports['Axial'])
# print(reports['Coronal'])
# print(reports['Sagittal'])

