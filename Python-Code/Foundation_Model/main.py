''' Importing Libaries'''
import pandas as pd
import torch
from test_train_split import test_train_split
from test_train_split import spliting_patient_id_and_images
from dinov2_base import DINOv2Base

'''Importing Functions form Files'''
from Loading_data import read_folders, sclicing_image_features
from Loading_data import sorting_by_orientation
from lable import label_change
from Image_control import image_control
# from Image_control import Image_histogram
from Patient_Level_Data_Splitting import patient_Table
from transform_images import Image_transforms_train_val
from test_with_less_data import test_data
from index_patient_tracking import index_tracking
from concat_feature_frames import concat_feature_frames

# Creating the main Path where the data is stored as well as the extracted feature
main_path = 'C:/Users/Johan/OneDrive - ucp.pt/Desktop/Privat/Portugal/UCP/Semester 3/Biomedical Project/MRI Project/Pre_processed' 
# feature_folder = 'C:/Users/Johan/OneDrive - ucp.pt/Desktop/Privat/Portugal/UCP/Semester 3/Biomedical Project/MRI Project/Features/dinov2_features'

# Reading in the folders of the Data
image_data = read_folders(main_path=main_path)
# feature_data = concat_feature_frames(feature_folder)
meta_data_frame = pd.DataFrame(image_data)
# meta_data_frame = feature_data




# Storing meta date such as Patient ID, Diagnosis, sclice number, and angle
meta_data_frame = sclicing_image_features(meta_data_frame=meta_data_frame)
print(meta_data_frame.head())
print(meta_data_frame.shape)
print(meta_data_frame.info())
axial, coronal, sagittal = sorting_by_orientation(meta_data_frame)

'''

# Mapping the Diagnosis into a range of 0-5
meta_data_frame = label_change(meta_data_frame)
# print(meta_data_frame.head())

meta_data_frame = image_control(meta_data_frame)

# Image_histogram(dataframe=meta_data_frame)
# print(meta_data_frame['diagnosis'].value_counts(normalize=True)*100) # Balanced Data all 16%
# print(meta_data_frame.duplicated().sum()) # Zero Duplicates

patient_id_Diagnosis = patient_Table(meta_data_frame)

print("Original number of patients:")
print(len(patient_id_Diagnosis))
print(patient_id_Diagnosis.sort_values('patient_id',ascending=False))
print(patient_id_Diagnosis.head(10))
print(patient_id_Diagnosis.info)
 

# --------------------------------------------------
# QUICK TEST: maximum 6 patients per diagnosis
# --------------------------------------------------

patient_id_Diagnosis_test = test_data(patient_id_Diagnosis,6)

print("Patients per diagnosis:")
print(patient_id_Diagnosis_test['lable'].value_counts())
print(patient_id_Diagnosis_test.sort_values('patient_id',ascending=False).reset_index())
print(patient_id_Diagnosis_test.head(10))

X_train, X_test, y_train, y_test, X_val,y_val = test_train_split(
    patient_id_Diagnosis_test
    )

# Fter splitting the data we remain the same index for the patient ID as in the patient_id_Diagnosis_test data Frame. Can we use that to  track the id after 
print(f'X Train Data Frame{X_train.head(10)}')
# X_train, X_test, y_train, y_test, X_val,y_val = test_train_split(patient_id_Diagnosis)



y_train, y_test, y_val, X_train_image_path, X_test_image_path, X_val_image_path = spliting_patient_id_and_images(
        meta_data_frame,
        X_train,
        X_test,
        X_val
    )

print(f'Y Train {y_train[:5]}\n')
print(f'Y test {y_test[:5]}\n')
print(X_test_image_path.head(10))
print(X_test_image_path)
'''