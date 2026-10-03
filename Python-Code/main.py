from Loading_data import read_folders, sclicing_image_features
from label import label_change
from Image_control import image_control
from Image_control import Image_histogram
from Patient_Level_Data_Splitting import patient_Table
import pandas as pd
import torch
from test_train_split import test_train_split
from test_train_split import spliting_patient_id_and_images
from dinov2_base import DINOv2Base

# Creating the main Path where rthe data is stored
main_path = 'C:/Users/Johan/OneDrive - ucp.pt/Desktop/Privat/Portugal/UCP/Semester 3/Biomedical Project/MRI Project/Pre_processed' # if you want to run the code on your PC, you need to set the path for your local 
# Reading in the folders of the Data
image_data = read_folders(main_path=main_path)

meta_data_frame = pd.DataFrame(image_data)
# Storing meta date such as Patient ID, Diagnosis, sclice number, and angle
meta_data_frame = sclicing_image_features(meta_data_frame=meta_data_frame)

# Mapping the Diagnosis into a range of 0-5
meta_data_frame = label_change(meta_data_frame)
# print(meta_data_frame.head())

meta_data_frame = image_control(meta_data_frame)

# Image_histogram(dataframe=meta_data_frame)
# print(meta_data_frame['diagnosis'].value_counts(normalize=True)*100) # Balanced Data all 16%
# print(meta_data_frame.duplicated().sum()) # Zero Duplicates

patient_id_Diagnosis = patient_Table(meta_data_frame)
# print(patient_id_Diagnosis.columns)

X_train, X_test, y_train, y_test, X_val,y_val = test_train_split(patient_id_Diagnosis)

y_train,y_test,y_val = spliting_patient_id_and_images(meta_data_frame, X_train, X_test, X_val)

print(y_train[0:5])

'''
model = DINOv2Base() 

x = torch.load(X_test_images)

# Feature extraction
features = model(x)
print(features)
'''