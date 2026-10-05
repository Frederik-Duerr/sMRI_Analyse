from Loading_data import read_folders, sclicing_image_features
from lable import label_change
from Image_control import image_control
from Image_control import Image_histogram
from Patient_Level_Data_Splitting import patient_Table
from transform_images import Image_transforms_train_val
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

print("Original number of patients:")
print(len(patient_id_Diagnosis))

# --------------------------------------------------
# QUICK TEST: maximum 10 patients per diagnosis
# --------------------------------------------------

patient_id_Diagnosis = (
    patient_id_Diagnosis
    .groupby('lable', group_keys=False)
    .sample(n=10, random_state=42)
    .reset_index(drop=True)
)

print("Patients per diagnosis:")
print(patient_id_Diagnosis['lable'].value_counts())

X_train, X_test, y_train, y_test, X_val, y_val = test_train_split(
    patient_id_Diagnosis
)

# X_train, X_test, y_train, y_test, X_val,y_val = test_train_split(patient_id_Diagnosis)

y_train, y_test, y_val, X_train_image_path, X_test_image_path, X_val_image_path = \
    spliting_patient_id_and_images(
        meta_data_frame,
        X_train,
        X_test,
        X_val
    )
print(f'Y Train {y_train[:5]}')
print(f'Y test {y_test[:5]}')
print(f' Y Val {y_val[:5]}')

X_train_tensors, X_val_tensors = Image_transforms_train_val(X_train_image_path,X_val_image_path)

print("Number of training images:", len(X_train_tensors))
print("Number of validation images:", len(X_val_tensors))

model = DINOv2Base(model_name="dinov2_vitb14")
model.eval()

batch_size = 4

with torch.no_grad():
    for i in range(0, len(X_train_tensors), batch_size):

        batch = X_train_tensors[i:i + batch_size]
        x = torch.stack(batch)

        features = model(x)

        print(
            f"Batch {i // batch_size + 1}: "
            f"{x.shape} → {features.shape}"
        )
