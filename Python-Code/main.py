from Loading_data import read_folders, sclicing_image_features
from lable import label_change
from Image_control import image_control
# from Image_control import Image_histogram
from Patient_Level_Data_Splitting import patient_Table
from transform_images import Image_transforms_train_val
from test_with_less_data import test_data
import pandas as pd
import torch
from test_train_split import test_train_split
from test_train_split import spliting_patient_id_and_images
from dinov2_base import DINOv2Base
from index_patient_tracking import index_tracking
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

X_train,X_test,y_train,y_test = test_train_split(
    patient_id_Diagnosis_test,'deep'
    )

# Fter splitting the data we remain the same index for the patient ID as in the patient_id_Diagnosis_test data Frame. Can we use that to  track the id after 
print(f'X Train Data Frame{X_train.head(10)}')
# X_train, X_test, y_train, y_test, X_val,y_val = test_train_split(patient_id_Diagnosis)




y_train, y_test, X_train_image_path, X_test_image_path = \
    spliting_patient_id_and_images(
        meta_data_frame,
        X_train,
        X_test,
        # X_val
    )

print(f'Y Train {y_train[:5]}\n')
print(f'Y test {y_test[:5]}\n')
print(X_test_image_path.head(10))
print(X_test_image_path)
'''
print('That is the number of axial images per id in the test set: ') , print(X_test_image_path.loc[X_test_image_path['Orientation']=='axial'].groupby('patient_id'))
print('That is the number of sagittal images per id in the test set: '),print(X_test_image_path.loc[X_test_image_path['Orientation']=='sagittal'].value_counts('patient_id'))
print('That is the number of coronal images per id in the test set: '),print(X_test_image_path.loc[X_test_image_path['Orientation']=='coronal'].value_counts('patient_id')) # All 619 within 6 Patient ID´s
print('That is the number of axial images per id in the test set: '),print(X_train_image_path.loc[X_train_image_path['Orientation']=='axial'].value_counts('patient_id')) # All 619 with 30 Patients ID´s 

'''

print(index_tracking(y_train, y_test, X_train_image_path, X_test_image_path))

'''
X_train_tensors_axial, X_train_tensors_sagittal,X_train_tensors_coronal = Image_transforms_train_val(X_train_image_path)

print("Number of training images:", len(X_train_tensors_axial))
print("Number of training images:", len(X_train_tensors_sagittal))
print("Number of training images:", len(X_train_tensors_coronal))
# print("Number of validation images:", len(X_val_tensors))

model = DINOv2Base(model_name="dinov2_vitb14")
model.eval()

batch_size = 10

with torch.no_grad():
    final_feature_table_axial = []
    final_feature_table_sagittal = []
    final_feature_table_coronal =[]

    for i in range(0, len(X_train_tensors_axial), batch_size):

        batch = X_train_tensors_axial[i:i + batch_size]
        x = torch.stack(batch)

        features = model(x)

        print(
            f"Batch {i // batch_size + 1}: "
            f"{x.shape} → {features.shape}"
        )
        final_feature_table_axial.append(features)
    final_feature_table_axial = torch.cat(final_feature_table_axial,dim=0)
    print(final_feature_table_axial.shape)

    for i in range(0, len(X_train_tensors_sagittal), batch_size):

        batch = X_train_tensors_sagittal[i:i + batch_size]
        x = torch.stack(batch)

        features = model(x)

        print(
            f"Batch {i // batch_size + 1}: "
            f"{x.shape} → {features.shape}"
        )
        final_feature_table_sagittal.append(features)
    final_feature_table_sagittal = torch.cat(final_feature_table_sagittal,dim=0)
    print(final_feature_table_sagittal.shape)

    for i in range(0, len(X_train_tensors_coronal), batch_size):

        batch = X_train_tensors_coronal[i:i + batch_size]
        x = torch.stack(batch)

        features = model(x)

        print(
            f"Batch {i // batch_size + 1}: "
            f"{x.shape} → {features.shape}"
        )
        final_feature_table_coronal.append(features)
    final_feature_table_coronal = torch.cat(final_feature_table_coronal,dim=0)
    print(final_feature_table_coronal.shape)
'''