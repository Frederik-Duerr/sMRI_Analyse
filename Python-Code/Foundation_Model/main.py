# ============================================================
# 1. IMPORTING LIBRARIES AND FUNCTIONS
# ============================================================

''' Importing Libaries'''
import pandas as pd
import torch
from test_train_split import test_train_split
from test_train_split import spliting_patient_id_and_images
from dinov2_base import DINOv2Base

# Importing functions from the project's own Python files
'''Importing Functions form Files'''
from Loading_data import read_folders, sclicing_image_features
from Loading_data import sorting_by_orientation
from lable import label_change
from Image_control import image_control
from Loading_data import reading_in_orientations
from concat_feature_frames import concat_feature_frames
# from Image_control import Image_histogram
from Patient_Level_Data_Splitting import patient_Table
from transform_images import Image_transforms_train_val
from test_with_less_data import test_data
from index_patient_tracking import index_tracking
from concat_feature_frames import concat_feature_frames


# ============================================================
# 2. DEFINING THE MAIN DATA PATHS
# ============================================================

# Main directory containing the preprocessed MRI images.
# The directory structure is expected to contain the diagnoses,
# patient IDs, and corresponding MRI slices.
# This path is used as the input for reading the image data.

# Creating the main Path where the data is stored 
main_path = 'C:/Users/Johan/OneDrive - ucp.pt/Desktop/Privat/Portugal/UCP/Semester 3/Biomedical Project/MRI Project/Pre_processed' 
axial_Collected_Features_path = "C:/Users/Johan/OneDrive - ucp.pt/Desktop/Privat/Portugal/UCP/Semester 3/Biomedical Project/MRI Project/Features/Axial_Features.parquet"
coronal_Collected_Features_path = "C:/Users/Johan/OneDrive - ucp.pt/Desktop/Privat/Portugal/UCP/Semester 3/Biomedical Project/MRI Project/Features/Coronal_Features.parquet"
sagittal_Collected_Features_path = "C:/Users/Johan/OneDrive - ucp.pt/Desktop/Privat/Portugal/UCP/Semester 3/Biomedical Project/MRI Project/Features/Sagittal_Features.parquet"
# Optional path to the directory containing the extracted DINOv2 features.
# Currently inactive because the line is commented out.
# feature_folder = 'C:\Users\Johan\OneDrive - ucp.pt\Desktop\Privat\Portugal\UCP\Semester 3\Biomedical Project\MRI Project\Features'


# ============================================================
# 3. LOADING THE MRI IMAGE DATA
# ============================================================

# Reading in the folders of the Data
# read_folders() collects information from the image directory structure.
# The result is stored in image_data.
image_data = read_folders(main_path=main_path)

# Optional alternative: load the previously extracted feature files.
# Currently inactive.
# feature_data = concat_feature_frames(feature_folder)

# Convert the loaded image information into a Pandas DataFrame.
# The DataFrame is used for the subsequent metadata processing.
meta_data_frame = pd.DataFrame(image_data)

# ============================================================
# 4. IMAGE METADATA AND ORIENTATION HANDLING
# ============================================================

# The following triple-quoted block is currently inactive.
# It contains code for extracting image metadata and separating
# the DataFrame into axial, coronal, and sagittal orientations.
# Python does not execute the statements inside this block.

# Storing meta date such as Patient ID, Diagnosis, sclice number, and angle only once necessary because the function copies the images into another folder, where we will work in

meta_data_frame = sclicing_image_features(meta_data_frame=meta_data_frame)
print(meta_data_frame.head())
print(meta_data_frame.shape)
print(meta_data_frame.info())
'''
axial, coronal, sagittal = sorting_by_orientation(meta_data_frame)
'''

# Load or process the previously extracted axial orientation features.
# The exact behavior depends on the implementation of this function.
# The returned value, if any, is not assigned to a variable here.
# 
orientation_DataFrame = reading_in_orientations(axial_Collected_Features_path)

Whole_dataFrame = concat_feature_frames(meta_data_frame,orientation_DataFrame,'axial')
print(Whole_dataFrame.columns)
# ============================================================
# 5. LABEL ENCODING AND IMAGE QUALITY CONTROL
# ============================================================

# Mapping the Diagnosis into a range of 0-5
# Convert the diagnosis labels into numerical class labels.
# This is useful when preparing the target variable for classification.
meta_data_frame = label_change(Whole_dataFrame)

# Optional inspection of the resulting DataFrame.
# Currently inactive.
# print(meta_data_frame.head())

# Apply the project's image-control procedure.
# The actual checks depend on the implementation of image_control().
meta_data_frame = image_control(meta_data_frame)

# Optional data inspection and quality checks.
# Image_histogram(dataframe=meta_data_frame)
# print(meta_data_frame['diagnosis'].value_counts(normalize=True)*100) # Balanced Data all 16%
# print(meta_data_frame.duplicated().sum()) # Zero Duplicates


# ============================================================
# 6. CREATING A PATIENT-LEVEL TABLE
# ============================================================

# Build a table containing patient-level information.
# This is important because the eventual train-test split should
# separate patients rather than randomly distributing individual slices.
patient_id_Diagnosis = patient_Table(meta_data_frame)

# Inspect the number of patients and the resulting patient-level table.
print("Original number of patients:")
print(len(patient_id_Diagnosis))
print(patient_id_Diagnosis.sort_values('patient_id',ascending=False))
print(patient_id_Diagnosis.head(10))
print(patient_id_Diagnosis.info)


# ============================================================
# 7. OPTIONAL QUICK TEST AND DATA SPLITTING
# ============================================================

# The following triple-quoted block is currently inactive.
# It contains the workflow for a smaller test dataset and the
# subsequent patient-level train, validation, and test split.
#
# The quick test is intended to limit the number of patients per
# diagnosis so that the workflow can be checked before processing
# the complete dataset.

'''
# --------------------------------------------------
# QUICK TEST: maximum 6 patients per diagnosis
# --------------------------------------------------

# Select a smaller dataset for testing the pipeline.
# The function is expected to select a maximum of six patients
# per diagnosis, depending on its implementation.

patient_id_Diagnosis_test = test_data(patient_id_Diagnosis,6)

# Inspect the distribution of diagnoses and the selected patients.
print("Patients per diagnosis:")
print(patient_id_Diagnosis_test['lable'].value_counts())
print(patient_id_Diagnosis_test.sort_values('patient_id',ascending=False).reset_index())
print(patient_id_Diagnosis_test.head(10))

# Split patients into training, test, and validation sets.
# X variables represent the patient-level input data,
# while y variables represent the corresponding diagnostic labels.
X_train, X_test, y_train, y_test, X_val,y_val = test_train_split(
    patient_id_Diagnosis_test
    )

# Fter splitting the data we remain the same index for the patient ID as in the patient_id_Diagnosis_test data Frame. Can we use that to  track the id after 
# Inspect the training data and its indices.
# Preserving patient identifiers is important for tracking the
# relationship between patients and their individual MRI slices.
print(f'X Train Data Frame{X_train.head(10)}')

# Alternative call using the complete patient-level dataset.
# Currently inactive.
# X_train, X_test, y_train, y_test, X_val,y_val = test_train_split(patient_id_Diagnosis)


# Map the patient splits back to the image-level metadata.
# This step associates the MRI images with the patients assigned
# to each split and obtains the corresponding diagnostic labels.
#
# The returned image paths allow the images to be located later
# when extracting features or applying image transformations.

y_train, y_test, y_val, X_train_image_path, X_test_image_path, X_val_image_path = spliting_patient_id_and_images(
        meta_data_frame,
        X_train,
        X_test,
        X_val
    )

# Inspect the labels and image paths after the split.
print(f'Y Train {y_train[:5]}\n')
print(f'Y test {y_test[:5]}\n')
print(X_test_image_path.head(10))
print(X_test_image_path)
'''