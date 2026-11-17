# ============================================================
# 1. IMPORTS
# ============================================================

# External libraries
import pandas as pd
import os
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
# Project-specific functions
from Loading_data import (
    read_folders,
    extracting_meta_data,
    reading_in_orientations,
)
from sklearn.metrics import confusion_matrix
from mil_multi_orientation import (
    build_long_data, check_data, run_cv,
    attention_by_orientation, top_slices,
)
from test_train_split import test_train_split
from lable import label_change
from concat_feature_frames import concat_feature_frames
from test_with_less_data import test_data

from pipeline import pipeline_sclice_level, patient_pipeline,pipeline_MIL,hold_out_pipline, sclice_independed
from test_train_Split_Matlab import test_train_split_matlab
from evaluation import evaluation_fold, sumariese_cv
from Statisticle_Visualisations import confucion_Matrix
from pipeline import aggregation_of_results
from MultiinstanceLearning import MRIDataset,AttentionAggregation,MLPClassifiere, Alzheimer
# ============================================================
# 2. PROJECT CONFIGURATION
# ============================================================

# Define the root directory of the Project
PROJECT_DIR = (
 r'C:\Users\Johan\Desktop\Privat\Portugal\UCP\Semester 3\Biomedical Project\MRI Project'
)

# Define the directory containing the preprocessed MRI images.
PREPROCESSED_DIR = PROJECT_DIR + r"\Pre_processed"
from pathlib import Path

root = Path(PREPROCESSED_DIR)

print("\n--- PATH DEBUGGING ---")
print("PROJECT_DIR:", Path(PROJECT_DIR).resolve())
print("PREPROCESSED_DIR:", root.resolve())
print("Directory exists:", root.exists())

print("\nContents of PREPROCESSED_DIR:")

for folder in root.iterdir():
    print(folder.name)

# Define the paths to the extracted DINOv2 features.
FEATURE_PATHS = {
    "axial": PROJECT_DIR + r"\Features\Axial_Features.parquet",
    "coronal": PROJECT_DIR + r"\Features\Coronal_Features.parquet",
    "sagittal": PROJECT_DIR + r"\Features\Sagittal_Features.parquet",
}


# Set the maximum number of patients per diagnosis for "quick" testing.
MAX_PATIENTS_PER_DIAGNOSIS = 20

# ============================================================
# 3. LOAD IMAGE METADATA
# ============================================================

# Read the original MRI directory structure.
# The resulting data contains information about the available images.
image_data = read_folders(main_path=PREPROCESSED_DIR)

# Convert the image information into a Pandas DataFrame.
image_data = pd.DataFrame(image_data)

# Prepare the image metadata for subsequent feature integration.
# Relevant information may include patient IDs, filenames,
# image paths, diagnoses, and orientations.
image_metadata_df = extracting_meta_data(meta_data_frame=image_data)

# Create patient-level diagnosis mapping from axial metadata

patient_labels = image_metadata_df.loc[
    image_metadata_df['Orientation'].str.lower() == 'axial',
    ['patient_id', 'diagnosis']
].drop_duplicates()

patient_labels['patient_id'] = (
    patient_labels['patient_id'].astype(str)
)

# print(patient_labels.groupby('diagnosis')['patient_id'].nunique())


# ============================================================
# 4. LOAD DINOv2 FEATURE DATA
# ============================================================

# Load the extracted feature tables for each MRI orientation.
# Each dictionary key identifies the orientation.
orientation_feature_dfs = {}

for orientation, feature_path in FEATURE_PATHS.items():
    orientation_feature_dfs[orientation] = reading_in_orientations(
        feature_path
    )
print(orientation_feature_dfs.items())
patient_features = {}

for orientation, data in orientation_feature_dfs.items():

    data = data.copy()
    data["patient_id"] = data.index.str.split("_").str[0]

    patient_features[orientation] = data.groupby("patient_id").mean()
combined_data = pd.concat(
    patient_features,
    axis=1,
    join="inner"
)
combined_data.columns = [
    f"{orientation}_{feature}"
    for orientation, feature in combined_data.columns
]
combined_data = combined_data.reset_index()

combined_data = combined_data.merge(
    patient_labels,
    on="patient_id",
    how="left"
)

combined_data = combined_data.set_index("patient_id")

print(combined_data.shape)
print(combined_data.index.nunique())
print(combined_data.isna().sum().sum())
print(combined_data.columns)

# ============================================================
# CHECK: NUMBER OF SLICES PER PATIENT IN PARQUET FILES
# ============================================================

# Create patient-diagnosis mapping from axial metadata
patient_labels = image_metadata_df.loc[
    image_metadata_df['Orientation'].str.lower() == 'axial',
    ['patient_id', 'diagnosis']
].drop_duplicates()

patient_labels['patient_id'] = (
    patient_labels['patient_id'].astype(str)
)

# Check whether patient IDs have unique diagnoses
print("Patients with multiple diagnoses:")
print(
    patient_labels.groupby('patient_id')['diagnosis'].nunique()
    .loc[lambda x: x > 1]
)

for orientation, feature_df in orientation_feature_dfs.items():

    # Extract filename from index
    Data = feature_df.reset_index()

    # Extract patient ID from filename
    Data['patient_id'] = (
        Data['filename'].str.extract(r'^([^_]+)_')[0]
    )

    Data['patient_id'] = Data['patient_id'].astype(str)

    # Count slices per patient
    patient_slice_counts = Data.groupby('patient_id').size()

    # Add diagnosis information
    Data = Data.merge(
        patient_labels,
        on='patient_id',
        how='left'
    )

    print(f"\n========== {orientation.upper()} ==========")

    print("Total embeddings:", len(Data))
    print("Unique patients:", Data['patient_id'].nunique())

    print("\nSlices per patient:")
    print(patient_slice_counts.value_counts().sort_index())

    print("\nSlices per diagnosis:")
    print(Data.groupby('diagnosis').size())

    print("\nPatients per diagnosis:")
    print(
        Data[['patient_id', 'diagnosis']]
        .drop_duplicates()
        .groupby('diagnosis')['patient_id']
        .nunique()
    )
    

# ============================================================
# CHECK: DINOv2 FEATURES PER DIAGNOSIS AND ORIENTATION
# ============================================================

for orientation, feature_df in orientation_feature_dfs.items():

    # Convert filename index into a regular column
    feature_df = feature_df.reset_index()

    # Select metadata for the current orientation
    metadata = image_metadata_df.loc[
        image_metadata_df['Orientation'].str.lower() == orientation,
        ['filename', 'patient_id', 'diagnosis']
    ]

    # Match filenames with their corresponding metadata
    check_df = feature_df[['filename']].merge(
        metadata,
        how='left',
        on='filename',
        indicator=True
    )

    print(f"\n========== {orientation.upper()} ==========")

    print("Total embeddings:", len(check_df))
    print("Matched files:", (check_df['_merge'] == 'both').sum())
    print("Unmatched files:", (check_df['_merge'] == 'left_only').sum())

    matched = check_df.loc[check_df['_merge'] == 'both']

    # Count images per diagnosis
    print("\nSlices per diagnosis:")
    print(matched['diagnosis'].value_counts().sort_index())

    # Count patients per diagnosis
    print("\nPatients per diagnosis:")
    print(
        matched[['patient_id', 'diagnosis']]
        .drop_duplicates()
        .groupby('diagnosis')['patient_id']
        .nunique()
    )

    # Count slices per patient and diagnosis
    patient_slices = matched.groupby(
        ['diagnosis', 'patient_id']
    ).size()

    print("\nSlices per patient:")
    print(
        patient_slices.groupby(level='diagnosis').agg(
            ['count', 'min', 'max', 'mean']
        )
    )
'''
# ============================================================
# 5. LOAD Matlab extracted Features
# ============================================================
# Set these paths to the directory where your features extraction data is stored
axial = pd.read_csv('C:/Users/Johan/Desktop/Privat/Portugal/UCP/Semester 3/Biomedical Project/MRI Project/Features/axial_features_organized_with_Target.csv')
sagittal = pd.read_csv('C:/Users/Johan/Desktop/Privat/Portugal/UCP/Semester 3/Biomedical Project/MRI Project/Features/sagittal_features_organized_with_Target.csv')
coronal = pd.read_csv('C:/Users/Johan/Desktop/Privat/Portugal/UCP/Semester 3/Biomedical Project/MRI Project/Features/coronal_features_organized_with_Target.csv')

feature_dictionary_matlab = {
    'Axial': axial,
    'Coronal' : coronal,
    'Sagittal' : sagittal
}
# print(feature_dictionary_matlab['Axial']['Group'].unique()) # [0 1 2 3 4 7 8]
# Homogenising the Column names



mapping = {'CN' : 0,
        'SMC' : 1,
        'EMCI' : 2,
        'MCI' : 3,
        'LMCI' : 4,
        'AD' : 5}

for orientation, Data in feature_dictionary_matlab.items():
    Data.rename(columns={'Target':'diagnosis',
                         'Group':'lable',
                         'PatientID':'patient_id'}, inplace = True)
    Data['lable'] = Data['diagnosis'].map(mapping)

# print(feature_dictionary_matlab['Axial']['lable'].unique())
'''

# ============================================================
# 6. COMBINE IMAGE METADATA AND DINOv2 FEATURES
# ============================================================

# Combine the original image metadata with the corresponding
# DINOv2 feature vectors for each orientation.

combined_feature_dfs = {}

for orientation, feature_df in orientation_feature_dfs.items():
    combined_feature_dfs[orientation] = concat_feature_frames(
        image_metadata_df,
        feature_df,
        orientation,
    )
    
for orientation, Data in combined_feature_dfs.items():

    print(f"\nOrientation: {orientation}")

    print("Rows:", len(Data))
    print("Patients:", Data['patient_id'].nunique())

    print("\nPatients per diagnosis:")
    print(
        Data[['patient_id', 'diagnosis']]
        .drop_duplicates()
        .groupby('diagnosis')['patient_id']
        .nunique()
    )

    print("\nSlices per patient:")
    print(Data.groupby('patient_id').size().value_counts())
# Inspect the columns of the combined axial feature DataFrame.



# ============================================================
# 7. ENCODE DIAGNOSTIC LABELS
# ============================================================

# Apply the diagnostic label conversion separately to each orientation.
# Store the resulting DataFrames without overwriting other orientations.
labeled_feature_dfs = {}

for orientation, feature_df in combined_feature_dfs.items():
    labeled_feature_dfs[orientation] = label_change(feature_df)


# print(labeled_feature_dfs.items())

'''
EXCEL_PATH = os.path.join(
    PROJECT_DIR, 'Concated_labled_DINO.xlsx'
)

with pd.ExcelWriter(EXCEL_PATH) as writer:
    for orientation, Data in labeled_feature_dfs.items():
        Data.to_excel(
            writer,
            sheet_name=orientation,
            index=False
        )

print(f'File saved to: {EXCEL_PATH}')
'''

# ============================================================
# 8. QUICK TEST: LIMIT THE NUMBER OF PATIENTS
# ============================================================

# Reduce the datasets for initial pipeline testing.
# The function is expected to limit the number of patients per diagnosis.
# Check that it selects consistent patient IDs across orientations.


test_feature_dfs = test_data( 
    labeled_feature_dfs,
    MAX_PATIENTS_PER_DIAGNOSIS
)

# test_feature_dfs_matlab = test_data(feature_dictionary_matlab,MAX_PATIENTS_PER_DIAGNOSIS)


# Inspect the resulting datasets.
for orientation, feature_df in test_feature_dfs.items():
    print(f"\nOrientation: {orientation}")
    print(f"Number of rows: {len(feature_df)}")
    print(f"Number of patients: {feature_df['patient_id'].nunique()}")
    print("Diagnostic label counts:")
    print(feature_df["lable"].value_counts())
'''
# Inspect the resulting datasets.
for orientation, feature_df in test_feature_dfs_matlab.items():
    print(f"\nOrientation: {orientation}")
    print(f"Number of rows: {len(feature_df)}")
    print(f"Number of patients: {feature_df['patient_id'].nunique()}")
    print("Diagnostic label counts:")
    print(feature_df["lable"].value_counts())
'''
# ============================================================
# 9. SEPARATE FEATURES, TARGETS, AND PATIENT IDENTIFIERS
# ============================================================

# Define metadata columns that must not be used as model input features.
# Keep the patient IDs separately for group-based data splitting.
metadata_columns = [
    "Orientation",
    "diagnosis",
    "patient_id",
    "filename",
    "image_path",
    "lable",
]



#Test_Train_Split_Matlab = test_train_split_matlab(test_feature_dfs_matlab)
Test_train_by_orientation= test_train_split(test_feature_dfs)
reference_train = set(
    Test_train_by_orientation['axial']['patient_ids_train']
)

reference_test = set(
    Test_train_by_orientation['axial']['patient_ids_test']
)


for orientation, Data in Test_train_by_orientation.items():
    print(orientation)
    print(Data['X_train'].shape)
    print(Data['y_train'].shape)
    print(Data['patient_ids_train'].shape)
    train_ids = set(Data['patient_ids_train'])
    test_ids = set(Data['patient_ids_test'])

    print(f"\n{orientation.upper()}")

    print("Same train patients:", train_ids == reference_train)
    print("Same test patients:", test_ids == reference_test)

    print("Train/Test overlap:", len(train_ids.intersection(test_ids)))




'''
# ============================================================
# 10. Running pipelines for Foundation Model, and Matlab with CV
# ============================================================
models = ['RF','LR','LDA']
results_sclice = {
    
}
results_patient  ={
    
}

results_independed ={
    
}
for model in models: 
    print(f'\n Processing {model}')
    results_independed[model] = sclice_independed(Test_train_by_orientation,model)
    results_sclice[model] = pipeline_sclice_level(Test_train_by_orientation,model)
    results_patient[model] = patient_pipeline(Test_train_by_orientation,model)


for model in models:
    confucion_Matrix(results_independed[model][2],'Independed',model)
    confucion_Matrix(results_sclice[model][2], 'Sclice', model )
    confucion_Matrix(results_patient[model][2], 'Patient', model )


'''
# ============================================================
# 11. Running hold-out versions for comparinson
# ============================================================
# hold_out_reports_Matlab = hold_out_pipline(Test_Train_Split_Matlab,labeled_feature_dfs['axial']['lable'].unique().astype(str))
#hold_out_reports_Foundation,confusion_matriX = hold_out_pipline(Test_train_by_orientation,labeled_feature_dfs['axial']['lable'].unique().astype(str))

import matplotlib.pyplot as plt
import seaborn as sns

reports, confusion_matrices = hold_out_pipline(Test_train_by_orientation,labeled_feature_dfs['axial']['lable'].unique().astype(str))
lables = ['AD', 'CN', 'EMCI', 'LMCI', 'MCI', 'SMC']
for orientation, cm in confusion_matrices.items():

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=lables,
        yticklabels=lables
    )

    plt.title(f'Confusion Matrix - {orientation}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()
    
from pipeline import final_training
features = combined_data.drop('diagnosis',axis=1)
lables = combined_data['diagnosis']
results = final_training(features,lables)
print("Accuracy:", results["accuracy"])
print("Macro F1:", results["macro_f1"])
print(results["classification_report"])
sns.heatmap(
    results["confusion_matrix"],
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=lables,
    yticklabels=lables
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Final Random Forest - Combined Orientations")
plt.show()
