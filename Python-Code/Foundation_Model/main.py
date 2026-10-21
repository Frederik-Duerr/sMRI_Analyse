# ============================================================
# 1. IMPORTS
# ============================================================

# External libraries
import pandas as pd

# Project-specific functions
from Loading_data import (
    read_folders,
    extracting_meta_data,
    reading_in_orientations,
)
from test_train_split import test_train_split
from lable import label_change
from concat_feature_frames import concat_feature_frames
from test_with_less_data import test_data
from pipeline import pipeline
from test_train_Split_Matlab import test_train_split_matlab
from pipeline import hold_out_pipline
# ============================================================
# 2. PROJECT CONFIGURATION
# ============================================================

# Define the root directory of the MRI project.
PROJECT_DIR = (
    r"C:\Users\Johan\OneDrive - ucp.pt\Desktop\Privat\Portugal"
    r"\UCP\Semester 3\Biomedical Project\MRI Project"
)

# Define the directory containing the preprocessed MRI images.
PREPROCESSED_DIR = PROJECT_DIR + r"\Pre_processed"

# Define the paths to the extracted DINOv2 features.
FEATURE_PATHS = {
    "axial": PROJECT_DIR + r"\Features\Axial_Features.parquet",
    "coronal": PROJECT_DIR + r"\Features\Coronal_Features.parquet",
    "sagittal": PROJECT_DIR + r"\Features\Sagittal_Features.parquet",
}

# Set the maximum number of patients per diagnosis for "quick" testing.
MAX_PATIENTS_PER_DIAGNOSIS = 6

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

# ============================================================
# 5. LOAD Matlab extracted Features
# ============================================================

axial = pd.read_csv('C:/Users/Johan/OneDrive - ucp.pt/Desktop/Privat/Portugal/UCP/Semester 3/Biomedical Project/MRI Project/Features/axial_features_organized_with_Target.csv')
sagittal = pd.read_csv('C:/Users/Johan/OneDrive - ucp.pt/Desktop/Privat/Portugal/UCP/Semester 3/Biomedical Project/MRI Project/Features/sagittal_features_organized_with_Target.csv')
coronal = pd.read_csv('C:/Users/Johan/OneDrive - ucp.pt/Desktop/Privat/Portugal/UCP/Semester 3/Biomedical Project/MRI Project/Features/coronal_features_organized_with_Target.csv')

feature_dictionary_matlab = {
    'Axial': axial,
    'Coronal' : coronal,
    'Sagittal' : sagittal
}
# Homogenising the Column names
for orientation, Data in feature_dictionary_matlab.items():
    Data.rename(columns={'Target':'lable',
                         'Group':'diagnosis',
                         'PatientID':'patient_id'}, inplace = True)
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

# Inspect the columns of the combined axial feature DataFrame.
print(combined_feature_dfs["axial"].columns)


# ============================================================
# 7. ENCODE DIAGNOSTIC LABELS
# ============================================================

# Apply the diagnostic label conversion separately to each orientation.
# Store the resulting DataFrames without overwriting other orientations.
labeled_feature_dfs = {}

for orientation, feature_df in combined_feature_dfs.items():
    labeled_feature_dfs[orientation] = label_change(feature_df)


print(labeled_feature_dfs.items())


# ============================================================
# 8. QUICK TEST: LIMIT THE NUMBER OF PATIENTS
# ============================================================

# Reduce the datasets for initial pipeline testing.
# The function is expected to limit the number of patients per diagnosis.
# Check that it selects consistent patient IDs across orientations.

test_feature_dfs = test_data( 
    labeled_feature_dfs,
    MAX_PATIENTS_PER_DIAGNOSIS,
)

# Inspect the resulting datasets.
for orientation, feature_df in test_feature_dfs.items():
    print(f"\nOrientation: {orientation}")
    print(f"Number of rows: {len(feature_df)}")
    print(f"Number of patients: {feature_df['patient_id'].nunique()}")
    print("Diagnostic label counts:")
    print(feature_df["lable"].value_counts())


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

# Test_Train_Split_Matlab = test_train_split_matlab(feature_dictionary_matlab)
Test_train_by_orientation= test_train_split(test_feature_dfs)
print(Test_train_by_orientation['axial'].keys())

# ============================================================
# 10. Running pipelines for Foundation Model, and Matlab with CV
# ============================================================

reports = pipeline(Test_train_by_orientation,labeled_feature_dfs['axial']['lable'].unique().astype(str))
# reports_matlab = pipeline(Test_Train_Split_Matlab,labeled_feature_dfs['axial']['lable'].unique().astype(str))

print(reports['axial'])
print(reports['coronal'])
print(reports['sagittal'])


# ============================================================
# 11. Running hold-out versions for comparinson
# ============================================================
# hold_out_reports_Matlab = hold_out_pipline(Test_Train_Split_Matlab,labeled_feature_dfs['axial']['lable'].unique().astype(str))
hold_out_reports_Foundation = hold_out_pipline(Test_train_by_orientation,labeled_feature_dfs['axial']['lable'].unique().astype(str))

# ============================================================
# 12. Saving the Classification repoerts in a Excel File 
# ============================================================
with pd.ExcelWriter('FoundationModel.xlsx') as writer:
    for orientation, Data in reports.items():
        pd.DataFrame(Data).to_excel(
            writer,
            sheet_name=f'DINO_{orientation}'
        )
'''
    for orientation, Data in reports_matlab.items():
        pd.DataFrame(Data).to_excel(
            writer,
            sheet_name=f'MATLAB_{orientation}'
        )
'''
with pd.ExcelWriter('Hold_Out_Foundation.xlsx') as writer:
    for orientation, Data in reports.items():
        pd.DataFrame(Data).to_excel(
            writer,
            sheet_name=f'DINO_{orientation}'
        )

'''
    for orientation, Data in reports_matlab.items():
        pd.DataFrame(Data).to_excel(
            writer,
            sheet_name=f'DINO_{orientation}'
        )
'''