# ============================================================
# 1. IMPORTS
# ============================================================

# External libraries
import pandas as pd

# Project-specific functions
from Loading_data import (
    read_folders,
    sclicing_image_features,
    reading_in_orientations,
)
from lable import label_change
from concat_feature_frames import concat_feature_frames
from test_with_less_data import test_data


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

# Set the maximum number of patients per diagnosis for quick testing.
MAX_PATIENTS_PER_DIAGNOSIS = 6


# ============================================================
# 3. LOAD IMAGE METADATA
# ============================================================

# Read the original MRI directory structure.
# The resulting data contains information about the available images.
image_data = read_folders(main_path=PREPROCESSED_DIR)

# Convert the image information into a Pandas DataFrame.
image_metadata_df = pd.DataFrame(image_data)

# Prepare the image metadata for subsequent feature integration.
# Relevant information may include patient IDs, filenames,
# image paths, diagnoses, and orientations.
image_metadata_df = sclicing_image_features(
    meta_data_frame=image_metadata_df
)


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
# 5. COMBINE IMAGE METADATA AND DINOv2 FEATURES
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
# 6. ENCODE DIAGNOSTIC LABELS
# ============================================================

# Apply the diagnostic label conversion separately to each orientation.
# Store the resulting DataFrames without overwriting other orientations.
labeled_feature_dfs = {}

for orientation, feature_df in combined_feature_dfs.items():
    labeled_feature_dfs[orientation] = label_change(feature_df)


# ============================================================
# 7. QUICK TEST: LIMIT THE NUMBER OF PATIENTS
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
# 8. SEPARATE FEATURES, TARGETS, AND PATIENT IDENTIFIERS
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

# Initialize dictionaries for the model inputs, target labels,
# and patient IDs for each orientation.
feature_matrices = {}
target_vectors = {}
patient_ids_by_orientation = {}

# Extract numerical model inputs, diagnostic targets,
# and patient identifiers from each orientation-specific DataFrame.
for orientation, feature_df in test_feature_dfs.items():

    # Keep patient identifiers for later patient-level splitting.
    patient_ids_by_orientation[orientation] = feature_df["patient_id"]

    # Extract the diagnostic labels.
    target_vectors[orientation] = feature_df["lable"]

    # Remove metadata columns from the feature matrix.
    # Any remaining columns should be checked to ensure they are
    # valid numerical DINOv2 features.
    feature_matrices[orientation] = feature_df.drop(
        columns=metadata_columns,
        errors="ignore",
    )


# ============================================================
# 9. VERIFY THE PREPARED DATA
# ============================================================

# Inspect the dimensions of each feature matrix and target vector.
for orientation in feature_matrices:

    print(f"\nOrientation: {orientation}")
    print(f"Feature matrix shape: {feature_matrices[orientation].shape}")
    print(f"Target vector shape: {target_vectors[orientation].shape}")
    print(
        f"Unique patients: "
        f"{patient_ids_by_orientation[orientation].nunique()}"
    )

# The feature matrices and target vectors are now prepared
# for the subsequent patient-level splitting and model evaluation.
#
# Important:
# The train-test split and cross-validation are not performed above.
# They should be added as a separate stage before model training.

