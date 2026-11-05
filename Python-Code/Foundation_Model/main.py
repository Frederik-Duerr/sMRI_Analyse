# ============================================================
# 1. IMPORTS
# ============================================================

# External libraries
import pandas as pd
from sklearn.metrics import classification_report
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
from Statisticle_Visualisations import confucion_matrix
from pipeline import aggregation_of_results
# ============================================================
# 2. PROJECT CONFIGURATION
# ============================================================

# Define the root directory of the Project
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
# Set these paths to the directory where your features extraction data is stored
axial = pd.read_csv('C:/Users/Johan/OneDrive - ucp.pt/Desktop/Privat/Portugal/UCP/Semester 3/Biomedical Project/MRI Project/Features/axial_features_organized_with_Target.csv')
sagittal = pd.read_csv('C:/Users/Johan/OneDrive - ucp.pt/Desktop/Privat/Portugal/UCP/Semester 3/Biomedical Project/MRI Project/Features/sagittal_features_organized_with_Target.csv')
coronal = pd.read_csv('C:/Users/Johan/OneDrive - ucp.pt/Desktop/Privat/Portugal/UCP/Semester 3/Biomedical Project/MRI Project/Features/coronal_features_organized_with_Target.csv')

feature_dictionary_matlab = {
    'Axial': axial,
    'Coronal' : coronal,
    'Sagittal' : sagittal
}
# print(feature_dictionary_matlab['Axial']['Group'].unique()) # [0 1 2 3 4 7 8]
# Homogenising the Column names
mapping = {'AD' : 0,
        'CN' : 1,
        'EMCI' : 2,
        'LMCI' : 3,
        'MCI' : 4,
        'SMC' : 5}
for orientation, Data in feature_dictionary_matlab.items():
    Data.rename(columns={'Target':'diagnosis',
                         'Group':'lable',
                         'PatientID':'patient_id'}, inplace = True)
    Data['lable'] = Data['diagnosis'].map(mapping)

# print(feature_dictionary_matlab['Axial']['lable'].unique())

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
# print(combined_feature_dfs["axial"].columns)


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

test_feature_dfs_matlab = test_data(feature_dictionary_matlab,MAX_PATIENTS_PER_DIAGNOSIS)

'''
# Inspect the resulting datasets.
for orientation, feature_df in test_feature_dfs.items():
    print(f"\nOrientation: {orientation}")
    print(f"Number of rows: {len(feature_df)}")
    print(f"Number of patients: {feature_df['patient_id'].nunique()}")
    print("Diagnostic label counts:")
    print(feature_df["lable"].value_counts())


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


for orientation, Data in Test_train_by_orientation.items():
    train_ids = set(Data['patient_ids_train'])
    test_ids = set(Data['patient_ids_test'])

    overlap = train_ids.intersection(test_ids)

    print(f"\n{orientation}")
    print("Train patients:", len(train_ids))
    print("Test patients:", len(test_ids))
    print("Overlapping patients:", len(overlap))

# ============================================================
# 10. Running pipelines for Foundation Model, and Matlab with CV
# ============================================================

reports,cv_reports,test_predictions = pipeline(Test_train_by_orientation)
# reports_matlab, cv_reports_matlab,test_predictions_matlab = pipeline(Test_Train_Split_Matlab,labeled_feature_dfs['axial']['lable'].unique().astype(str))
# print(reports_matlab.items())

for orientation, results in cv_reports.items():
    print(f"\n{orientation}")
    print(f"CV Accuracy: {results['accuracy_mean']:.4f} "
          f"+/- {results['accuracy_std']:.4f}")
    print(f"CV F1 Macro: {results['f1_macro_mean']:.4f} "
          f"+/- {results['f1_macro_std']:.4f}")
print(f'Axial: \n {reports['axial']}')
print(f'Coronal: \n {reports['coronal']}')
print(f'Sagittal: \n {reports['sagittal']}')

results = aggregation_of_results(test_predictions)

for orientation, values in results.items():

    print(f"\n{orientation}")

    y_true = values['True patient Diagnosis'].astype(int)
    y_pred = values['Predicted diagnosis'].astype(int)

    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3, 4, 5],
        zero_division=0
    ))
    
'''
for orientation, results in cv_reports_matlab.items():
    print(f"\n{orientation}")
    print(f"CV Accuracy: {results['accuracy_mean']:.4f} "
          f"+/- {results['accuracy_std']:.4f}")
    print(f"CV F1 Macro: {results['f1_macro_mean']:.4f} "
          f"+/- {results['f1_macro_std']:.4f}")
print(f'Axial: \n {reports_matlab['Axial']}')
print(f'Coronal: \n {reports_matlab['Coronal']}')
print(f'Sagittal: \n {reports_matlab['Sagittal']}')
'''
confucion_matrix(results)
#confucion_matrix(test_predictions_matlab)

'''
# ============================================================
# 11. Running hold-out versions for comparinson
# ============================================================
# hold_out_reports_Matlab = hold_out_pipline(Test_Train_Split_Matlab,labeled_feature_dfs['axial']['lable'].unique().astype(str))
#hold_out_reports_Foundation = hold_out_pipline(Test_train_by_orientation,labeled_feature_dfs['axial']['lable'].unique().astype(str))
'''