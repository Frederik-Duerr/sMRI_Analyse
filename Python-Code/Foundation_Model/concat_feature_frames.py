from pathlib import Path
import pandas as pd

def concat_feature_frames(meta_data, feature_vector, orientation):

    # ============================================================
    # 1. Create patient-level diagnosis mapping
    # ============================================================

    patient_labels = meta_data.loc[
        meta_data['Orientation'].str.lower() == 'axial',
        ['patient_id', 'diagnosis']
    ].drop_duplicates(subset='patient_id').copy()

    patient_labels['patient_id'] = (
        patient_labels['patient_id'].astype(str)
    )

    # ============================================================
    # 2. Convert filename index into a regular column
    # ============================================================

    feature_vector = feature_vector.reset_index()

    # ============================================================
    # 3. Extract patient ID from filename
    # ============================================================

    feature_vector['patient_id'] = (
        feature_vector['filename'].str.extract(r'^([^_]+)_')[0]
    )

    feature_vector['patient_id'] = (
        feature_vector['patient_id'].astype(str)
    )

    # Add orientation
    feature_vector['Orientation'] = orientation

    # ============================================================
    # 4. Merge patient diagnoses with DINOv2 features
    # ============================================================

    Whole_data_Frame = feature_vector.merge(
        patient_labels,
        how='left',
        on='patient_id',
        validate='many_to_one'
    )

    # Check for missing diagnoses
    if Whole_data_Frame['diagnosis'].isna().any():
        raise ValueError(
            f"Missing diagnoses found in {orientation}"
        )

    return Whole_data_Frame