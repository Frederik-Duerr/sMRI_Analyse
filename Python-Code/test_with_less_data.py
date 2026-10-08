import pandas as pd
def test_data(patient_id_Diagnosis,number_of_patients):
    patient_id_Diagnosis = (
    patient_id_Diagnosis
    .groupby('lable', group_keys=False)
    .sample(n=number_of_patients, random_state=42)
    .reset_index(drop=True)
)
    return patient_id_Diagnosis