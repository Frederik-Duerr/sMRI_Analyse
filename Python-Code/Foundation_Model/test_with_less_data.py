import pandas as pd
def test_data(dictionary_with_all_three_orientationen,number_of_patients):
    patient_id_Diagnosis = {

    }
    for key, value in dictionary_with_all_three_orientationen.items():
        patient_id_Diagnosis[key] = (
        value
        .groupby('lable', group_keys=False)
        .sample(n=number_of_patients, random_state=42)
        .reset_index(drop=True)
    )
    return patient_id_Diagnosis