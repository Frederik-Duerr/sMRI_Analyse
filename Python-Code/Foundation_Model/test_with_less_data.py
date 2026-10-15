import pandas as pd
def test_data(dictionary_with_all_three_orientationen,number_of_patients):
    patient_id_Diagnosis = {

    }
    for key, value in dictionary_with_all_three_orientationen.items():
        patients = value['patient_id'].drop_duplicates().sample(n=number_of_patients,random_state = 45)
        patient_id_Diagnosis[key] = value[value['patient_id'].isin(patients)].copy()
    return patient_id_Diagnosis