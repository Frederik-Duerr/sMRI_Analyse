def test_data(dictionary_with_all_three_orientationen, number_of_patients):

    # Use the first orientation to select patients
    first_orientation = next(
        iter(dictionary_with_all_three_orientationen.values())
    )

    patient_table = first_orientation[
        ['patient_id', 'diagnosis']
    ].drop_duplicates(subset='patient_id')

    selected_patients = []

    # Select patients separately for each diagnosis
    for diagnosis in patient_table['diagnosis'].unique():

        patients = patient_table.loc[
            patient_table['diagnosis'] == diagnosis,
            'patient_id'
        ]

        n = min(number_of_patients, len(patients))

        selected = patients.sample(
            n=n,
            random_state=45
        )

        selected_patients.extend(selected.tolist())

    # Apply the same patient selection to all orientations
    patient_id_Diagnosis = {}

    for key, value in dictionary_with_all_three_orientationen.items():

        patient_id_Diagnosis[key] = value[
            value['patient_id'].isin(selected_patients)
        ].copy()

    return patient_id_Diagnosis