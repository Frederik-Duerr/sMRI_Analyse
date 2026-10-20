from sklearn.model_selection import train_test_split

def test_train_split_matlab(id_diag):
    test_train_values = {}

    for orientation, features in id_diag.items():
        test_train_values[orientation] = {}

        # Patient IDs and Labels saving seperatly 
        patient_ids = features['patient_id']
        groups = features['diagnosis']
        y = features['lable']

        # Only Features
        X = features.drop(
            ['patient_id', 'diagnosis', 'lable'],
            axis=1
        )

        # Patienten splitting
        patient_table = features[
            ['patient_id', 'lable']
        ].drop_duplicates()

        train_patients, test_patients = train_test_split(
            patient_table['patient_id'],
            test_size=0.15,
            random_state=45,
            stratify=patient_table['lable']
        )

        # Slices per patient
        train_mask = patient_ids.isin(train_patients)
        test_mask = patient_ids.isin(test_patients)

        test_train_values[orientation].update({
            'X_train': X.loc[train_mask],
            'X_test': X.loc[test_mask],
            'y_train': y.loc[train_mask],
            'y_test': y.loc[test_mask],
            'groups_train': groups.loc[train_mask],
            'patient_ids_train': patient_ids.loc[train_mask],
            'patient_ids_test': patient_ids.loc[test_mask]
        })

    return test_train_values