from sklearn.model_selection import train_test_split


def test_train_split(id_diag):
    test_train_values = {}

    for orientation, features in id_diag.items():
        test_train_values[orientation] = {}

        # Patient IDs und Labels separat speichern
        patient_ids = features['patient_id']
        groups = features['diagnosis']
        y = features['lable']

        # Nur tatsächliche Features verwenden
        X = features.drop(
            ['patient_id','lable','diagnosis','Orientation','filename'],
            axis=1
        )

        # Patienten statt einzelner Slices aufteilen
        patient_table = features[
            ['patient_id', 'lable']
        ].drop_duplicates()

        train_patients, test_patients = train_test_split(
            patient_table['patient_id'],
            test_size=0.15,
            random_state=45,
            stratify=patient_table['lable']
        )

        # Slices anhand der Patienten-IDs zuordnen
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
'''
def test_train_split(id_diag):
    test_train_values = {
        
    }
    for orientation, features in id_diag.items():
        test_train_values[orientation] = {

        }
        X = id_diag[orientation].drop(['patient_id','lable','diagnosis','Orientation','filename'],axis=1)
        y = id_diag[orientation]['lable']
        
        X_train_val, X_test, y_train_val, y_test = train_test_split(
                X,
                y,
                test_size=0.15,
                random_state=45,
                stratify=y
                )
        # Second: separate validation from train + validation
        X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.1765,
        random_state=45,
        stratify=y_train_val
    )
        test_train_values[orientation].update({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_val': X_val,
            'y_val': y_val
        })
    return test_train_values
'''
# Gathering all the images belonging to the patient id´s of each set.
def spliting_patient_id_and_images(meta_data_frame, X_train, X_test,X_val):

    train_data = meta_data_frame[
        meta_data_frame['patient_id'].isin(X_train)
    ].copy()

    test_data = meta_data_frame[
        meta_data_frame['patient_id'].isin(X_test)
    ].copy()

    val_data = meta_data_frame[
        meta_data_frame['patient_id'].isin(X_val)
    ].copy()

    y_train = train_data[['patient_id', 'lable']]
    y_test = test_data[['patient_id', 'lable']]
    y_val = val_data[['patient_id', 'lable']]

    X_train_image_path = train_data[['image_path','Orientation','patient_id']]
    X_test_image_path = test_data[['image_path','Orientation','patient_id']]
    X_val_image_path = val_data['image_path']

    return (
        y_train,
        y_test,
        y_val,
        X_train_image_path,
        X_test_image_path,
        X_val_image_path
    )