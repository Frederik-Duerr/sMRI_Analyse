from sklearn.model_selection import train_test_split

def test_train_split(id_diag,model):
    if model == 'deep':
        X = id_diag['patient_id']
        y = id_diag['lable']

        X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=45,
        stratify=y
        )  
        return X_train,X_test,y_train,y_test
    else:  
        X = id_diag['patient_id']
        y = id_diag['lable']
        
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
        return X_train, X_test, y_train, y_test, X_val,y_val


# Gathering all the images belonging to the patient id´s of each set.
def spliting_patient_id_and_images(meta_data_frame, X_train, X_test):

    train_data = meta_data_frame[
        meta_data_frame['patient_id'].isin(X_train)
    ].copy()

    test_data = meta_data_frame[
        meta_data_frame['patient_id'].isin(X_test)
    ].copy()
    '''
    val_data = meta_data_frame[
        meta_data_frame['patient_id'].isin(X_val)
    ].copy()
    '''
    y_train = train_data[['patient_id', 'lable']]
    y_test = test_data[['patient_id', 'lable']]
    # y_val = val_data[['patient_id', 'lable']]

    X_train_image_path = train_data[['image_path','Orientation']]
    X_test_image_path = test_data[['image_path','Orientation']]
    # X_val_image_path = val_data['image_path']

    return (
        y_train,
        y_test,
        # y_val,
        X_train_image_path,
        X_test_image_path,
        # X_val_image_path
    )