from sklearn.model_selection import train_test_split

def test_train_split(id_diag):
    X = id_diag['patient_id']
    y = id_diag['diagnosis']

    X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=45,
    stratify=y
)    
    # Second: separate validation from train+validation
    X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.1765,
    random_state=45,
    stratify=y_train_val
)
    return X_train, X_test, y_train, y_test, X_val,y_val


# Gathering all the images belonging to the patioent id´s of each set.
def spliting_patient_id_and_images(meta_data_frame, X_train, X_test, X_val):
    y_train = meta_data_frame[
    meta_data_frame['patient_id'].isin(X_train)
]

    y_val = meta_data_frame[
    meta_data_frame['patient_id'].isin(X_val)
]

    y_test = meta_data_frame[
    meta_data_frame['patient_id'].isin(X_test)
]
    return y_train,y_test,y_val