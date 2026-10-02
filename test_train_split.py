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



def spliting_patient_id_and_images(meta_data_frame,X_train, X_test, y_train, y_test, X_val,y_val):
    X_train = meta_data_frame[meta_data_frame[X_train],'image_path']
    print(X_train)