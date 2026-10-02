from sklearn.model_selection import train_test_split

def test_train_split(id_diag):
    X = id_diag['patient_id']
    y = id_diag['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=True, shuffle=True
)    
    return X_train, X_test, y_train, y_test