from sklearn.model_selection import train_test_split

def test_train_split(id_diag):
    test_train_values = {
        
    }
    for orientation, features in id_diag.items():
        test_train_values[orientation] = {

        }
        X = id_diag[orientation].drop(['PatientID','Group','Target'],axis=1)
        y = id_diag[orientation]['Target']
        
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