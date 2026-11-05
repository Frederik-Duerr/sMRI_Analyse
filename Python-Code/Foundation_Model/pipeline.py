import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import pandas as pd

def pipeline(Data_Dictionary):

    reports = {}
    cv_results = {}
    test_predictions = {}

    param_grid = {
        'classify__n_estimators': [200, 300],
        'classify__max_features': [0.05, 0.1],
        'classify__min_samples_split': [2],
        'classify__min_samples_leaf': [ 4 ],
        'classify__max_samples': [ None],
        'classify__class_weight': ['balanced_subsample']
    }

    for orientation, Data in Data_Dictionary.items():

        print(f"\n{orientation}")

        X_train = Data['X_train']
        y_train = Data['y_train']

        X_test = Data['X_test']
        y_test = Data['y_test']

        # Random Forest
        model = RandomForestClassifier(
            n_estimators=100,
            max_features='sqrt',
            bootstrap=True,
            max_depth=None,
            max_leaf_nodes=None,
            ccp_alpha=0,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=45,
            n_jobs=1
        )

        model = Pipeline([
            ('classify', model)
        ])

        # Patient-level cross-validation
        groups_train = Data['patient_ids_train']

        cv = StratifiedGroupKFold(
            n_splits=3,
            shuffle=True,
            random_state=42
        )

        # Grid Search        
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring={
                'accuracy': 'accuracy',
                'f1_macro': 'f1_macro'
            },
            refit='f1_macro',
            cv=cv,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        

        search.fit(
            X_train,
            y_train,
            groups=groups_train
        )

        # Best model and CV results
        best_model = search.best_estimator_
        best_index = search.best_index_

        cv_results[orientation] = {
            'accuracy_mean': search.cv_results_['mean_test_accuracy'][best_index],
            'accuracy_std': search.cv_results_['std_test_accuracy'][best_index],
            'f1_macro_mean': search.cv_results_['mean_test_f1_macro'][best_index],
            'f1_macro_std': search.cv_results_['std_test_f1_macro'][best_index],
            'best_params': search.best_params_
        }
        cv_results[orientation].update({
            'train_f1_macro': search.cv_results_[
                'mean_train_f1_macro'
            ][best_index],

            'train_accuracy': search.cv_results_[
                'mean_train_accuracy'
            ][best_index]
        })
        print("Best parameters:", search.best_params_)
        print("Best CV F1:", search.best_score_)

        # Test evaluation using the best model
        prediction = best_model.predict(X_test)

        test_predictions[orientation] = {
            'y_true': y_test.to_numpy(),
            'y_pred': prediction,
            'patient_ids': Data['patient_ids_test'].to_numpy(),
            'y_prob': best_model.predict_proba(X_test),
            'classes': best_model.named_steps['classify'].classes_
        }

        reports[orientation] = classification_report(
            y_test,
            prediction,
            zero_division=0
        )

    return reports, cv_results, test_predictions

def aggregation_of_results(test_predictions):
    results_dict = {
        
    }
    for orinetation, values in test_predictions.items():
        Data = values
        y_true = Data['y_true']
        y_prob = Data['y_prob']
        patient_id = Data['patient_ids']
        classes = Data['classes']
        
        
        results = pd.DataFrame(
        y_prob,
        columns=classes
        )

        results['patient_id'] = patient_id
        results['true_lables'] = y_true
        
        Grouped_id_prob = results.groupby('patient_id')[classes].mean()
        number_diagnosis = results.groupby('patient_id')['true_lables'].nunique()
        if (number_diagnosis > 1).any():
            raise ValueError("Some patients have multiple diagnostic labels!")
            
        true_patinet_diagnosis = results.groupby('patient_id')['true_lables'].first()
        predicted_diagnosis = Grouped_id_prob.idxmax(axis=1)
        results_dict[orinetation] = {
            'Probabilities': Grouped_id_prob,
            'True patient Diagnosis': true_patinet_diagnosis,
            'Predicted diagnosis': predicted_diagnosis
        }
    
    return results_dict

    '''
    reports = {

    }
    for orientation, Data in Data_Dictionary.items():
        pipeline = Pipeline([
            ('scaling', StandardScaler()),
            ('reduce_dim', PCA(n_components=100)), 
            ('classify', DecisionTreeClassifier())
        ])
        TREE_DEPTH_OPTIONS = [5, 10, 20, None]
        LEAF_OPTIONS = [1, 2, 5]
        N_FEATURES_OPTIONS = np.arange(2, 300, 2)
        C_OPTIONS = [1, 10, 100, 1000]
        param_grid = [
            {
                "reduce_dim": [PCA(iterated_power=7), NMF(max_iter=1_000)],
                "reduce_dim__n_components": N_FEATURES_OPTIONS,
                "classify__max_depth": TREE_DEPTH_OPTIONS,
                "classify__min_samples_leaf": LEAF_OPTIONS,
            },
            {
                "reduce_dim": [SelectKBest(mutual_info_classif)],
                "reduce_dim__k": N_FEATURES_OPTIONS,
                "classify__max_depth": TREE_DEPTH_OPTIONS,
                "classify__min_samples_leaf": LEAF_OPTIONS,
            },
        ]
        grid = GridSearchCV(pipeline, n_jobs=1, param_grid=param_grid)
        grid.fit(Data['X_train'],Data['y_train'])

        prediction = grid.predict(Data['X_test'])
        reports[orientation] = classification_report(
                                Data['y_test'],
                                prediction,
                                target_names=lables,
                                zero_division=0
                            )

    return reports
'''
def hold_out_pipline(features, lables):
    reports = {

    }
    for orientation, Data in features.items():
        pipeline = Pipeline([
        ('scaling', StandardScaler()),
        ('reduce_dim', PCA(n_components=100)), 
        ('classify', DecisionTreeClassifier())
        ])
        pipeline.fit_transform(Data['X_train'],Data['y_train'])
        y_pred = pipeline.predict(Data['X_test'])
        reports[orientation] = classification_report(Data['y_test'],y_pred,
                                target_names=lables,
                                zero_division=0) 
    return reports


