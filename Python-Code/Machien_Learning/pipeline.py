import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA,  NMF
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif

def pipeline(Data_Dictionary,lables):
    reports = {

    }
    for orientation, Data in Data_Dictionary.items():
        pipeline = Pipeline([
            ('scaling', MinMaxScaler()),
            ('reduce_dim', PCA(n_components=100)),
            ('classify', DecisionTreeClassifier())
        ])
        TREE_DEPTH_OPTIONS = [5, 10, 20, None]
        LEAF_OPTIONS = [1, 2, 5]
        N_FEATURES_OPTIONS = np.arange(2, 300, 2)
        C_OPTIONS = [1, 10, 100, 1000]
        cv = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=43
        )
        
        param_grid = [
            {
                "reduce_dim": [PCA(iterated_power=7)],
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
        grid = GridSearchCV(pipeline, n_jobs=1, param_grid=param_grid,cv=cv)
        grid.fit(Data['X_train'],Data['y_train'],groups=Data['groups_train'])

        prediction = grid.predict(Data['X_test'])
        reports[orientation] = classification_report(
                                Data['y_test'],
                                prediction,
                                target_names=lables,
                                zero_division=0
                            )
        reports['Best Parameters'] = grid.best_params_
        reports['Best Score'] = grid.best_score_
    return reports