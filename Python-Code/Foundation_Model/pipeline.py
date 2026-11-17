import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    RepeatedStratifiedKFold,
)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import pandas as pd
from MultiinstanceLearning import (
    MRIDataset,
    AttentionAggregation,
    MLPClassifiere,
    Alzheimer,
)
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torchmetrics

# ----------------------------------------------------------------------
# Imports (add the new ones to your existing import block)
# ----------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.base import clone  # NEW
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # NEW
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression  # NEW
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler  # Normalizer is NEW
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif


# ----------------------------------------------------------------------
# Slice-level pipeline (same name and signature as before)
# model_name: "RF", "SVC", "LR" or "LDA"
# ----------------------------------------------------------------------
def pipeline_sclice_level(Data_Dictionary, model_name):

    reports = {}
    cv_results = {}
    test_predictions = {}

    random_state = 42  # one seed for CV splits and models

    # ---------------- Model + hyperparameter grid (built once) ----------------
    # Everything that is tuned lives ONLY in the grid, so the constructor values
    # can no longer be silently overridden.
    if model_name == "RF":
        model = RandomForestClassifier(
            bootstrap=True,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,  # GridSearchCV already runs in parallel (n_jobs=-1)
        )
        # Many highly correlated slices per patient: large leaves and small
        # bootstrap samples stop the trees from memorising individual patients.
        param_grid = {
            "classify__n_estimators": [300],
            "classify__max_features": [0.05, 0.1],
            "classify__max_depth": [None, 10],
            "classify__min_samples_leaf": [20, 50],
            "classify__max_samples": [0.2, 0.5],
            "select__k": [20, 40, 60, 80, 120, 160, 200, 300],
        }
        steps = [("select", SelectKBest(score_func=f_classif))]

    elif model_name == "SVC":
        # probability=False during the search: probability=True triggers an extra
        # internal 5-fold CV for every fit and is very slow. The probabilities
        # are computed once after the search (see "final model" below).
        model = SVC(
            probability=False, class_weight="balanced", random_state=random_state
        )
        # Small C = strong regularisation (better with few samples and 384 features).
        # The polynomial kernel was removed (degree 4 overfits easily).
        param_grid = [
            {
                "classify__kernel": ["linear"],
                "classify__C": [0.01, 0.1, 1, 10],
                "select__k": [20, 40, 60, 80, 120, 160, 200, 300],
            },
            {
                "classify__kernel": ["rbf"],
                "classify__C": [0.1, 1, 10],
                "classify__gamma": ["scale", 0.001],
                "select__k": [20, 40, 60, 80, 120, 160, 200, 300],
            },
        ]
        steps = [
            ("l2norm", Normalizer(norm="l2")),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=40)),
        ]

    elif model_name == "LR":
        # L2-regularised logistic regression: strong baseline for embeddings.
        model = LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=random_state
        )
        param_grid = {
            "classify__C": [0.001, 0.01, 0.1, 1],
            "select__k": [20, 40, 60, 80, 120, 160, 200, 300],
        }
        steps = [
            ("l2norm", Normalizer(norm="l2")),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif)),
        ]

    elif model_name == "LDA":
        # LDA with covariance shrinkage works well when n_features ~ n_samples.
        model = LinearDiscriminantAnalysis(solver="lsqr")
        param_grid = {
            "classify__shrinkage": ["auto", 0.1, 0.5, 0.9],
            "select__k": [20, 40, 60, 80, 120, 160, 200, 300],
        }
        steps = [
            ("l2norm", Normalizer(norm="l2")),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif)),
        ]

    else:
        raise ValueError(
            f"model_name must be 'RF', 'SVC', 'LR' or 'LDA', got {model_name!r}"
        )

    # L2 normalisation (per sample) -> standardisation (per feature, fitted per CV fold)
    # -> classifier. Fitting the scaler inside the pipeline avoids leakage.
    pipeline = Pipeline(steps + [("classify", model)])

    # Metrics evaluated in every fold. balanced_accuracy is robust against unequal
    # class sizes in a fold; neg_log_loss penalises over-confident wrong predictions.
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "balanced_accuracy": "balanced_accuracy",
    }
    if model_name != "SVC":  # SVC (probability=False) has no predict_proba
        scoring["neg_log_loss"] = "neg_log_loss"

    # ---------------- Loop over the three orientations ----------------
    for orientation, Data in Data_Dictionary.items():

        print(f"\n{orientation}")

        X_train = Data["X_train"]
        y_train = Data["y_train"]

        X_test = Data["X_test"]
        y_test = Data["y_test"]

        # Patient-level cross-validation: all slices of one patient stay in the same fold
        groups_train = Data["patient_ids_train"]

        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)

        search = GridSearchCV(
            estimator=pipeline,  # GridSearchCV clones it, so reusing it is safe
            param_grid=param_grid,
            scoring=scoring,
            refit="f1_macro",  # best parameters are selected by macro F1
            cv=cv,
            n_jobs=-1,
            verbose=1,
            return_train_score=True,  # needed to detect overfitting
        )
        search.fit(X_train, y_train, groups=groups_train)

        # CV results of the best parameter set (same keys as before + new metrics)
        best_index = search.best_index_
        cv_results[orientation] = {"best_params": search.best_params_}
        for metric in scoring:
            cv_results[orientation][f"{metric}_mean"] = search.cv_results_[
                f"mean_test_{metric}"
            ][best_index]
            cv_results[orientation][f"{metric}_std"] = search.cv_results_[
                f"std_test_{metric}"
            ][best_index]
            cv_results[orientation][f"train_{metric}"] = search.cv_results_[
                f"mean_train_{metric}"
            ][best_index]

        print("Best parameters:", search.best_params_)
        print("Best CV F1:", search.best_score_)
        # Large gap between train F1 and CV F1 = overfitting
        print(
            f"Train F1: {cv_results[orientation]['train_f1_macro']:.3f} | "
            f"CV F1: {cv_results[orientation]['f1_macro_mean']:.3f}"
        )

        # ---------------- Final model for the test evaluation ----------------
        if model_name == "SVC":
            # Refit the best SVC ONCE with probability=True so predict_proba works.
            # (For the SVC, predict() can differ slightly from argmax(predict_proba).)
            best_model = clone(search.best_estimator_)
            best_model.set_params(classify__probability=True)
            best_model.fit(X_train, y_train)
        else:
            best_model = search.best_estimator_

        prediction = best_model.predict(X_test)

        test_predictions[orientation] = {
            "y_true": y_test.to_numpy(),
            "y_pred": prediction,
            "patient_ids": Data["patient_ids_test"].to_numpy(),
            "y_prob": best_model.predict_proba(X_test),
            "classes": best_model.named_steps["classify"].classes_,
        }

        reports[orientation] = classification_report(
            y_test, prediction, zero_division=0
        )

    return reports, cv_results, test_predictions


# ----------------------------------------------------------------------
# Patient-level pipeline (same name and signature as before)
# model_name: "RF", "SVC", "LR" or "LDA"
# ----------------------------------------------------------------------
def patient_pipeline(Data_Dictionary, model_name):

    cv_results = {}
    reports = {}
    test_predictions = {}
    k_values = [40, 80, 160, 300]
    random_state = 42  # one seed for CV splits and models

    # ---------------- Model + hyperparameter grid (built once) ----------------
    if model_name == "RF":
        model = RandomForestClassifier(
            bootstrap=True,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,  # GridSearchCV already runs in parallel (n_jobs=-1)
        )
        # One row per patient: much less correlation than at slice level,
        # so smaller leaves are fine here.
        param_grid = {
            "classify__n_estimators": [300],
            "classify__max_features": [0.05, 0.1],
            "classify__max_depth": [None, 10],
            "classify__min_samples_leaf": [2, 4, 8],
            "classify__max_samples": [None],
            "select__k": k_values,
        }
        steps = [("select", SelectKBest(score_func=f_classif))]

    elif model_name == "SVC":
        # probability=False during the search (much faster); probabilities are
        # computed once for the final model below.
        model = SVC(
            probability=False, class_weight="balanced", random_state=random_state
        )
        param_grid = [
            {
                "classify__kernel": ["linear"],
                "classify__C": [0.01, 0.1, 1, 10],
            },
            {
                "classify__kernel": ["rbf"],
                "classify__C": [0.1, 1, 10],
                "classify__gamma": ["scale", 0.001],
            },
        ]
        steps = [
            ("l2norm", Normalizer(norm="l2")),
            ("scaler", StandardScaler(), ("select", SelectKBest(f_classif, k=40))),
        ]

    elif model_name == "LR":
        model = LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=random_state
        )
        param_grid = {"classify__C": [0.001, 0.01, 0.1, 1], "select__k": k_values}
        steps = [
            ("l2norm", Normalizer(norm="l2")),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif)),
        ]

    elif model_name == "LDA":
        model = LinearDiscriminantAnalysis(solver="lsqr")
        param_grid = {
            "classify__shrinkage": ["auto", 0.1, 0.5, 0.9],
            "select__k": k_values,
        }
        steps = [
            ("l2norm", Normalizer(norm="l2")),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif)),
        ]

    else:
        raise ValueError(
            f"model_name must be 'RF', 'SVC', 'LR' or 'LDA', got {model_name!r}"
        )

    pipeline = Pipeline(steps + [("classify", model)])

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "balanced_accuracy": "balanced_accuracy",
    }
    if model_name != "SVC":  # SVC (probability=False) has no predict_proba
        scoring["neg_log_loss"] = "neg_log_loss"

    # One row per patient -> a plain StratifiedKFold is enough (no groups needed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # ---------------- Loop over the three orientations ----------------
    for orientation, Data in Data_Dictionary.items():

        print(f"\n{orientation}")

        X_train = Data["X_train"]
        y_train = Data["y_train"]

        X_test = Data["X_test"]
        y_test = Data["y_test"]

        # Use the patient ids as index so we can average the slices per patient
        df_train = X_train.copy()
        df_train.index = Data["patient_ids_train"].to_numpy()

        df_test = X_test.copy()
        df_test.index = Data["patient_ids_test"].to_numpy()

        Lables_train = pd.Series(y_train.to_numpy(), index=df_train.index)
        Lables_test = pd.Series(y_test.to_numpy(), index=df_test.index)

        # Mean of all slice features per patient -> one row per patient
        X_train_patient_df = df_train.groupby(level=0).mean()
        X_train_id_feature_mean = X_train_patient_df.to_numpy(copy=True)
        # All slices of a patient share one label; .loc keeps the same row order
        # as the feature matrix (guards against misaligned features/labels)
        id_lable_train = (
            Lables_train.groupby(level=0).first().loc[X_train_patient_df.index]
        )

        X_test_patient_df = df_test.groupby(level=0).mean()
        patient_ids_test = X_test_patient_df.index.to_numpy()
        X_test_id_feature_mean = X_test_patient_df.to_numpy(copy=True)
        id_lable_test = (
            Lables_test.groupby(level=0).first().loc[X_test_patient_df.index]
        )

        print("Train Features:", X_train_id_feature_mean.shape)
        print("Train Labels:", id_lable_train.shape)

        print("Test Features:", X_test_id_feature_mean.shape)
        print("Test Labels:", id_lable_test.shape)

        search = GridSearchCV(
            estimator=pipeline,  # GridSearchCV clones it, so reusing it is safe
            param_grid=param_grid,
            scoring=scoring,
            refit="f1_macro",  # best parameters are selected by macro F1
            cv=cv,
            n_jobs=-1,
            verbose=1,
            return_train_score=True,  # needed to detect overfitting
        )
        search.fit(X_train_id_feature_mean, id_lable_train)

        # CV results of the best parameter set (same keys as before + new metrics)
        best_index = search.best_index_
        cv_results[orientation] = {"best_params": search.best_params_}
        for metric in scoring:
            cv_results[orientation][f"{metric}_mean"] = search.cv_results_[
                f"mean_test_{metric}"
            ][best_index]
            cv_results[orientation][f"{metric}_std"] = search.cv_results_[
                f"std_test_{metric}"
            ][best_index]
            cv_results[orientation][f"train_{metric}"] = search.cv_results_[
                f"mean_train_{metric}"
            ][best_index]

        print("Best parameters:", search.best_params_)
        print("Best CV F1:", search.best_score_)
        # Large gap between train F1 and CV F1 = overfitting
        print(
            f"Train F1: {cv_results[orientation]['train_f1_macro']:.3f} | "
            f"CV F1: {cv_results[orientation]['f1_macro_mean']:.3f}"
        )

        # ---------------- Final model for the test evaluation ----------------
        if model_name == "SVC":
            # Refit the best SVC ONCE with probability=True so predict_proba works.
            best_model = clone(search.best_estimator_)
            best_model.set_params(classify__probability=True)
            best_model.fit(X_train_id_feature_mean, id_lable_train)
        else:
            best_model = search.best_estimator_

        prediction = best_model.predict(X_test_id_feature_mean)

        test_predictions[orientation] = {
            "y_true": id_lable_test,
            "y_pred": prediction,
            "patient_ids": patient_ids_test,
            "y_prob": best_model.predict_proba(X_test_id_feature_mean),
            "classes": best_model.named_steps["classify"].classes_,
        }

        reports[orientation] = classification_report(
            id_lable_test, prediction, zero_division=0
        )

    return reports, cv_results, test_predictions


def aggregation_of_results(test_predictions):
    results_dict = {}
    for orinetation, values in test_predictions.items():
        Data = values
        y_true = Data["y_true"]
        y_prob = Data["y_prob"]
        patient_id = Data["patient_ids"]
        classes = Data["classes"]

        results = pd.DataFrame(y_prob, columns=classes)

        results["patient_id"] = patient_id
        results["true_lables"] = y_true

        Grouped_id_prob = results.groupby("patient_id")[classes].mean()
        number_diagnosis = results.groupby("patient_id")["true_lables"].nunique()
        if (number_diagnosis > 1).any():
            raise ValueError("Some patients have multiple diagnostic labels!")

        true_patinet_diagnosis = results.groupby("patient_id")["true_lables"].first()
        predicted_diagnosis = Grouped_id_prob.idxmax(axis=1)
        results_dict[orinetation] = {
            "Probabilities": Grouped_id_prob,
            "True patient Diagnosis": true_patinet_diagnosis,
            "Predicted diagnosis": predicted_diagnosis,
        }


def hold_out_pipline(features, lables):
    reports = {}
    confusion_matrices = {}

    for orientation, Data in features.items():

        pipeline = Pipeline([
            ("scaling", StandardScaler()),
            ("classify", RandomForestClassifier(
                n_estimators=300,
                criterion='entropy',
                max_depth=10,
                max_features=0.1,
                min_samples_leaf=20,
                max_samples=0.5
            ))
        ])

        pipeline.fit(Data["X_train"], Data["y_train"])
        y_pred = pipeline.predict(Data["X_test"])

        reports[orientation] = classification_report(
            Data["y_test"], y_pred,
            target_names=lables,
            zero_division=0
        )

        confusion_matrices[orientation] = confusion_matrix(
            Data["y_test"],
            y_pred,
            labels=range(len(lables))
        )

    return reports, confusion_matrices

def final_training(features, labels):
    models = {}


    pipeline = Pipeline([
            ("scaling", StandardScaler()),
            ("classify", RandomForestClassifier(
                n_estimators=300,
                criterion="entropy",
                max_depth=10,
                max_features=0.1,
                min_samples_leaf=20,
                max_samples=0.5
            ))
        ])

    pipeline.fit(features, labels)
    y_pred = pipeline.predict(features)
    results = {
        "model": pipeline,
        "predictions": y_pred,
        "accuracy": accuracy_score(labels, y_pred),
        "macro_f1": f1_score(labels, y_pred, average="macro"),
        "classification_report": classification_report(
            labels, y_pred, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(labels, y_pred)
    }

    return results

def pipeline_MIL(Data_Dictionary):
    weights = {}
    results = {}
    loss_orinetation = {}
    all_attention, all_predictions = [], []
    for orientation, Data in Data_Dictionary.items():

        X_train = Data["X_train"]
        y_train = Data["y_train"]

        X_test = Data["X_test"]
        y_test = Data["y_test"]

        # Patient-level cross-validation
        groups_train = Data["patient_ids_train"]

        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

        loss_fold = {}
        loss_fold_idx = 0
        # ============================================================
        # Cross-fold Validation
        # ============================================================
        for train_idx, val_idx in cv.split(X_train, y_train, groups=groups_train):
            from sklearn.preprocessing import StandardScaler

            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]

            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]

            groups_fold_train = groups_train.iloc[train_idx]
            groups_fold_val = groups_train.iloc[val_idx]
            scaler = StandardScaler().fit(X_fold_train)
            X_fold_train = pd.DataFrame(
                scaler.transform(X_fold_train), index=X_fold_train.index
            )
            X_fold_val = pd.DataFrame(
                scaler.transform(X_fold_val), index=X_fold_val.index
            )
            mil_train_dataset = MRIDataset(
                X_fold_train, y_fold_train, groups_fold_train
            )
            mil_val_dataset = MRIDataset(X_fold_val, y_fold_val, groups_fold_val)

            train_loader = DataLoader(mil_train_dataset, batch_size=6, shuffle=True)
            val_loader = DataLoader(
                mil_val_dataset,
                batch_size=6,
            )

            input_dim = 384
            hidden_dim = 128
            inputlayer = 384
            hiddenlayer1 = 256
            hiddenlayer2 = 128
            hiddenlayer3 = 64
            torch.manual_seed(42)
            alzheimer_model = Alzheimer(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                inputlayer=inputlayer,
                hiddenlayer3=hiddenlayer3,
                hiddenlayer2=hiddenlayer2,
                hiddenlayer1=hiddenlayer1,
            )
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            optimizer = torch.optim.AdamW(
                alzheimer_model.parameters(), lr=0.001, weight_decay=0.01
            )  # lr is the learning rate

            loss_epoch = {}
            loss_epoch_val = {}
            accuracy_epoch_val = {}
            f1_score_ = {}
            # ============================================================
            # Running 20 Epochs
            # ============================================================
            for epoch in range(100):
                epoch_mean = []  # Das speichert den mean von einem epoch
                val_predictions_ = []
                val_targets_ = []
                # print(epoch)
                # ============================================================
                # Loading batches and training
                # ============================================================
                for batch_x, batch_y, batch_rows in train_loader:

                    optimizer.zero_grad()

                    class_score, attention_weights = alzheimer_model(batch_x)

                    loss = criterion(class_score, batch_y)

                    loss.backward()

                    optimizer.step()

                    epoch_mean.append(loss.item())

                loss_epoch[epoch] = np.mean(epoch_mean)
                alzheimer_model.eval()
                # ============================================================
                # Validation of the Model
                # ============================================================
                with torch.no_grad():
                    val_epoch_mean = []
                    correct_val = 0
                    total_val = 0

                    for batch_x, batch_y, batch_rows in val_loader:
                        class_score_val, attention_weights_val = alzheimer_model(
                            batch_x
                        )
                        predicted_labels = torch.argmax(class_score_val, dim=1)
                        correct_predictions = torch.sum(predicted_labels == batch_y)
                        correct_val = correct_val + correct_predictions.item()
                        total_val = total_val + batch_y.size(0)

                        loss_val = criterion(class_score_val, batch_y)
                        val_epoch_mean.append(loss_val.item())
                        val_predictions_.append(predicted_labels)
                        val_targets_.append(batch_y)

                loss_epoch_val[epoch] = np.mean(val_epoch_mean)
                val_targets = torch.cat(val_targets_, dim=0)
                val_predictions = torch.cat(val_predictions_, dim=0)
                val_targets = val_targets.cpu().numpy()
                val_predictions = val_predictions.cpu().numpy()
                f1_score_[epoch] = f1_score(
                    val_targets,
                    val_predictions,
                    average="macro",
                    labels=[0, 1, 2, 3, 4, 5],
                    zero_division=0,
                )

                accuracy = correct_val / total_val
                accuracy_epoch_val[epoch] = accuracy

                alzheimer_model.train()
                # ---- Vorhersage + Attention der Validierungspatienten speichern ----
            alzheimer_model.eval()
            with torch.no_grad():
                for batch_x, batch_y, batch_rows in val_loader:
                    class_score_val, attention_weights_val = alzheimer_model(batch_x)
                    probs = torch.softmax(class_score_val, dim=1)

                    for i in range(batch_x.shape[0]):  # alle Patienten im Batch
                        slices = Data["meta_train"].loc[batch_rows[i].numpy()].copy()
                        slices["attention"] = attention_weights_val[i, :, 0].numpy()
                        slices["fold"] = loss_fold_idx
                        all_attention.append(slices)

                        all_predictions.append(
                            {
                                "patient_id": slices["patient_id"].iloc[0],
                                "label": batch_y[i].item(),
                                "pred": probs[i].argmax().item(),
                                "fold": loss_fold_idx,
                                **{f"p{c}": probs[i, c].item() for c in range(6)},
                            }
                        )
            loss_fold[loss_fold_idx] = {}
            loss_fold[loss_fold_idx]["train"] = loss_epoch
            loss_fold[loss_fold_idx]["validation"] = loss_epoch_val
            loss_fold[loss_fold_idx]["validation_accuracy"] = accuracy_epoch_val
            loss_fold[loss_fold_idx]["validation_f1"] = f1_score_
            loss_fold[loss_fold_idx]["Accuracy"] = torchmetrics.Accuracy(
                task="multiclass", num_classes=6
            )
            loss_fold_idx = loss_fold_idx + 1
        loss_orinetation[orientation] = loss_fold
    plot_f1_score(loss_orinetation)
    return (
        loss_orinetation,
        pd.DataFrame(all_predictions),
        pd.concat(all_attention, ignore_index=True),
    )


def plot_f1_score(loss_orientation):

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    for orientation_index, (ori, folds) in enumerate(loss_orientation.items()):

        for fold_index, data in folds.items():

            # Epoch numbers
            epochs = list(data["train"].keys())

            # Training Loss
            axes[0, orientation_index].plot(
                epochs, list(data["train"].values()), label=f"Fold {fold_index + 1}"
            )

            # Validation Loss
            axes[1, orientation_index].plot(
                epochs,
                list(data["validation"].values()),
                label=f"Fold {fold_index + 1}",
            )

            # Validation Macro F1
            axes[2, orientation_index].plot(
                epochs,
                list(data["validation_f1"].values()),
                label=f"Fold {fold_index + 1}",
            )

        # Titles
        axes[0, orientation_index].set_title(f"{ori}: Training Loss")
        axes[1, orientation_index].set_title(f"{ori}: Validation Loss")
        axes[2, orientation_index].set_title(f"{ori}: Validation Macro F1")

        # Axis labels and legends
        for row in range(3):
            axes[row, orientation_index].set_xlabel("Epoch")
            axes[row, orientation_index].legend(loc="best")
            axes[row, orientation_index].grid(True)

        axes[0, orientation_index].set_ylabel("Loss")
        axes[1, orientation_index].set_ylabel("Loss")
        axes[2, orientation_index].set_ylabel("Macro F1")
        axes[2, orientation_index].set_ylim(0, 1)

    fig.tight_layout()
    plt.show()


def sclice_independed(Data_Dictionary, model_name):
    X_train_combined = []
    X_train_groups_combined = []
    y_train_combined = []

    X_test_combined = []
    y_test_combined = []
    PATIENT_ID_TEST = []
    y_test_groups_combined = []
    for orientation, data in Data_Dictionary.items():

        X_train_combined.append(data["X_train"])
        y_train_combined.append(data["y_train"])
        X_train_groups_combined.append(data["groups_train"])

        X_test_combined.append(data["X_test"])
        y_test_combined.append(data["y_test"])
        PATIENT_ID_TEST.append(data["patient_ids_test"])

        
    X_train_combined = pd.concat(X_train_combined)
    y_train_combined = pd.concat(y_train_combined)
    X_train_groups_combined = pd.concat(X_train_groups_combined)

    X_test_combined = pd.concat(X_test_combined)
    y_test_combined = pd.concat(y_test_combined)
    PATIENT_ID_TEST = pd.concat(PATIENT_ID_TEST)
    
    reports = {}
    cv_results = {}
    test_predictions = {}

    random_state = 42  # one seed for CV splits and models

    # ---------------- Model + hyperparameter grid (built once) ----------------
    # Everything that is tuned lives ONLY in the grid, so the constructor values
    # can no longer be silently overridden.
    if model_name == "RF":
        model = RandomForestClassifier(
            bootstrap=True,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,  # GridSearchCV already runs in parallel (n_jobs=-1)
        )
        # Many highly correlated slices per patient: large leaves and small
        # bootstrap samples stop the trees from memorising individual patients.
        param_grid = {
            "classify__n_estimators": [300],
            "classify__max_features": [0.05, 0.1],
            "classify__max_depth": [None, 10],
            "classify__min_samples_leaf": [20, 50],
            "classify__max_samples": [0.2, 0.5],
            "select__k": [20, 40, 60, 80, 120, 160, 200, 300],
        }
        steps = [("select", SelectKBest(score_func=f_classif))]

    elif model_name == "SVC":
        # probability=False during the search: probability=True triggers an extra
        # internal 5-fold CV for every fit and is very slow. The probabilities
        # are computed once after the search (see "final model" below).
        model = SVC(
            probability=False, class_weight="balanced", random_state=random_state
        )
        # Small C = strong regularisation (better with few samples and 384 features).
        # The polynomial kernel was removed (degree 4 overfits easily).
        param_grid = [
            {
                "classify__kernel": ["linear"],
                "classify__C": [0.01, 0.1, 1, 10],
                "select__k": [20, 40, 60, 80, 120, 160, 200, 300],
            },
            {
                "classify__kernel": ["rbf"],
                "classify__C": [0.1, 1, 10],
                "classify__gamma": ["scale", 0.001],
                "select__k": [20, 40, 60, 80, 120, 160, 200, 300],
            },
        ]
        steps = [
            ("l2norm", Normalizer(norm="l2")),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=40)),
        ]

    elif model_name == "LR":
        # L2-regularised logistic regression: strong baseline for embeddings.
        model = LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=random_state
        )
        param_grid = {
            "classify__C": [0.001, 0.01, 0.1, 1],
            "select__k": [20, 40, 60, 80, 120, 160, 200, 300],
        }
        steps = [
            ("l2norm", Normalizer(norm="l2")),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif)),
        ]

    elif model_name == "LDA":
        # LDA with covariance shrinkage works well when n_features ~ n_samples.
        model = LinearDiscriminantAnalysis(solver="lsqr")
        param_grid = {
            "classify__shrinkage": ["auto", 0.1, 0.5, 0.9],
            "select__k": [20, 40, 60, 80, 120, 160, 200, 300],
        }
        steps = [
            ("l2norm", Normalizer(norm="l2")),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif)),
        ]

    else:
        raise ValueError(
            f"model_name must be 'RF', 'SVC', 'LR' or 'LDA', got {model_name!r}"
        )

    # L2 normalisation (per sample) -> standardisation (per feature, fitted per CV fold)
    # -> classifier. Fitting the scaler inside the pipeline avoids leakage.
    pipeline = Pipeline(steps + [("classify", model)])

    # Metrics evaluated in every fold. balanced_accuracy is robust against unequal
    # class sizes in a fold; neg_log_loss penalises over-confident wrong predictions.
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "balanced_accuracy": "balanced_accuracy",
    }
    if model_name != "SVC":  # SVC (probability=False) has no predict_proba
        scoring["neg_log_loss"] = "neg_log_loss"


        # Patient-level cross-validation: all slices of one patient stay in the same fold

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)

    search = GridSearchCV(
            estimator=pipeline,  # GridSearchCV clones it, so reusing it is safe
            param_grid=param_grid,
            scoring=scoring,
            refit="f1_macro",  # best parameters are selected by macro F1
            cv=cv,
            n_jobs=-1,
            verbose=1,
            return_train_score=True,  # needed to detect overfitting
        )
    search.fit(X_train_combined, y_train_combined, groups=X_train_groups_combined)

        # CV results of the best parameter set (same keys as before + new metrics)
    best_index = search.best_index_
    cv_results = {"best_params": search.best_params_}
    for metric in scoring:
            cv_results[f"{metric}_mean"] = search.cv_results_[
                f"mean_test_{metric}"
            ][best_index]
            cv_results[f"{metric}_std"] = search.cv_results_[
                f"std_test_{metric}"
            ][best_index]
            cv_results[f"train_{metric}"] = search.cv_results_[
                f"mean_train_{metric}"
            ][best_index]

    print("Best parameters:", search.best_params_)
    print("Best CV F1:", search.best_score_)
        # Large gap between train F1 and CV F1 = overfitting
    print(
            f"Train F1: {cv_results['train_f1_macro']:.3f} | "
            f"CV F1: {cv_results['f1_macro_mean']:.3f}"
        )

        # ---------------- Final model for the test evaluation ----------------
    if model_name == "SVC":
            # Refit the best SVC ONCE with probability=True so predict_proba works.
            # (For the SVC, predict() can differ slightly from argmax(predict_proba).)
            best_model = clone(search.best_estimator_)
            best_model.set_params(classify__probability=True)
            best_model.fit(X_train_combined, y_train_combined)
    else:
            best_model = search.best_estimator_

    prediction = best_model.predict(X_test_combined)

    test_predictions = {
            "y_true": y_test_combined,
            "y_pred": prediction,
            "patient_ids": PATIENT_ID_TEST.to_numpy(),
            "y_prob": best_model.predict_proba(X_test_combined),
            "classes": best_model.named_steps["classify"].classes_,
        }

    reports = classification_report(
            y_test_combined, prediction, zero_division=0
        )

    return reports, cv_results, test_predictions
