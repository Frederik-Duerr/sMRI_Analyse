import pandas as pd
def evaluation_fold(loss_orientations):

    orientation_results = []

    for orientation, loss_fold in loss_orientations.items():

        for fold, epoch in loss_fold.items():

            fold_data = {}

            f1_values = epoch['validation_f1']

            fold_data['Orientation'] = orientation
            fold_data['Fold'] = fold + 1

            fold_data['Final F1 (Epoch30)'] = f1_values[len(f1_values)-1]

            fold_data['Best F1'] = max(f1_values.values())

            fold_data['Best Epoch'] = max(
                f1_values,
                key=f1_values.get
            ) + 1

            orientation_results.append(fold_data)

    return pd.DataFrame(orientation_results)

def sumariese_cv(orientation_results):

    orientation = ['axial', 'coronal', 'sagittal']

    summary = {}
    ories = {}

    for ori in orientation:

        if ori == 'axial':
            axial = orientation_results.loc[
                orientation_results['Orientation'] == ori
            ]
            ories[ori] = axial

        elif ori == 'coronal':
            coronal = orientation_results.loc[
                orientation_results['Orientation'] == ori
            ]
            ories[ori] = coronal

        else:
            sagittal = orientation_results.loc[
                orientation_results['Orientation'] == ori
            ]
            ories[ori] = sagittal

    for ori, values in ories.items():

        mid_resulst = {}

        mid_resulst['mean'] = values['Best F1'].mean()
        mid_resulst['std'] = values['Best F1'].std()

        summary[ori] = mid_resulst

    return summary
        
        