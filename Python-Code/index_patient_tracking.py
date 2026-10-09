
def index_tracking(y_train, y_test, X_train_image_path, X_test_image_path):
    dict_of_all_orientations_and_orientations = {

    }
    # For each orientation we will recive the indices of each patient id 
    axial_images_X_test = X_test_image_path.loc[X_test_image_path['Orientation']=='axial']
    indices_per_patient_X_test_axial = axial_images_X_test.groupby('patient_id').groups
    print(indices_per_patient_X_test_axial.keys())
    dict_of_all_orientations_and_orientations.update(indices_per_patient_X_test_axial)

    sagittal_images_X_test = X_test_image_path.loc[X_test_image_path['Orientation']=='sagital']
    indices_per_patient_X_test_sagittal = sagittal_images_X_test.groupby('patient_id').groups
    print(indices_per_patient_X_test_sagittal.keys())

    coronal_images_X_test = X_test_image_path.loc[X_test_image_path['Orientation']=='coronal']
    indices_per_patient_X_test_coronal = coronal_images_X_test.groupby('patient_id').groups
    print(indices_per_patient_X_test_coronal.keys())

    axial_images_X_train= X_train_image_path.loc[X_train_image_path['Orientation']=='axial']
    indices_per_patient_X_train_axial = axial_images_X_train.groupby('patient_id').groups

    saggital_images_X_train= X_train_image_path.loc[X_train_image_path['Orientation']=='saggital']
    indices_per_patient_X_train_axial = saggital_images_X_train.groupby('patient_id').groups

    coronal_images_X_train= X_train_image_path.loc[X_train_image_path['Orientation']=='coronal']
    indices_per_patient_X_train_coronal = coronal_images_X_test.groupby('patient_id').groups        





    return dict_of_all_orientations_and_orientations