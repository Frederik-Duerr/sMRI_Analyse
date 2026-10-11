
def index_tracking(image_path_df):
    return {
        patient_id: {
            orientation: group.index.tolist()
            for orientation, group in patient_data.groupby(
                'Orientation', sort=False
            )
        }
        for patient_id, patient_data in image_path_df.groupby(
            'patient_id', sort=False
        )
    }