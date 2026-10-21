def patient_Table(dataFrame):
    diagnose_counts = (
        dataFrame.groupby('patient_id')['diagnosis'].nunique()
    )
    conflictting_ids = diagnose_counts[diagnose_counts > 1].index.tolist()
    if conflictting_ids:
        raise ValueError(
        f'More Diagnosis per patient found: '
        f' {conflictting_ids[:10]}'
        )
    new_df = (
        dataFrame[['patient_id','diagnosis']]
        .drop_duplicates(subset='patient_id')
        .reset_index(drop=True)
    )
    return new_df