def label_change(dataframe):
    mapping = {
        'AD' : 0,
        'CN' : 1,
        'EMCI' : 2,
        'LMCI' : 3,
        'MCI' : 4,
        'SMC' : 5
    }
    
    dataframe['lable'] = dataframe['diagnosis'].map(mapping)

    if dataframe['lable'].isna().any():
        raise ValueError(
            'Unknown Diagnosis in the column diagnosis found'
        )
    
    return dataframe