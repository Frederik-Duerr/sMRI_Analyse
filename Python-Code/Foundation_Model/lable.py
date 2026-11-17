def label_change(dataframe):
    mapping = {
        'CN' : 0,
        'SMC' : 1,
        'EMCI' : 2,
        'MCI' : 3,
        'LMCI' : 4,
        'AD' : 5
    }
    
    dataframe['lable'] = dataframe['diagnosis'].map(mapping)

    if dataframe['lable'].isna().any():
        print(dataframe['diagnosis'].value_counts(dropna=False))

        print(dataframe['diagnosis'].unique())

        print(dataframe.loc[dataframe['lable'].isna(), 'diagnosis'].apply(repr).unique())
        raise ValueError(
            'Unknown Diagnosis in the column diagnosis found'
        )
    
    return dataframe