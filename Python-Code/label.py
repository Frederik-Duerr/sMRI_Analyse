import pandas as pd

def label_change(dataframe):
    diagnose = dataframe['diagnosis']
    dataframe['label'] = diagnose.map({
        'AD' : 0,
        'CN' : 1,
        'EMCI' : 2,
        'LMCI' : 3,
        'MCI' : 4,
        'SMC' : 5
    })
    return dataframe