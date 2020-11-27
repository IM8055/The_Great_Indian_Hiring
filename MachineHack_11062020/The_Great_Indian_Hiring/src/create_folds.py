import pandas as pd
import os
from sklearn import model_selection
from The_Great_Indian_Hiring.src.config import clsProjectConfig


def funcLoadData(folds=5):

    '''
    This function reads a csv file and returns a dataframe with a Kfold number variable

    :param folds: Number of folds to split the dataset.
    :return: dataframe
    '''

    df = pd.read_csv(os.path.join(clsProjectConfig.INPUTPATH, 'Train.csv'))
    df['kfold'] = -1

    df = df.sample(frac=1).reset_index(drop=True)
    kf = model_selection.KFold(n_splits=folds)
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_,'kfold'] = fold
    return df


if __name__ == '__main__':
    df = funcLoadData()
    print(df.sample(20))

