import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

from The_Great_Indian_Hiring.src.config import clsTrain, clsTest



class clsRegressionFeatureSelection:
    def __init__(self, train=clsTrain(), test=clsTest(clsTrain), topNFeatures=10):
        self.train = train
        self.test = test
        self.topNFeatures = topNFeatures

    def funcFeatureSelectionUsingFRegression(self):
        """
        This function selects the top N features using chi2 method.
        """
        print(f'\tEvaluating top {self.topNFeatures} best features')
        bestFeatures = SelectKBest(score_func=f_regression, k=self.topNFeatures)
        fit = bestFeatures.fit(self.train.data, self.train.target.values.ravel())
        dfScores = pd.DataFrame(fit.scores_)
        dfColumns = pd.DataFrame(self.train.data.columns)

        # concat two dataframes for better visualization
        featureScores = pd.concat([dfColumns, dfScores], axis=1)
        # naming the dataframe columns
        featureScores.columns = ['Specs', 'Score']
        # # print 10 best features
        # print(featureScores.nlargest(10, 'Score'))
        features = self.train.data.columns
        importantFeatures = list(featureScores.nlargest(self.topNFeatures, 'Score')['Specs'])
        columnsToDrop = set(features).difference(importantFeatures)

        print('\tPerforming Feature Selection on Train dataset')
        self.train.data = self.train.data[self.train.data.columns[~self.train.data.columns.isin(columnsToDrop)]]
        print('\tPerforming Feature Selection on Test dataset')
        self.test.data = self.test.data[self.test.data.columns[~self.test.data.columns.isin(columnsToDrop)]]


if __name__ == '__main__':
    from The_Great_Indian_Hiring.src.preprocessing import clsPreProcessing

    preProcessing = clsPreProcessing()
    train = clsTrain()
    test = clsTest(train)
    train.COLUMNSTODROP = ['InvoiceDate']
    train, test, dictPreProcessingFunctionsTrain, dictPreProcessingFunctionsTest = preProcessing.funcPreprocessing(train, test)

    print('\n-----------------Train Set--------------------------')
    print(train.data.sample(5).T)
    print('\n-----------------Test Set--------------------------')
    print(test.data.sample(5).T)