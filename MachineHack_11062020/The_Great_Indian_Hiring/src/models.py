import numpy as np
import pandas as pd

import datetime
import pytz

from sklearn import metrics
from sklearn.model_selection import train_test_split

from The_Great_Indian_Hiring.src.config import clsTrain, clsTest, clsEngineeredSet
from The_Great_Indian_Hiring.src.preprocessing import clsPreProcessing
from The_Great_Indian_Hiring.src.loggging import clsLogFile

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

from sklearn.model_selection import KFold


class clsCompareModelsRegression:
    def __init__(self, traindata):
        """
            Parameters:
                traindata (clsTrain or clsEngineeredSet)
        """
        self.trainData = traindata
        self.modelsToCompare = {
            'RandomForestRegressor': RandomForestRegressor(),
            'KNeighborsRegressor': KNeighborsRegressor(),
            'LinearRegression': LinearRegression(),
            'Lasso': linear_model.Lasso(),
            'Ridge': linear_model.Ridge(),
            'SGDRegressor': linear_model.SGDRegressor(),
            'HuberRegressor': linear_model.HuberRegressor(),
            'PassiveAggressiveRegressor': linear_model.PassiveAggressiveRegressor(),
            'TheilSenRegressor': linear_model.TheilSenRegressor(),
            'RANSACRegressor': linear_model.RANSACRegressor(),
            'ElasticNet': linear_model.ElasticNet()
        }

    def funcCompareModels(self, fncpModelPreprocessingStepComment=[], fncpGeneralComments=[], fncpGitCommentID=000,
                          fncpKFoldValue=5):
        """
        This function compares the models and logs the model that has the best score in logging file.

            Returns:
                Compared Models dataframe
        """
        print('\nPreparing to compare models')
        compare_models = {'Model': [],
                          'Score': [],
                          'RMSE': [],
                          'RMSLE': []
                          }
        # Train Test Split
        # Xtrain, Xtest, ytrain, ytest = train_test_split(self.trainData.data, self.trainData.target, test_size=0.30,
        #                                                 random_state=12345)

        BESTSCORE = np.inf
        for enum_, (key, value) in enumerate(self.modelsToCompare.items()):
            SCORE = []
            RMSE = []
            RMSLE = []
            currentTime = str(datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%d-%m-%Y %H:%M:%S"))
            print(f'{str(enum_ + 1)}. Fitting {key} {currentTime}')
            model = value
            kf = KFold(n_splits=fncpKFoldValue, shuffle=False, random_state=12345)
            for count, (train_index, test_index) in enumerate(kf.split(self.trainData.data)):
                print(f'\tFitting and predicting for fold {count+1}')
                Xtrain = self.trainData.data.loc[train_index, :]
                ytrain = self.trainData.target.loc[train_index, :]
                Xtest = self.trainData.data.loc[test_index, :]
                ytest = self.trainData.target.loc[test_index, :]
                model.fit(X=Xtrain, y=ytrain.values.ravel())
                Y_pred = np.abs(model.predict(Xtest))
                SCORE.append(round(model.score(Xtest, ytest), 2))
                RMSE.append(round(np.sqrt(metrics.mean_squared_error(ytest, Y_pred)), 4))
                RMSLE.append(round(np.sqrt(metrics.mean_squared_log_error(np.abs(ytest), Y_pred)), 4))
            compare_models['Model'].append(str(key))
            compare_models['Score'].append(np.mean(SCORE))
            compare_models['RMSE'].append(np.mean(RMSE))
            compare_models['RMSLE'].append(np.mean(RMSLE))
            if BESTSCORE > np.mean(RMSE):
                BESTSCORE = np.mean(RMSE)
                BESTMODEL = model
                BESTHYPERPARAMETER = model.get_params()

        # Logging
        log = clsLogFile(ModelName=str(type(BESTMODEL).__name__),
                         ModelFeatures=self.trainData.data.columns,
                         ModelHyperParameters=BESTHYPERPARAMETER,
                         ModelCVScore=BESTSCORE,
                         ModelFileName='Model Comparison',
                         OutputFileName='None',
                         ModelPreProcessingSteps=fncpModelPreprocessingStepComment,
                         comments=fncpGeneralComments,
                         gitCommentID=fncpGitCommentID
                         )
        log.funcLogging()
        print('Best Model log saved after comparing the Regression models')
        return pd.DataFrame(compare_models)


if __name__ == '__main__':
    # # New Dataset
    preProcessing = clsPreProcessing()
    train = clsTrain()
    train.COLUMNSTODROP = ['InvoiceDate', 'CustomerID']
    test = clsTest(train)
    topFeatures = 17
    train, test, dictPreProcessingFunctionsTrain, dictPreProcessingFunctionsTest = preProcessing.funcPreprocessing(
        train, test, fncpTopFeatures=topFeatures)

    # # Existing Dataset
    # engineeredSet = clsEngineeredSet(trainfilename='FeatureEngineeredTrainSet.csv',
    #                                  testfilename='FeatureEngineeredTestSet.csv',
    #                                  targetfilename='FeatureEngineeredTargetSet.csv')


    compareModels = clsCompareModelsRegression(train)

    GITCOMMENTID = 519
    COMMENT = [f'1. Feature selection - Top {topFeatures}\n',
               '2. Model comparison after target transformation\n',
               '3. Removed "InvoiceDate" and "CustomerID"',
               '4. Top X Categories OHE "Description" column\n',
               '5. Implemented target transformation\n',
               '6. Removed quantity outlier treatment\n',
               '7. Added CountryID Aggregation\n',
               '8. Added CountryID bucketing']
    GENERALCOMMENT = ['Model Comparison']
    comparedModelsDF = compareModels.funcCompareModels(fncpModelPreprocessingStepComment=COMMENT,
                                                       fncpGeneralComments=GENERALCOMMENT,
                                                       fncpGitCommentID=GITCOMMENTID)
    print(comparedModelsDF)
