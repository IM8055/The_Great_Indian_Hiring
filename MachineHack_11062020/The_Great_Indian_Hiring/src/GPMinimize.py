import numpy as np

import time
import datetime
import pytz

from sklearn import model_selection
from sklearn import metrics

from functools import partial
from skopt import gp_minimize

from The_Great_Indian_Hiring.src.config import clsTrain, clsEngineeredSet
from The_Great_Indian_Hiring.src.preprocessing import clsPreProcessing
from The_Great_Indian_Hiring.src.loggging import clsLogFile
from The_Great_Indian_Hiring.src.GPMinimizeConstants import PARAMNAMES, PARAMSPACE


from sklearn.ensemble import RandomForestRegressor


class clsGPMinimize:

    def __init__(self, model, trainset, fncpParamSpace=[], fncpParamNames=[],
                 fncpmetrics=metrics.mean_squared_log_error):
        """
        This function chooses the best parameters for RF Regressor and logs the result.

          Parameters:
              model: ML Model
              trainset (clsTrain or clsEngineeredSet): Train dataset
              fncpParamSpace (list): list of parameter space to search.
              fncpParamNames (list): Hyper parameters of the model that needs to be tuned
              fncpmetrics (metrics): metrics that need to be optimized

          Returns:
              Output - Instantiate a clsGPMinimize object
        """

        self.model = model
        self.trainset = trainset.data
        self.targetset = trainset.target
        self.ParamSpace = fncpParamSpace
        self.ParamNames = fncpParamNames
        self.metrics = fncpmetrics

    # Double underscore tells that it is a private method
    def __optimize(self, params, fncpkfoldvalue=5):
        params = dict(zip(self.ParamNames, params))

        model = self.model(**params)
        kf = model_selection.KFold(n_splits=fncpkfoldvalue)
        X = self.trainset.values
        y = self.targetset.values

        accuracies = []
        for idx in kf.split(X=X, y=y):
            train_idx, test_idx = idx[0], idx[1]
            xtrain = X[train_idx]
            ytrain = y[train_idx].ravel()
            xtest = X[test_idx]
            ytest = y[test_idx].ravel()

            model.fit(xtrain, ytrain)
            pred = model.predict(xtest)
            pred = np.abs(pred)
            fold_acc = np.sqrt(self.metrics(ytest, pred))
            accuracies.append(fold_acc)

            return np.mean(accuracies)

    def funcGPMinimize(self, n_calls=10, fncpkfoldvalue=5, preProcessingComment=[], mdlComment=[], gitCommentID=000):
        """
        This function returns and logs the optimized parameters found using GP Minimize algorithm after searching
        through the hyper parameters mentioned in the ParamNames.

            Parameters:
                 n_calls (int) : Number of iterations. Minimum number is 10
                 fncpkfoldvalue (int): No Folds
                 preProcessingComment (list)(optional) : Pre processing comments.
                 mdlComment (list)(optional) : Model comments
                 gitCommentID (int)(optional) :Git Comment ID

            Returns:
                GP Minimize object.

            Note:
            Refer https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html
            for the various attributes associated with this returned object.
        """

        start_time = time.time()
        current_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%d-%m-%Y %H:%M:%S")
        print(f'\nModel training start time : {current_time}\n')
        optimize_function = partial(
            self.__optimize,
            fncpkfoldvalue=fncpkfoldvalue
        )

        result = gp_minimize(
            optimize_function,
            dimensions=self.ParamSpace,
            n_calls=n_calls,
            n_random_starts=10,
            verbose=30
        )

        print(f'Minutes taken to complete training : {(time.time() - start_time) / 60}')

        mdlHyperParameters = dict(zip(self.ParamNames, result.x))

        print('\nHyper Parameters are:\n')
        print(mdlHyperParameters)

        # Logging Hyper Parameter
        log = clsLogFile(ModelName=str(type(self.model()).__name__),
                         ModelFeatures=self.trainset.columns,
                         ModelHyperParameters=mdlHyperParameters,
                         ModelCVScore=result.fun,
                         ModelFileName='GP Minimize Hyper Parameters',
                         OutputFileName='None',
                         ModelPreProcessingSteps=preProcessingComment,
                         comments=mdlComment,
                         gitCommentID=gitCommentID
                         )
        log.funcLogging()

        return result, mdlHyperParameters


if __name__ == '__main__':

    preProcessing = clsPreProcessing()
    # # New File
    # train = clsTrain()
    # train, test, dictPreProcessingFunctionsTrain, dictPreProcessingFunctionsTest = preProcessing.funcPreprocessing(
    #     train, test)

    # # Existing File
    engineeredSet = clsEngineeredSet(trainfilename='FeatureEngineeredTrainSet.csv',
                                     testfilename='FeatureEngineeredTestSet.csv',
                                     targetfilename='FeatureEngineeredTargetSet.csv')

    gpMinimize = clsGPMinimize(model=RandomForestRegressor,
                               trainset=engineeredSet,
                               fncpParamSpace=PARAMSPACE['RandomForestRegressor'],
                               fncpParamNames=PARAMNAMES['RandomForestRegressor'],
                               fncpmetrics=constMETRICS)
    res = gpMinimize.funcGPMinimize(n_calls=10,
                                    fncpkfoldvalue=5,
                                    preProcessingComment=['Testing GP minimize'],
                                    mdlComment=['Random Forest Regressor Model Test'],
                                    gitCommentID=000)
    print('Hyper Parameters are:')
    print(res.x)


