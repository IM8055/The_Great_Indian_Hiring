import pandas as pd
import numpy as np

import os

import datetime
import pytz

import pickle

from The_Great_Indian_Hiring.src.config import clsProjectConfig, clsTrain
from The_Great_Indian_Hiring.src.constants import constTARGETCOLUMN
from The_Great_Indian_Hiring.src.utils import clsDataFrameUtilityFunctions


class clsPrediction:

    def __init__(self, traindata, targetdata, testdata, train=clsTrain()):
        """
            Parameters:
                traindata (DataFrame)
                targetdata (DataFrame)
                testdata (DataFrame)
                train (clsTrain)(optional): Used only when we transform the output variable.

        """
        self.traindata = traindata
        self.targetdata = targetdata
        self.testdata = testdata
        self.train = train
        self.currentTime = str(datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%d_%m_%Y_%H%M%S"))


    def funcSaveFile(self, funcpdataframe=pd.DataFrame()):
        """
            Parameters:
                funcpdataframe (pd.DataFrame()) : Dataframe to be saved
        """
        dfUtilityFunction = clsDataFrameUtilityFunctions()

        outputFileName = 'Output_'+self.currentTime+'.csv'
        outputFile = os.path.join(clsProjectConfig.OUTPUTPATH, outputFileName)
        dfUtilityFunction.funcSavingDFToCSV(funcpdataframe, outputFile)
        return outputFileName

    def funcLoadModelAndPredict(self, funcpfilename, fncponlypositive=True):
        """
        This function load the model and saves the output.

            Parameters:
                funcpfilename (str) = Model filename\n
                fncponlypositive (bool) = If true np.abs would be applied to the predicted values

            Returns:
                outputFileName (str)
        """

        dfComplete = pd.DataFrame()
        model = pickle.load(open(os.path.join(clsProjectConfig.MODELPATH, funcpfilename), 'rb'))
        predComplete = model.predict(self.testdata)
        if fncponlypositive:
            predComplete = np.abs(predComplete)
        dfComplete[constTARGETCOLUMN] = pd.DataFrame(predComplete, columns=[constTARGETCOLUMN])[constTARGETCOLUMN].values
        outputFileName = self.funcSaveFile(dfComplete)
        print(f'Prediction file {outputFileName} successfully saved in the OUTPUTPATH !!!')

        return outputFileName

    def funcPredict(self, fncpmodel, fncponlypositive=True):
        """
        This function
            1) Fits the model.
            2) Saves the model in the MODELPATH.
            3) Saves the predicted values in OUTPUTPATH.

            Parameters:
                fncpmodel (model): Model that needs to be used for prediction
                fncponlypositive (bool): If true np.abs would be applied to the predicted values

            Returns:
                modelFileName
                outputFileName
        """

        # Fitting Model
        print(f'\nFitting Model {type(fncpmodel).__name__} {self.currentTime}')
        print('\nModel Parameters Used for Fitting the model on complete data:')
        print(fncpmodel.get_params())
        fncpmodel.fit(self.traindata.values, self.targetdata.values.ravel())

        # Saving Model
        print('\nSaving Model')
        dfComplete = pd.DataFrame()
        modelName = str(type(fncpmodel).__name__)
        modelFileName = modelName+'_'+self.currentTime+'.sav'
        pickle.dump(fncpmodel, open(os.path.join(clsProjectConfig.MODELPATH, modelFileName), 'wb'))
        print(f'\tSaved the Model {modelFileName} in the MODELPATH !!!')

        print('Loading the model')
        model = pickle.load(open(os.path.join(clsProjectConfig.MODELPATH, modelFileName), 'rb'))

        print('Predicting values')
        predComplete = model.predict(self.testdata.values)
        if fncponlypositive:
            predComplete = np.abs(predComplete)
        # Checks if we have power transformed the target variable. The key 'PowerTransform' is added to the dict
        # only when we power transform target variable. Check class TargetPreprocessing in featurepreprocessing.py
        if 'PowerTransform' in self.train.dictValuesTreated.keys():
            power = self.train.dictValuesTreated['PowerTransform']
            predComplete = power.inverse_transform(predComplete.reshape(-1, 1))
        dfComplete[constTARGETCOLUMN] = pd.DataFrame(predComplete, columns=[constTARGETCOLUMN])[constTARGETCOLUMN].values
        outputFileName = self.funcSaveFile(funcpdataframe=dfComplete)
        print(f'\tPrediction file {outputFileName} successfully saved in the OUTPUTPATH')

        return modelFileName, outputFileName


if __name__ == '__main__':

    print(str(datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%d_%m_%Y_%H%M%S")))