import os
import pandas as pd
from The_Great_Indian_Hiring.src.constants import constPROJECTPATH, constTRAINFILENAME, constTESTFILENAME, constTARGETCOLUMN, COLUMNSTODROP
import time


class clsProjectConfig:
    PROJECTPATH = constPROJECTPATH
    INPUTPATH = os.path.join(PROJECTPATH, "input")
    OUTPUTPATH = os.path.join(PROJECTPATH, "output")
    MODELPATH = os.path.join(PROJECTPATH, "models")
    LOGPATH = os.path.join(PROJECTPATH, "logs")
    TRAINFILENAME = constTRAINFILENAME
    TESTFILENAME = constTESTFILENAME


    def __init__(self):
        self.data = pd.DataFrame()
        self.Columns = self.data.columns
        self.CategoryColumns = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        self.NumericalColumns = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    def funcColumnsToDrop(self, fncplistcolumn=[]):
        """
        Drops the columns specified in the list if the columns are available in the dataframe

        :fncpColumnList - List of columns to be dropped
        :fncpDf - DataFrame where the columns to be dropped
        """

        for col in fncplistcolumn:
            if col in self.data:
                self.data = self.data.drop([col], axis=1)
        # self.Columns = self.data.columns

    @property
    def Columns(self):
        """
        Returns the actual columns in the dataframe after adding or deleting columns
        """
        return self.data.columns

    @property
    def CategoryColumns(self):
        """
        Returns the category columns after changing datatype
        """
        return self.data.select_dtypes(include=['object', 'category']).columns.tolist()

    @property
    def NumericalColumns(self):
        """
        Returns the category columns after changing datatype
        """
        return self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()


class clsTrain(clsProjectConfig):

    dictValuesTreated = {}
    COLUMNSTODROP = COLUMNSTODROP

    def __init__(self):
        self.completeData = pd.read_csv(os.path.join(clsProjectConfig.INPUTPATH, clsProjectConfig.TRAINFILENAME))
        self.completeData = self.completeData.sample(frac=1).reset_index(drop=True)  # Shuffling the dataframe
        self.TARGETCOLUMN = constTARGETCOLUMN
        self.data = self.completeData.drop(columns=self.TARGETCOLUMN)
        self.target = self.completeData.loc[:, [self.TARGETCOLUMN]]
        self.name = 'TRAIN'


class clsEngineeredSet(clsProjectConfig):

    def __init__(self, trainfilename, testfilename, targetfilename=''):
        if '.csv' not in trainfilename:
            trainfilename = trainfilename+'.csv'
        if '.csv' not in testfilename:
            trainfilename = testfilename + '.csv'
        if '.csv' not in targetfilename and targetfilename != '':
            trainfilename = targetfilename+'.csv'

        if targetfilename != '':
            print('\nReading Feature Engineered Files')
            time.sleep(2)
            self.TARGETCOLUMN = constTARGETCOLUMN
            self.data = pd.read_csv(os.path.join(clsProjectConfig.OUTPUTPATH, trainfilename))
            self.target = pd.read_csv(os.path.join(clsProjectConfig.OUTPUTPATH, targetfilename))
            self.testdata = pd.read_csv(os.path.join(clsProjectConfig.OUTPUTPATH, testfilename))
            self.completeData = pd.concat([self.data, self.target], axis=1)
            print('Loaded Feature Engineered Files')
        else:
            print('\nReading Feature Engineered Files')
            time.sleep(1)
            self.TARGETCOLUMN = constTARGETCOLUMN
            self.completeData = pd.read_csv(os.path.join(clsProjectConfig.OUTPUTPATH, trainfilename))
            self.data = self.completeData.drop(columns=self.TARGETCOLUMN)
            self.target = self.completeData.loc[:, [self.TARGETCOLUMN]]
            self.testdata = pd.read_csv(os.path.join(clsProjectConfig.OUTPUTPATH, testfilename))
            print('Loaded Feature Engineered Files')


class clsTest(clsProjectConfig):

    def __init__(self, clsTrainData):
        self.data = pd.read_csv(os.path.join(clsProjectConfig.INPUTPATH, clsProjectConfig.TESTFILENAME))
        self.clsTrainData = clsTrainData
        self.name = 'TEST'
