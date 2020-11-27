import pandas as pd
import numpy as np

import os

import datetime
import pytz

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PowerTransformer

from The_Great_Indian_Hiring.src.config import clsProjectConfig
from The_Great_Indian_Hiring.src.config import clsTrain, clsTest


def get_part_of_day(hour):
    """Returns the part of the day based on the hour"""
    return (
        "Morning" if 5 <= hour <= 11
        else
        "Afternoon" if 12 <= hour <= 17
        else
        "Evening" if 18 <= hour <= 22
        else
        "Night"
    )


class clsDataFrameUtilityFunctions:

    @staticmethod
    def funcCheckNull(fncpdf, boolany=True):
        """
        Function checks the null values and returns if there are any. Make the optional bool value to
        'False' if you want to return the records for which all the values is Nan.

            Parameters:
                fncpdf (pd.DataFrame) : Takes the dataframe as an input
                boolany (bool) : If True returns the records if any of the values are NaN

            Returns:
                Records whose values are NaN

        """
        if boolany:
            return fncpdf[fncpdf.isna().any(axis=1)]
        else:
            return fncpdf[fncpdf.isna().all(axis=1)]


    @staticmethod
    def funcChangeDataTypeToDateTime(fncpdf, fncplistcolumn=[]):
        """
        Changes the list of columns datatype to DateTime

            Parameters:
                fncplistcolumn (list) : List of columns whose datatype needs to be changed
                fncpdf (pd.DataFrame) : DataFrame where the columns datatype needs to be changed

            Returns:
                Returns a new dataframe after changing the datatype
        """

        for col in fncplistcolumn:
            fncpdf[col] = pd.to_datetime(fncpdf[col])
        return fncpdf

    @staticmethod
    def funcChangeDataTypeToObject(fncpdf, fncplistcolumn=[]):
        """
        Changes the list of columns datatype to object datatype

            Parameters:
                fncplistcolumn (list) : List of columns whose datatype needs to be changed
                fncpdf (pd.DataFrame): DataFrame where the columns datatype needs to be changed

            Returns:
                Returns a new dataframe after changing the datatype
        """

        for col in fncplistcolumn:
            fncpdf[col] = fncpdf[col].astype('object')
        return fncpdf

    @staticmethod
    def funcOutlierTreatment(fncpdf, fncpcolumn):
        """
        Finds the lower range and upper range values for a column and returns them

            Parameters:
            fncpcolumn (str) : Columns whose range needs to be found
            fncpdf (pd.DataFrame): DataFrame where the column is present

            Returns:
                lower_range (int) : Lower Range value
                upper_range (int) : Upper Range value
        """
        Q1 = np.percentile(fncpdf[fncpcolumn], 25)
        Q3 = np.percentile(fncpdf[fncpcolumn], 75)
        IQR = Q3 - Q1
        lower_range = Q1 - (1.5 * IQR)
        upper_range = Q3 + (1.5 * IQR)
        return lower_range, upper_range

    @staticmethod
    def funcDateTimeFeatureExtraction(fncpdf, fncpcolumn):
        """
        Extracts features such as year, month, dayofweek, quarter, is_month_start from the datetime object

            Parameters:
                fncpcolumn (str) : Datetime column name
                fncpdf (pd.DataFrame): DataFrame where the column is present

            Returns:
                Feature engineered dataframe
        """
        colName = 'f_'+fncpcolumn
        fncpdf[colName+'Year'] = pd.DatetimeIndex(fncpdf[fncpcolumn]).year
        fncpdf[colName+'Month'] = pd.DatetimeIndex(fncpdf[fncpcolumn]).month
        fncpdf[colName+'Day'] = pd.DatetimeIndex(fncpdf[fncpcolumn]).dayofweek
        fncpdf[colName + 'WeekOfYear'] = fncpdf[fncpcolumn].dt.isocalendar().week

        # fncpdf[colName+'Year'].astype(str) + fncpdf[colName+'Month'].astype(str)

        fncpdf[colName +'Weekend'] = (fncpdf[fncpcolumn].dt.weekday >= 5).astype(int)
        fncpdf[colName+'Quarter'] = pd.DatetimeIndex(fncpdf[fncpcolumn]).quarter
        fncpdf[colName+'MonthStart'] = pd.DatetimeIndex(fncpdf[fncpcolumn]).is_month_start
        fncpdf[colName + 'PartOfDay'] = pd.DatetimeIndex(fncpdf[fncpcolumn]).hour.map(get_part_of_day)


        fncpdf[colName+'MonthStart'] = fncpdf[colName+'MonthStart'].astype('object')
        fncpdf[colName + 'WeekOfYear'] = fncpdf[colName + 'WeekOfYear'].astype('int32')
        return fncpdf

    @staticmethod
    def funcValueCountCheck(fncpdf, fncpcolumn, fncpCheckCountValue=1):
        """
        This function create a bool column if the value count is greater than a specific value.

            Parameters:
                fncpcolumn (str) : Column for which we need to check the value count
                fncpdf (pd.DataFrame): DataFrame where the column is present
                fncpCheckCountValue (optional)(int) : Threshold value to mark the new column as 1.

            Returns:
                Feature engineered dataframe
        """
        tempDF = fncpdf[fncpcolumn].value_counts()
        tempList = tempDF[tempDF > fncpCheckCountValue].index.tolist()
        fncpdf['f_ValueCountCheck' + fncpcolumn] = 0
        fncpdf.loc[fncpdf[fncpcolumn].isin(tempList), ['f_ValueCountCheck' + fncpcolumn]] = 1
        del tempDF, tempList
        return fncpdf

    @staticmethod
    def funcSavingDFToCSV(fncpdf, fncpfilename, fncppathtosave=clsProjectConfig.OUTPUTPATH):
        """
        By default saves the dataframe as an .csv file in the OUTPUTPATH path specified in config.py

            Parameters:
                fncpdf (pd.DataFrame): DataFrame to be saved
                fncpfilename (str) : Filename
                fncppathtosave (str) : Path directory to save the file.

            Returns:
                None
        """
        if '.csv' not in fncpfilename:
            fncpfilename = fncpfilename+'.csv'
        fncpdf.to_csv(os.path.join(fncppathtosave, fncpfilename), index=False)
        print(f'\nSuccessfully Saved {fncpfilename} !!!')

    @staticmethod
    def funcCustomPipeLine(fncpclsdata, fncpdictFunc):
        """
        This function takes an list of functions and the class as an input and
        performs feature processing on the dataframe stored in the class object and returns the class.

            Parameters:
                fncpclsdata (dataframe object): Input class from config.py\n
                fncpdictFunc (dictionary): A dictionary of functions.

            Returns:
                fncpclsdata (dataframe object) : Dataframe object that contains processed dataframe.
        Note
        To drop columns, the key name should be 'DroppingColumns' and its values needs to be
        the list of columns that needs to be dropped.
        """

        start_time = datetime.datetime.now()
        current_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%d-%m-%Y %H:%M:%S")
        print(f'\nPreprocessing Data for {type(fncpclsdata).__name__} class {current_time}\n')
        for enum_, (key, value) in enumerate(fncpdictFunc.items()):
            print(str(enum_ + 1), key)
            if key == 'DroppingColumns':
                fncpclsdata.funcColumnsToDrop(fncpdictFunc['DroppingColumns'])
            elif key == 'CategoryCoding':
                pass
            else:
                fncpclsdata = value(fncpclsdata)
        end_time = datetime.datetime.now()
        print(f'\nTotal number of variables: {fncpclsdata.data.shape[1]}')
        print(f'Total number of rows: {fncpclsdata.data.shape[0]}')
        print(f'\nCompleted Preprocessing Data for {type(fncpclsdata).__name__} class in {(end_time - start_time)}')

        return fncpclsdata

    @staticmethod
    def funcListComparison(fncpfirstlist, fncpsecondlist):
        """
        Function takes in two list and compares them and returns a list of missing values
        present in the second list and also returns a list of additional values present in second list.

            Parameters:
                fncpfirstlist (list) : First List\n
                fncpsecondlist (list) : Second List

            Returns:
                secondListMissingValues (list) : Missing Values in second list
                secondListAdditionalValues (list) : Additional values in second list
        Note:
          First returns missing values and then the additional values
        """
        secondListMissingValues = set(fncpfirstlist).difference(fncpsecondlist)
        print("\nNumber of missing values in second list:", len(secondListMissingValues))
        secondListAdditionalValues = set(fncpsecondlist).difference(fncpfirstlist)
        print("Number of additional values in second list:", len(secondListAdditionalValues))

        return secondListMissingValues, secondListAdditionalValues

    @staticmethod
    def funcOneHotEncode(fncpclstraindata=clsTrain(), fncpclstestdata=clsTest(clsTrain)):
        """
        This function takes in the dataframe object as an input and one hot encodes
        the categorical columns.

            Parameters:
                fncpclstraindata (clsTrain) : Takes in Train class as an input
                fncpclstestdata (clsTest) : Takes in Test class as an input

            Returns:
                  fncpclstraindata (clsTrain) - One hot encoded train class
                  fncpclstestdata (clsTest) - One hot encoded test class
        """

        fncpclstraindata.data['r_Trainset'] = 1
        fncpclstestdata.data['r_Trainset'] = 0
        mergedDF = pd.concat([fncpclstraindata.data, fncpclstestdata.data])

        # listCategoryColumns = fncpclstraindata.CategoryColumns
        mergedDF = pd.get_dummies(mergedDF)
        # for col_ in listCategoryColumns:
        #     dummyDF = pd.get_dummies(mergedDF[col_])
        #     mergedDF = pd.concat([mergedDF, dummyDF], axis=1)

        print('\tOne Hot Encoding Train Set')
        fncpclstraindata.data = mergedDF.loc[mergedDF['r_Trainset']==1, :].copy()
        fncpclstraindata.data.drop(columns=['r_Trainset'], inplace=True)
        # print(f'\tTotal number of variables after OHE in Train set: {fncpclstraindata.data.shape[1]}')
        # print(f'\tTotal number of rows after OHE in Train set: {fncpclstraindata.data.shape[0]}')

        print('\tOne Hot Encoding Test Set')
        fncpclstestdata.data = mergedDF.loc[mergedDF['r_Trainset']==0, :].copy()
        fncpclstestdata.data.drop(columns=['r_Trainset'], inplace=True)
        # print(f'\tTotal number of variables after OHE in Test set: {fncpclstestdata.data.shape[1]}')
        # print(f'\tTotal number of rows after OHE in Test set: {fncpclstestdata.data.shape[0]}')

        del mergedDF

        return fncpclstraindata, fncpclstestdata

    @staticmethod
    def funcNormalizingFeature(fncpdf, fncpnorm=None):
        """
        This function normalizes the data using MinMaX Scaler.

            Parameters:
                fncpdf (pd.DataFrame)
                fncpnorm: Normalizing object. If None it would fit and transform and if not, it would only transform.

            Returns:
                norm: Normalizing object
                fncpdf: Normalized dataframe
        """
        columnNames = fncpdf.columns
        if fncpnorm is None:
            norm = MinMaxScaler().fit(fncpdf)
        else:
            norm = fncpnorm
        fncpdf = pd.DataFrame(norm.transform(fncpdf), columns=columnNames)

        return norm, fncpdf

    @staticmethod
    def funcPolynomialTransformation(fncpclstrain=clsTrain(), fncpclstest=clsTest(clsTrain),
                                 fncpPolyInt=2):
        """
        This function does polynomial transformation along with interaction features on the dataframe.
        Note that all the columns must be numerical.

            Parameters:
                 fncpclstrain (clsTrain)
                 fncpclstest (clsTest)
                 fncpPolyInt (int) : Polynomial degree.

            Return:
                fncpclstraindata
                fncpclstestdata
        """
        poly = PolynomialFeatures(fncpPolyInt)
        fncpclstrain.data = pd.DataFrame(poly.fit_transform(fncpclstrain.data))
        fncpclstest.data = pd.DataFrame(poly.fit_transform(fncpclstest.data))

        return fncpclstrain, fncpclstest

    @staticmethod
    def funcTargetTransformation(fncpTrain=clsTrain()):
        """
        This function power transform the output variable. This is used especially when the output variable is skewed.
        """
        fncpTrain.target = fncpTrain.target.values.reshape(-1, 1)
        # power transform the raw data
        power = PowerTransformer(method='yeo-johnson', standardize=True)
        fncpTrain.target = pd.DataFrame(power.fit_transform(fncpTrain.target))
        fncpTrain.dictValuesTreated['PowerTransform'] = power

        return fncpTrain

    @staticmethod
    def funcTopXFeatures(fncpClsObj, fncpColName, fncpTopXFeatures=10):
        """
        This function only encodes Top X categories and drops the original column.

            Parameters:
                fncpClsObj : This can be either clsTrain or clsTest
                fncpColName (str) : Column Name
                fncpTopXFeatures (int): Top X features to be one hot encoded.
        """
        if fncpClsObj.name=='TRAIN':
            print(f'\t Top {fncpTopXFeatures} categories OHE for {fncpColName} on train dataset')
            topX = [x for x in fncpClsObj.data[fncpColName].value_counts().sort_values(ascending=False).head(fncpTopXFeatures).index]
            for label in topX:
                fncpClsObj.data[fncpColName+'_'+str(label)] = np.where(fncpClsObj.data[fncpColName]==label, 1, 0)
            fncpClsObj.dictValuesTreated['TopXFeatures'+fncpColName] = topX
            fncpClsObj.data.drop(columns=[fncpColName], inplace=True)
        else:
            print(f'\t Top {fncpTopXFeatures} categories OHE for {fncpColName} on test dataset')
            fncpTopXList = fncpClsObj.clsTrainData.dictValuesTreated['TopXFeatures'+fncpColName]
            for label in fncpTopXList:
                fncpClsObj.data[fncpColName+'_'+str(label)] = np.where(fncpClsObj.data[fncpColName]==label, 1, 0)
            fncpClsObj.data.drop(columns=[fncpColName], inplace=True)
        return fncpClsObj


if __name__ == '__main__':
    pass
