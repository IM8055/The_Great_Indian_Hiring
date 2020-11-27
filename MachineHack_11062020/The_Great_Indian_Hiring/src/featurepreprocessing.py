import pandas as pd
import numpy as np

from The_Great_Indian_Hiring.src.utils import clsDataFrameUtilityFunctions
from The_Great_Indian_Hiring.src.projectutils import clsProjectUtilityFunction
from The_Great_Indian_Hiring.src.config import clsTrain, clsTest

dfUtilityFunc = clsDataFrameUtilityFunctions()
dfProjectUtilityFunc = clsProjectUtilityFunction()


class BasicCheck:

    @staticmethod
    def funcTrainBasicChecksAndFilters(fncpclstraindata=clsTrain()):
        """
        This function performs basic checks on the train dataframe.
        """
        # Filters out records whose prices are equal to and less than zero.
        print('\tBefore Removing Duplicates', fncpclstraindata.completeData.shape)
        fncpclstraindata.completeData = fncpclstraindata.completeData.drop_duplicates()
        print('\tAfter Removing Duplicates', fncpclstraindata.completeData.shape)
        fncpclstraindata.completeData = fncpclstraindata.completeData.loc[
                                        fncpclstraindata.completeData['UnitPrice'] > 0, :].reset_index(drop=True)
        fncpclstraindata.data = fncpclstraindata.completeData.loc[fncpclstraindata.completeData['UnitPrice'] > 0,
                                                                  fncpclstraindata.completeData.columns != fncpclstraindata.TARGETCOLUMN]
        fncpclstraindata.target = fncpclstraindata.completeData.loc[
            fncpclstraindata.completeData['UnitPrice'] > 0, ['UnitPrice']]

        return fncpclstraindata


class FeatureQuantity:

    @staticmethod
    def funcTrainQuantityFeatureProcessing(fncpclstrain=clsTrain()):
        """
        Takes the train class and preprocesses 'Quantity' variable and returns the class
        """
        fncpclstrain.data['f_ProductReturned'] = 0
        fncpclstrain.data.loc[fncpclstrain.data['Quantity'] < 0, ['f_ProductReturned']] = 1

        #                                               Negative Quantity
        fncpclstrain.data.loc[fncpclstrain.data['Quantity'] < 0, ['Quantity']] = fncpclstrain.data['Quantity'] * -1
        fncpclstrain.dictValuesTreated['TreatedNegativeQuantity'] = 'Yes'

        #                                                Binning
        # Note it uses pd.qcut
        fncpclstrain.data['f_QuantityBins'], bins = pd.qcut(x=fncpclstrain.data['Quantity'],
                                                            q=5,
                                                            retbins=True,
                                                            labels=['Micro', 'Small', 'Medium', 'Large', 'vLarge'])
        fncpclstrain.data = dfUtilityFunc.funcChangeDataTypeToObject(fncpclstrain.data, ['f_QuantityBins'])
        bins = np.concatenate(([-np.inf], bins[1:-1], [np.inf]))
        fncpclstrain.dictValuesTreated['f_QuantityBins'] = bins

        #                                              Outlier
        # Dealing Outlier by capping the outlier values to upper and lower range
        lowerRange, upperRange = dfUtilityFunc.funcOutlierTreatment(fncpclstrain.data, 'Quantity')
        fncpclstrain.data.loc[fncpclstrain.data['Quantity'] < lowerRange, ['Quantity']] = lowerRange
        fncpclstrain.data.loc[fncpclstrain.data['Quantity'] > upperRange, ['Quantity']] = upperRange
        #
        # fncpclstrain.dictValuesTreated['Quantity'] = [lowerRange, upperRange]

        return fncpclstrain

    @staticmethod
    def funcTestQuantityFeatureProcessing(fncpclstest=clsTest(clsTrain)):
        """
        Takes the test class and preprocesses 'Quantity' variable and returns the class
        """
        fncpclstest.data['f_ProductReturned'] = 0
        fncpclstest.data.loc[fncpclstest.data['Quantity'] < 0, ['f_ProductReturned']] = 1

        if 'TreatedNegativeQuantity' in fncpclstest.clsTrainData.dictValuesTreated.keys():
            fncpclstest.data.loc[fncpclstest.data['Quantity'] < 0, ['Quantity']] = fncpclstest.data['Quantity'] * -1

        #                                                Binning
        # Note it uses pd.qcut
        fncpclstest.data['f_QuantityBins'] = pd.cut(x=fncpclstest.data['Quantity'],
                                                    bins=fncpclstest.clsTrainData.dictValuesTreated[
                                                        'f_QuantityBins'],
                                                    labels=['Micro', 'Small', 'Medium', 'Large', 'vLarge'])
        fncpclstest.data = dfUtilityFunc.funcChangeDataTypeToObject(fncpclstest.data, ['f_QuantityBins'])
        #
        #                                               Outlier

        # # Dealing Outlier by capping the outlier values to upper and lower range
        if 'Quantity' in fncpclstest.clsTrainData.dictValuesTreated.keys():
            lowerRange = fncpclstest.clsTrainData.dictValuesTreated['Quantity'][0]
            upperRange = fncpclstest.clsTrainData.dictValuesTreated['Quantity'][1]
            fncpclstest.data.loc[fncpclstest.data['Quantity'] < lowerRange, ['Quantity']] = lowerRange
            fncpclstest.data.loc[fncpclstest.data['Quantity'] > upperRange, ['Quantity']] = upperRange

        return fncpclstest


class FeatureInvoiceDate:

    @staticmethod
    def funcInvoiceDateDateFeatureExtraction(fncpclsdata=clsTrain()):
        """
        This class is common for both train data and test data. It takes class as an input and returns a class after
        preprocessing 'InvoiceDate' feature
        """

        # Changing Object Data type to DateTime Data type
        fncpclsdata.data = dfUtilityFunc.funcChangeDataTypeToDateTime(fncpclsdata.data, ['InvoiceDate'])

        # Extracting Date Features
        fncpclsdata.data = dfUtilityFunc.funcDateTimeFeatureExtraction(fncpclsdata.data, 'InvoiceDate')

        # fncpclsdata.data = clsProjectUtilityFunction.funcColumnAggregation(fncpclsdata.data,
        #                                                                    fncpcolumn='CustomerID',
        #                                                                    fncpNumericCol=['Quantity'],
        #                                                                    fncpCategoricalCol=['InvoiceNo',
        #                                                                                        'StockCode'],
        #                                                                    fncpmonthcol=[])

        return fncpclsdata


class FeatureCustomerID:

    @staticmethod
    def funcCustomerIDFeatureProcessing(fncpclsdata):
        """
        This class is common for both train data and test data. It takes class as an input and returns a class after
        preprocessing 'CustomerID' feature
        """

        fncpclsdata.data = clsProjectUtilityFunction.funcColumnAggregation(fncpclsdata.data,
                                                                           fncpcolumn='CustomerID',
                                                                           fncpNumericCol=['Quantity'],
                                                                           fncpCategoricalCol=['InvoiceNo',
                                                                                               'StockCode'],
                                                                           fncpmonthcol=[])

        return fncpclsdata


class FeatureInvoiceNo:

    @staticmethod
    def funcInvoiceNoFeatureProcessing(fncpclsdata):
        """
        This class is common for both train data and test data. It takes class as an input and returns a class after
        preprocessing 'InvoiceNo' feature
        """

        fncpclsdata.data = clsProjectUtilityFunction.funcColumnAggregation(fncpclsdata.data, fncpcolumn='InvoiceNo',
                                                                           fncpNumericCol=['Quantity'],
                                                                           fncpCategoricalCol=['StockCode'],
                                                                           fncpmonthcol=[])

        return fncpclsdata


class FeatureStockCode:

    @staticmethod
    def funcStockCodeFeatureProcessing(fncpclsdata):
        """
        This class is common for both train data and test data. It takes class as an input and returns a class after
        adding features to indicate whether a 'StockCode' was available in a year or not.
        """

        listTempCol = ['StockCode', 'f_InvoiceDateYear']
        listyear = [2010, 2011]
        listColToMerge = ['StockCode']
        mergedDF = dfProjectUtilityFunc.funcMergingDFBasedOnYear(fncpdf=fncpclsdata.data,
                                                                 fncplistyear=listyear,
                                                                 fncplistmergecolumnon=listColToMerge,
                                                                 fncpjointype='outer',
                                                                 fncplistcol=listTempCol)
        for year_ in listyear:
            tmpColName = 'f_StockAvailablity' + str(year_)
            # 'f_InvoiceDateYear_'+str(year_)' ex: 'f_InvoiceDateYear_2010'  is a col name in the merged
            # dataframe on which we will be filtering non NaN values.
            mergedDFColName = 'f_InvoiceDateYear_' + str(year_)
            fncpclsdata.data[tmpColName] = 0
            listAvailableStock = np.unique(np.array(mergedDF.loc[~mergedDF[mergedDFColName].isna(), ['StockCode']]))
            fncpclsdata.data.loc[fncpclsdata.data[listColToMerge[0]].isin(listAvailableStock), [tmpColName]] = 1

        return fncpclsdata

    @staticmethod
    def funcStockCodeLength(fncpclsdata):
        """
        This class is common for both train data and test data. It takes class as an input and returns a class after
        adding features to indicate the length of the stock code
        """
        fncpclsdata.data['f_StockCodeLength'] = fncpclsdata.data['StockCode'].astype(str).str.len()

        return fncpclsdata

    @staticmethod
    def funcTrainStockCodeProductReturned(fncpclsdata=clsTrain()):
        """
         It takes training class as an input and returns a class after adding feature to the dataframe to indicate
         whether this particular product has ever been returned. We use Quantity variable to check if the
         product has been returned.
        """

        lst = np.unique(fncpclsdata.data.loc[fncpclsdata.data['Quantity'] < 0, ['StockCode']].values)
        fncpclsdata.data['f_StockEverReturned'] = np.where(fncpclsdata.data['StockCode'].isin(lst), 1, 0)
        fncpclsdata.dictValuesTreated['ReturnedStockCodes'] = lst

        return fncpclsdata

    @staticmethod
    def funcTestStockCodeProductReturned(fncpclsdata=clsTest(clsTrain)):
        """
         It takes testing class as an input and returns a class after adding feature to the dataframe to indicate
         whether this particular product has ever been returned. We use Quantity variable from the train dataframe
         to check if the product has been returned.
        """
        lst = fncpclsdata.clsTrainData.dictValuesTreated['ReturnedStockCodes']
        fncpclsdata.data['f_StockEverReturned'] = np.where(fncpclsdata.data['StockCode'].isin(lst), 1, 0)

        return fncpclsdata


class CategoryPreprocessing:

    @staticmethod
    def funcOHE(fncpclstraindata=clsTrain(), fncpclstestdata=clsTest(clsTrain)):
        """
        This function One Hot Encodes categorical variables.
        """
        train, test = dfUtilityFunc.funcOneHotEncode(fncpclstraindata, fncpclstestdata)

        return train, test

    @staticmethod
    def funcTopFeatures(fncpCls, fncpColName, fncpTopXFeatures=10):
        """
        This function One Hot Encodes top X variables of a column alone. It uses the name attribute of the train class
        and test class to differentiate between them and accordingly performs OHE of top X categories.
        """
        cls = dfUtilityFunc.funcTopXFeatures(fncpCls, fncpColName, fncpTopXFeatures)
        return cls

    @staticmethod
    def funcMultiColumnAggregation(fncpCls):
        """
        This function is common for both test and train classes. This function performs multi column
        ('Country', 'StockCode', 'f_InvoiceDateYear', 'f_InvoiceDateMonth') aggregation.

        """
        # StockCode, Year, Month
        colName = 'StockCodeYearMonth'
        fncpCls.data[colName] = fncpCls.data['StockCode'].astype(str) + \
                                fncpCls.data['f_InvoiceDateYear'].astype(str) + \
                                fncpCls.data['f_InvoiceDateMonth'].astype(str)

        fncpCls.data = clsProjectUtilityFunction.funcColumnAggregation(fncpCls.data,
                                                                       fncpcolumn=colName,
                                                                       fncpNumericCol=['Quantity'],
                                                                       fncpCategoricalCol=['StockCode', 'InvoiceNo'],
                                                                       fncpmonthcol=[])
        fncpCls.data.drop(columns=[colName], inplace=True)

        # Country, StockCode, Year, Month
        colName = 'CountryStockCodeYearMonth'
        fncpCls.data[colName] = fncpCls.data['Country'].astype(str) + \
                                fncpCls.data['StockCode'].astype(str) + \
                                fncpCls.data['f_InvoiceDateYear'].astype(str) + \
                                fncpCls.data['f_InvoiceDateMonth'].astype(str)

        fncpCls.data = clsProjectUtilityFunction.funcColumnAggregation(fncpCls.data,
                                                                       fncpcolumn=colName,
                                                                       fncpNumericCol=['Quantity'],
                                                                       fncpCategoricalCol=['StockCode', 'InvoiceNo'],
                                                                       fncpmonthcol=[])
        fncpCls.data.drop(columns=[colName], inplace=True)

        # CustomerID, InvoiceNo, StockCode, Year
        colName = 'CustomerIDInvoiceNoStockCodeYearMonth'
        fncpCls.data[colName] = fncpCls.data['CustomerID'].astype(str) + \
                                fncpCls.data['InvoiceNo'].astype(str) + \
                                fncpCls.data['StockCode'].astype(str) + \
                                fncpCls.data['f_InvoiceDateYear'].astype(str)
        fncpCls.data = clsProjectUtilityFunction.funcColumnAggregation(fncpCls.data,
                                                                       fncpcolumn=colName,
                                                                       fncpNumericCol=['Quantity'],
                                                                       fncpCategoricalCol=['StockCode', 'InvoiceNo'],
                                                                       fncpmonthcol=[])
        fncpCls.data.drop(columns=[colName], inplace=True)

        # # InvoiceNo, StockCode
        # fncpCls.data['InvoiceNoStockCode'] = fncpCls.data['InvoiceNo'].astype(str) + \
        #                                      fncpCls.data['StockCode'].astype(str)
        # fncpCls.data = clsProjectUtilityFunction.funcColumnAggregation(fncpCls.data, fncpcolumn='InvoiceNoStockCode',
        #                                                                fncpNumericCol=['Quantity'],
        #                                                                fncpCategoricalCol=['StockCode', 'InvoiceNo'],
        #                                                                fncpmonthcol=[])
        # fncpCls.data.drop(columns=['InvoiceNoStockCode'], inplace=True)

        # CustomerID, StockCode
        colName = 'CustomerIDStockCode'
        fncpCls.data[colName] = fncpCls.data['CustomerID'].astype(str) + \
                                fncpCls.data['StockCode'].astype(str)
        fncpCls.data = clsProjectUtilityFunction.funcColumnAggregation(fncpCls.data,
                                                                       fncpcolumn=colName,
                                                                       fncpNumericCol=['Quantity'],
                                                                       fncpCategoricalCol=['StockCode', 'InvoiceNo'],
                                                                       fncpmonthcol=[])
        fncpCls.data.drop(columns=[colName], inplace=True)


        return fncpCls


class NumericalPreprocessing:

    @staticmethod
    def MinMaxNormalizing(fncpclstraindata=clsTrain(), fncpclstestdata=clsTest(clsTrain)):
        """
        This function normalizes the features using MinMaxScalar.main.py
        """
        norm, fncpclstraindata.data = dfUtilityFunc.funcNormalizingFeature(fncpdf=fncpclstraindata.data, fncpnorm=None)
        norm, fncpclstestdata.data = dfUtilityFunc.funcNormalizingFeature(fncpdf=fncpclstestdata.data, fncpnorm=norm)

        return fncpclstraindata, fncpclstestdata

    @staticmethod
    def PolynomialTransformation(fncpclstraindata=clsTrain(), fncpclstestdata=clsTest(clsTrain),
                                 fncpPolynomialInteger=2):
        """
        This function performs polynomial transformation on the features.
        """
        fncpclstraindata, fncpclstestdata = dfUtilityFunc.funcPolynomialTransformation(fncpclstrain=fncpclstraindata,
                                                                                       fncpclstest=fncpclstestdata,
                                                                                       fncpPolyInt=fncpPolynomialInteger)

        return fncpclstraindata, fncpclstestdata


class TargetPreprocessing:

    @staticmethod
    def TargetTransformation(fncpTrain=clsTrain()):
        """
        This function transforms the target data  using PowerTransform.
        """
        fncpTrain = dfUtilityFunc.funcTargetTransformation(fncpTrain)

        return fncpTrain


class CountryID:

    @staticmethod
    def funcCountryIDFeatureProcessing(fncpclsdata):
        """
        This class is common for both train data and test data. It takes class as an input and returns a class after
        preprocessing 'Country' feature
        """

        fncpclsdata.data = clsProjectUtilityFunction.funcColumnAggregation(fncpclsdata.data,
                                                                           fncpcolumn='Country',
                                                                           fncpNumericCol=['Quantity'],
                                                                           fncpCategoricalCol=['InvoiceNo',
                                                                                               'StockCode'],
                                                                           fncpmonthcol=[])

        return fncpclsdata

    @staticmethod
    def CountryIDBucket(fncpclstraindata=clsTrain()):
        """
        This function buckets the most frequent countries, First most occurring country has ID 35 and
        the second most occurring country has ID in the list [14, 13, 10, 30, 23, 3, 32, 26] and the remaining
        are least occurring.
        """
        fncpclstraindata.data['f_CountryID_35'] = 0
        fncpclstraindata.data['f_CountryID_35'] = np.where(fncpclstraindata.data['Country'] == 35, 1, 0)
        countryList = [14, 13, 10, 30, 23, 3, 32, 26]
        fncpclstraindata.data['f_CountryID_rest'] = 0
        fncpclstraindata.data['f_CountryID_rest'] = np.where(fncpclstraindata.data['Country'].isin(countryList), 1, 0)
        fncpclstraindata.data.drop(columns=['Country'], inplace=True)
        return fncpclstraindata


#  TODO LOGIC TO BE CHECKED
# class FeatureInvoiceNoStockCode:
#
#     @staticmethod
#     def funcFeatureStockCodeInvoiceNoColumnAggregation(fncpcls):
#         """
#         This is common for both train and test classes. This function creates and returns a aggregated dataframe based
#         on 'InvoiceNo', 'StockCode'.
#
#         :fncpcls - Column for which we need to apply groupby function
#
#         :output - class
#         """
#         aggsDF = fncpcls.data.groupby(['InvoiceNo', 'StockCode']).agg({'Quantity': ['sum', 'min', 'max', 'mean']})
#         aggsDF = aggsDF.reset_index()
#         fncpcolumn='InvoiceNoStockCode'
#         aggsDF.columns = ['f_Grp'+fncpcolumn+'_'+'_'.join(x) if x not in fncpcls.data.columns else x for x in aggsDF.columns.ravel()]
#         aggsDF = aggsDF.rename(columns={'f_Grp'+fncpcolumn+'_'+fncpcolumn+'_': fncpcolumn})
#         aggsDF['f_GrpInvoiceNoStockCode'] = aggsDF['f_GrpInvoiceNoStockCode_InvoiceNo_'].astype(str) + aggsDF['f_GrpInvoiceNoStockCode_StockCode_'].astype(str)
#         aggsDF.drop(columns=['f_GrpInvoiceNoStockCode_InvoiceNo_', 'f_GrpInvoiceNoStockCode_StockCode_'], inplace=True)
#         fncpcls.data['f_GrpInvoiceNoStockCode'] = fncpcls.data['InvoiceNo'].astype(str) + fncpcls.data['StockCode'].astype(str)
#         fncpcls.data = fncpcls.data.merge(aggsDF, how='inner', on='f_GrpInvoiceNoStockCode')
#         fncpcls.data.drop(columns=['f_GrpInvoiceNoStockCode'], inplace=True)
#         del aggsDF
#         return fncpcls


if __name__ == '__main__':
    ftrain = clsTrain()
    ftest = clsTest(ftrain)
    temp = FeatureCustomerID.funcCustomerIDFeatureProcessing(ftest)
    print(temp.data.sample(5).T)
