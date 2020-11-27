import pandas as pd

from The_Great_Indian_Hiring.src.constants import COLUMNSTODROP
from The_Great_Indian_Hiring.src.config import clsTrain, clsTest
from The_Great_Indian_Hiring.src.utils import clsDataFrameUtilityFunctions
from The_Great_Indian_Hiring.src.projectutils import clsProjectUtilityFunction
from The_Great_Indian_Hiring.src.featurepreprocessing import BasicCheck, FeatureQuantity, FeatureInvoiceDate, \
    FeatureCustomerID, FeatureInvoiceNo, FeatureStockCode, NumericalPreprocessing, \
    TargetPreprocessing, CategoryPreprocessing, CountryID
from The_Great_Indian_Hiring.src.featureselection import clsRegressionFeatureSelection

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 2000)


class clsPreProcessing:

    @staticmethod
    def funcPreprocessing(fncptrain=clsTrain(), fncptest=clsTest(clsTrain), fncpTopFeatures=10):
        """
        This function processes train data set.

            Parameters:
                fncptrain (clstrain): Train class
                fncptest (clsTest): Test class
                fncpTopFeatures (int): Top N Features to be selected
            Returns:
                fncptrain (clstrain): Processed train class.\n
                fncptest (clstest): Processed test class.\n
                dictFuncTrainPreprocessing: Dictionary of functions used for train set processing.\n
                dictFuncTestPreprocessing: Dictionary of functions used for test set processing.
        """

        UtilityFunc = clsDataFrameUtilityFunctions()

        # fncptrain.COLUMNSTODROP = COLUMNSTODROP
        print('Columns to Drop')
        print(fncptrain.COLUMNSTODROP)

        # Feature Selection
        # addColumnsToDrop

        # addColumnsToDrop = ['f_QuantityBins', 'f_InvoiceDatePartOfDay', 'f_InvoiceDateMonthStart']
        # fncptrain.COLUMNSTODROP.extend(addColumnsToDrop)

        dictFuncTrainPreprocessing = {'BasicCheck': BasicCheck.funcTrainBasicChecksAndFilters,
                                    'FeatureQuantity': FeatureQuantity.funcTrainQuantityFeatureProcessing,
                                    'FeatureInvoiceDate': FeatureInvoiceDate.funcInvoiceDateDateFeatureExtraction,
                                    'FeatureStockCode': FeatureStockCode.funcStockCodeFeatureProcessing,
                                    'FeatureStockCodeLength': FeatureStockCode.funcStockCodeLength,
                                    'FeatureStockCodeProductReturned': FeatureStockCode.funcTrainStockCodeProductReturned,
                                    'FeatureCustomerID': FeatureCustomerID.funcCustomerIDFeatureProcessing,
                                    'FeatureInvoiceNo': FeatureInvoiceNo.funcInvoiceNoFeatureProcessing,
                                    'FeatureMultiColumnAggregation': CategoryPreprocessing.funcMultiColumnAggregation,
                                    # 'CountryIDAggregation': CountryID.funcCountryIDFeatureProcessing,
                                    'CountryID': CountryID.CountryIDBucket,
                                    'DroppingColumns': fncptrain.COLUMNSTODROP
                                      }
        # Train Set
        fncptrain = UtilityFunc.funcCustomPipeLine(fncptrain, dictFuncTrainPreprocessing)

        # Test Set
        dictFuncTestPreprocessing = dictFuncTrainPreprocessing.copy()
        if 'BasicCheck' in dictFuncTestPreprocessing.keys():
            dictFuncTestPreprocessing.pop('BasicCheck')
        if 'FeatureStockCodeProductReturned' in dictFuncTestPreprocessing.keys():
            dictFuncTestPreprocessing['FeatureStockCodeProductReturned'] = FeatureStockCode.funcTestStockCodeProductReturned
        if 'FeatureQuantity' in dictFuncTestPreprocessing.keys():
            dictFuncTestPreprocessing['FeatureQuantity'] = FeatureQuantity.funcTestQuantityFeatureProcessing
        if 'DroppingColumns' in dictFuncTrainPreprocessing.keys():
            dictFuncTestPreprocessing['DroppingColumns'] = fncptest.clsTrainData.COLUMNSTODROP
        fncptest = UtilityFunc.funcCustomPipeLine(fncptest, dictFuncTestPreprocessing)

        print('\nPerforming Top X category OHE on Description column')
        fncptrain = CategoryPreprocessing.funcTopFeatures(fncptrain, 'Description', fncpTopXFeatures=5)
        fncptest = CategoryPreprocessing.funcTopFeatures(fncptest, 'Description', fncpTopXFeatures=5)

        print('\nPerforming One Hot Encoding on Train DataFrame and Test DataFrame')
        fncptrain, fncptest = CategoryPreprocessing.funcOHE(fncptrain, fncptest)

        print('\nPerforming Min Max Normalization on Train DataFrame and Test DataFrame')
        fncptrain, fncptest = NumericalPreprocessing.MinMaxNormalizing(fncptrain, fncptest)

        # print('\nPerforming Polynomial Transformation on Train DataFrame and Test DataFrame')
        # fncptrain, fncptest = NumericalPreprocessing.PolynomialTransformation(fncptrain, fncptest, fncpPolynomialInteger=2)

        print('\nPerforming Feature Selection on Train DataFrame and Test DataFrame')
        featureSelection = clsRegressionFeatureSelection(fncptrain, fncptest, fncpTopFeatures)
        featureSelection.funcFeatureSelectionUsingFRegression()

        # print('\nPerforming Power Transformation on the Target Variable')
        # fncptrain = TargetPreprocessing.TargetTransformation(fncptrain)

        return fncptrain, fncptest, dictFuncTrainPreprocessing, dictFuncTestPreprocessing


if __name__ == '__main__':

    # # TODO There are additional customerID in training data compared to test data
    # # TODO There are missing customerID in training data compared to test data

    dfUtilityFunc = clsDataFrameUtilityFunctions()
    dfProjectUtilityFunc = clsProjectUtilityFunction()

    preProcessing = clsPreProcessing()
    train = clsTrain()
    train.COLUMNSTODROP = ['InvoiceDate']  # , 'CustomerID']
    test = clsTest(train)


    preProcess = True
    if preProcess:
        train, test, dictPreProcessingFunctionsTrain, dictPreProcessingFunctionsTest = preProcessing.funcPreprocessing(train,
                                                                                                                       test,
                                                                                                                       fncpTopFeatures=15)

    # print(train.data.Country.value_counts())
    # print(len(train.data.Description.unique()))


    random_state = 12345
    samples = 5

    print('\nTrain Data\n')
    print(train.data.sample(samples, random_state=random_state).T)

    # print('\nTarget Data\n')
    # print(train.target.sample(samples, random_state=random_state).T)
    #
    # print('\nTest Data\n')
    # print(test.data.sample(5, random_state=random_state).T)
    # # print(test.data.T)

    # print('\nComplete Data\n')
    # print(train.completeData.sample(samples, random_state=random_state).T)
    #
    # print(train.data.shape)

    # dfUtilityFunc.funcSavingDFToCSV(train.data, 'FeatureEngineeredTrainSet.csv')
    # dfUtilityFunc.funcSavingDFToCSV(train.target, 'FeatureEngineeredTargetSet.csv')
    # dfUtilityFunc.funcSavingDFToCSV(test.data, 'FeatureEngineeredTestSet.csv')








