from sklearn import linear_model
from sklearn import metrics
from sklearn import ensemble
from sklearn import neighbors

from The_Great_Indian_Hiring.src.config import clsTrain, clsTest
from The_Great_Indian_Hiring.src.preprocessing import clsPreProcessing
from The_Great_Indian_Hiring.src.train import clsTrainModel

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    model = ensemble.RandomForestRegressor
    # ModelParameters = {}
    topFeatures = 50
    train = clsTrain()
    train.COLUMNSTODROP = ['InvoiceDate'] #, 'CustomerID']
    test = clsTest(train)

    preProcessing = clsPreProcessing()
    train, test, dictPreProcessingFunctionsTrain, dictPreProcessingFunctionsTest = preProcessing.funcPreprocessing(
        fncptrain=train,
        fncptest=test,
        fncpTopFeatures=topFeatures)
    preProcessingComment_GP = [f'1. {type(model()).__name__} GP minimize\n',
                     f'2. Selecting top {topFeatures} features\n',
                       '3. Implemented target transformation\n',
                       '4. Removed outlier treatment\n',
                       '5. Dropped duplicates\n'
                       '7. Capped outlier\n',
                       '8. Added one more bins for "Quantity"\n'
                       '9. Did not treat negative "Quantity"\n',
                       '10. Multi Column Aggregation based on "InvoiceNo", "CustomerID", "Country", "StockCode", "Year" and "Month"'
                               ]
    preProcessingComment_FIT = preProcessingComment_GP + [f'10. {type(model()).__name__} fitted on top {topFeatures} features.']

    funcTrainParams = {'fncpMODEL': model,
                       'fncpGITID': 525,
                       'fncpTopKFeatures': topFeatures,
                       'fncpPreprocessingComments_GP': preProcessingComment_GP,
                       'fncpComment_GP': [f'{type(model()).__name__}- GP Minimize HyperParameters'],
                       'fncpPreprocessingComments_FIT': preProcessingComment_FIT,
                       'fncpComment_FIT': [f'1. Predicted values after fitting the model using GPMinimize.'],
                       'fncpTrain': train,
                       'fncpTest': test,
                       'fncpTrainFilename': 'FeatureEngineeredTrainSet.csv',
                       'fncpTargetFilename': 'FeatureEngineeredTargetSet.csv',
                       'fncpTestFilename': 'FeatureEngineeredTestSet.csv',
                       'fncpNewFile': True,
                       'fncpWhatsApp': False,
                       'fncpGPMinimize': True,
                       'fncpModelParameters': None
                       }

    modelTrain = clsTrainModel()
    modelTrain.funcTrain(**funcTrainParams)
