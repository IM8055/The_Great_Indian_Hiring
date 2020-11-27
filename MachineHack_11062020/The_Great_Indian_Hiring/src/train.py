from sklearn import linear_model
from sklearn import metrics
from sklearn import ensemble

from The_Great_Indian_Hiring.src.config import clsTrain, clsTest, clsEngineeredSet
from The_Great_Indian_Hiring.src.prediction import clsPrediction
from The_Great_Indian_Hiring.src.loggging import clsLogFile
from The_Great_Indian_Hiring.src.GPMinimize import clsGPMinimize
from The_Great_Indian_Hiring.src.GPMinimizeConstants import PARAMNAMES, PARAMSPACE
from The_Great_Indian_Hiring.src.utils import clsDataFrameUtilityFunctions
from The_Great_Indian_Hiring.src.twilliowhatsapp import clsTwilioWhatsapp


class clsTrainModel:

    @staticmethod
    def funcTrain(fncpMODEL, fncpGITID, fncpTopKFeatures, fncpPreprocessingComments_GP, fncpComment_GP,
                  fncpPreprocessingComments_FIT, fncpComment_FIT, fncpTrain, fncpTest, fncpTrainFilename,
                  fncpTargetFilename, fncpTestFilename, fncpGPMinimize=True, fncpModelParameters=None, fncpNewFile=True, fncpWhatsApp=False):

        """
        This function would train a model and saves predicted values output.
        """
        dfUtilityFunc = clsDataFrameUtilityFunctions()

        if fncpNewFile:
            dfUtilityFunc.funcSavingDFToCSV(fncpTrain.data, fncpTrainFilename)
            dfUtilityFunc.funcSavingDFToCSV(fncpTrain.target, fncpTargetFilename)
            dfUtilityFunc.funcSavingDFToCSV(fncpTest.data, fncpTestFilename)

        engineeredSet = clsEngineeredSet(trainfilename=fncpTrainFilename,
                                         targetfilename=fncpTargetFilename,
                                         testfilename=fncpTestFilename)
        # This checks if model parameters are passed or not. If they are not passed and
        # GPMinimize is set to false, it uses model's default parameters.
        if fncpModelParameters is None:
            # Condition to check whether to perform GP Minimize or not
            if fncpGPMinimize:
                gpMinimize = clsGPMinimize(model=fncpMODEL,
                                           trainset=engineeredSet,
                                           fncpParamSpace=PARAMSPACE[type(fncpMODEL()).__name__],
                                           fncpParamNames=PARAMNAMES[type(fncpMODEL()).__name__],
                                           fncpmetrics=metrics.mean_squared_error)
                res, modelHyperParameters = gpMinimize.funcGPMinimize(n_calls=10,
                                                                      fncpkfoldvalue=5,
                                                                      preProcessingComment=fncpPreprocessingComments_GP,
                                                                      mdlComment=fncpComment_GP,
                                                                      gitCommentID=fncpGITID)
                modelType = fncpMODEL(**modelHyperParameters)
            else:
                modelType = fncpMODEL()
        else:
            modelHyperParameters = fncpModelParameters
            modelType = fncpMODEL(**modelHyperParameters)

        prediction = clsPrediction(engineeredSet.data, engineeredSet.target, engineeredSet.testdata)

        modelFileName, outputFileName = prediction.funcPredict(modelType)
        log = clsLogFile(ModelName=type(modelType).__name__,
                         ModelCVScore=0,
                         ModelFeatures=engineeredSet.data.columns,
                         ModelHyperParameters=modelHyperParameters,
                         ModelFileName=modelFileName,
                         OutputFileName=outputFileName,
                         ModelPreProcessingSteps=fncpPreprocessingComments_FIT,
                         comments=fncpComment_FIT,
                         gitCommentID=fncpGITID)
        log.funcLogging()

        # Sending whatsapp message
        if fncpWhatsApp:
            whatsApp = clsTwilioWhatsapp()
            if whatsApp.sendingWhatsAppMessage('Model Training completed and output is saved'):
                print('\nWhatsApp message sent')

        print('\n==============================> $$$ MODEL SUCCESSFULLY TRAINED $$$ <================================')


if __name__ == '__main__':
    model = ensemble.GradientBoostingRegressor
    ModelParameters = {'learning_rate': 0.6196132926827669, 'n_estimators': 1918, 'subsample': 0.901777017820474,
     'min_samples_split': 0.37504403071309733, 'min_samples_leaf': 0.05769203990674111,
     'min_weight_fraction_leaf': 0.19670520421773324, 'max_depth': 115, 'max_features': 0.09918135198782199}
    topFeatures = 7
    train = clsTrain()
    train.COLUMNSTODROP = ['InvoiceDate']
    test = clsTest(train)
    funcTrainParams = {'fncpMODEL': model,
                       'fncpGITID': 514,
                       'fncpTopKFeatures': topFeatures,
                       'fncpPreprocessingComments_GP': [f'1. {type(model()).__name__} GP minimize\n',
                                                        f'2. {type(model()).__name__} after feature selection\n',
                                                        f'3. Selecting top {topFeatures} features\n',
                                                        '4. Implemented Polynomial Transformation of degree 2\n',
                                                        '5. Implemented target transformation '],
                       'fncpComment_GP': [f'{type(model()).__name__}- GP Minimize HyperParameters'],

                       'fncpPreprocessingComments_FIT': [f'1. {type(model()).__name__} GP minimize\n',
                                                         f'2. {type(model()).__name__} fitted on top {topFeatures} features.\n',
                                                         f'3. {type(model()).__name__} after feature selection',
                                                         '4. Implemented Polynomial Transformation of degree 2\n',
                                                         '5. Implemented target transformation '],
                       'fncpComment_FIT': [f'1. Predicted values after fitting the model using GPMinimize.'],
                       'fncpTrain': train,
                       'fncpTest': test,
                       'fncpTrainFilename': 'FeatureEngineeredTrainSet.csv',
                       'fncpTargetFilename': 'FeatureEngineeredTargetSet.csv',
                       'fncpTestFilename': 'FeatureEngineeredTestSet.csv',
                       'fncpNewFile': True,
                       'fncpGPMinimize': True,
                       'fncpWhatsApp': False,
                       'fncpModelParameters': ModelParameters
                       }

    modelTrain = clsTrainModel()
    modelTrain.funcTrain(**funcTrainParams)