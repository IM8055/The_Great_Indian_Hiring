import os
import pickle

import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.regularizers import l2
from keras.layers.core import Dropout


import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

from The_Great_Indian_Hiring.src.config import clsTrain, clsTest
from The_Great_Indian_Hiring.src.preprocessing import clsPreProcessing
from The_Great_Indian_Hiring.src.prediction import clsPrediction
from The_Great_Indian_Hiring.src.constants import constTARGETCOLUMN
from The_Great_Indian_Hiring.src.loggging import clsLogFile
from The_Great_Indian_Hiring.src.utils import clsDataFrameUtilityFunctions

import warnings


# warnings.filterwarnings("ignore")


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


if __name__ == '__main__':
    # ModelParameters = {}
    dfUtilsFunc = clsDataFrameUtilityFunctions()
    topFeatures = 50
    train = clsTrain()
    train.COLUMNSTODROP = ['InvoiceDate']  # , 'InvoiceNo', 'StockCode', 'CustomerID']
    test = clsTest(train)

    preProcessing = clsPreProcessing()
    train, test, dictPreProcessingFunctionsTrain, dictPreProcessingFunctionsTest = preProcessing.funcPreprocessing(
        fncptrain=train,
        fncptest=test,
        fncpTopFeatures=topFeatures)
    prediction = clsPrediction(train.data, train.target, test.data, train)

    training = True
    saveModelFile = True
    if training:
        model = Sequential()
        neurons = 30
        activation = 'relu'
        layers = 3
        lr = 1e-3
        epochs = 200
        batchSize = 32
        model.add(Dense(neurons, input_shape=(train.data.shape[1],), activation=activation))
        # model.add(Dropout(0.25))
        model.add(Dense(neurons, activation='selu'))
        model.add(Dropout(0.25))
        # model.add(Dense(neurons, activation=activation))
        model.add(Dense(30, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        model.add(Dense(neurons, activation=activation))
        # model.add(Dense(neurons, activation=activation))
        model.add(Dense(1, ))
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=10000,
            decay_rate=0.9)
        # optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        optimizer = keras.optimizers.Adam(lr=lr)
        model.compile(optimizer,
                      loss=root_mean_squared_error,
                      metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])

        # Pass several parameters to 'EarlyStopping' function and assigns it to 'earlystopper'
        earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')

        X_train, X_test, y_train, y_test = train_test_split(train.data, train.target, test_size=0.1)

        # Fits model over 2000 iterations with 'earlystopper' callback, and assigns it to history
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batchSize, shuffle=True, validation_split=0.1, verbose=1,
                            callbacks=[earlystopper])

        # Runs model with its current weights on the training and testing data
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculates and prints r2 score of training and testing data
        print("The RMSE score on the Train set is:\t{:0.5f}".format(np.sqrt(mean_squared_error(y_train, y_train_pred))))
        print("The RMSE score on the Test set is:\t{:0.5f}".format(np.sqrt(mean_squared_error(y_test, y_test_pred))))

        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1)

        loss_train = history.history['loss']
        loss_val = history.history['val_loss']
        listEpochs = range(0, len(loss_train))
        ax1.plot(listEpochs, loss_train, 'g', label='Training loss')
        ax1.plot(listEpochs, loss_val, 'b', label='validation loss')
        ax1.set_ylabel('Loss')
        ax1.legend()

        loss_train = history.history['rmse']
        loss_val = history.history['val_rmse']
        listEpochs = range(0, len(loss_train))
        ax2.plot(listEpochs, loss_train, 'g', label='Training RMSE')
        ax2.plot(listEpochs, loss_val, 'b', label='validation RMSE')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('RMSE')
        ax2.legend()

        plt.show()

        # Model Prediction
        predComplete = model.predict(prediction.testdata)
        predComplete = np.abs(predComplete)
        # Checks if we have power transformed the target variable. The key 'PowerTransform' is added to the dict
        # only when we power transform target variable. Check class TargetPreprocessing in featurepreprocessing.py
        if 'PowerTransform' in prediction.train.dictValuesTreated.keys():
            power = prediction.train.dictValuesTreated['PowerTransform']
            predComplete = power.inverse_transform(predComplete.reshape(-1, 1))
        dfComplete = pd.DataFrame()
        dfComplete[constTARGETCOLUMN] = pd.DataFrame(predComplete, columns=[constTARGETCOLUMN])[
            constTARGETCOLUMN].values

        if saveModelFile:
            modelFileNameJson = 'NeuralNetwork' + '_' + prediction.currentTime + '.json'
            modelFileNameH5 = 'NeuralNetwork' + '_' + prediction.currentTime + '.h5'
            # serialize model to JSON
            model_json = model.to_json()
            with open(os.path.join(train.MODELPATH, modelFileNameJson), "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights(os.path.join(train.MODELPATH, modelFileNameH5))

            print(f'\tSaved the Model {modelFileNameJson} and {modelFileNameH5} in the MODELPATH !!!')

            outputFileName = prediction.funcSaveFile(funcpdataframe=dfComplete)
            print(f'\tPrediction file {outputFileName} successfully saved in the OUTPUTPATH')

            mdlName = 'Neural Network'
            mdlFeatures = train.data.columns
            mdlHyperParameters = 'None'
            mdlFileName = modelFileNameJson
            mdlOutputFileName = outputFileName
            modelTrainScore = np.sqrt(mean_squared_error(y_train, y_train_pred))
            modelTestScore = np.sqrt(mean_squared_error(y_test, y_test_pred))
            mdlComment = [f'1. Model Config:\n\t{layers} hidden layers\n\t{neurons} neurons in each layer\n\t{activation} activation\n',
                          f'2. learning rate = {lr}\n',
                          f'3. Epochs = {epochs}\n',
                          f'4. Batch Size {batchSize}\n',
                          f'5. Optimizer {optimizer.get_config()}'
                          f'5. {modelFileNameH5}']
            GITCOMMENTID = 524
            preProcessingComment = [f'1. Neural Network\n',
                                    f'2. Selecting top {topFeatures} features\n',
                                    '4. Implemented target transformation\n',
                                    '5. Removed quantity outlier treatment\n',
                                    '6. Did not convert quantity to positive\n',
                                    '7. Changed learning rate to 1e-3"\n',
                                    '8. Dropped duplicates\n'
                                    '9. Changed decay rate to 1e-6\n',
                                    '10. Capped outlier\n',
                                    '11. Added one more bins for "Quantity"\n'
                                    '12. Treated negative "Quantity"'
                                    ]
            modelLog = clsLogFile(ModelName=mdlName,
                                  ModelFeatures=mdlFeatures,
                                  ModelHyperParameters=mdlHyperParameters,
                                  ModelCVScore=str(modelTrainScore)+', '+str(modelTestScore),
                                  ModelFileName=mdlFileName,
                                  OutputFileName=outputFileName,
                                  ModelPreProcessingSteps=preProcessingComment,
                                  comments=mdlComment,
                                  gitCommentID=GITCOMMENTID)
            modelLog.funcLogging()


