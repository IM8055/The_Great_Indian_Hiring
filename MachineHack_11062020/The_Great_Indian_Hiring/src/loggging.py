import os
import datetime
import pytz
from The_Great_Indian_Hiring.src.constants import constLOGFILENAME
from The_Great_Indian_Hiring.src.config import clsProjectConfig


class clsLogFile:

    def __init__(self, ModelName, ModelCVScore, ModelFeatures, ModelHyperParameters, ModelFileName,
                 ModelPreProcessingSteps, OutputFileName, comments='None', gitCommentID=000):
        self.ModelName = ModelName
        self.ModelFileName = ModelFileName
        self.ModelCVScore = ModelCVScore
        self.ModelFeatures = ModelFeatures
        self.ModelHyperParameters = ModelHyperParameters
        self.Comments = comments
        self.GitComment = gitCommentID
        self.LogFileName = constLOGFILENAME
        self.ModelPreProcessingSteps = ModelPreProcessingSteps
        self.OutputFileName = OutputFileName

    def funcLogging(self):

        with open(os.path.join(clsProjectConfig.LOGPATH, self.LogFileName), "a+") as file:
            file.write('\n')
            current_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%d-%m-%Y %H:%M:%S")
            commentDate = 'Date Time: ' + str(current_time)
            file.write(commentDate)
            file.write('\n\n')
            file.write('Model Name: '+self.ModelName)
            file.write('\n')
            file.write('Model File Name: '+self.ModelFileName)
            file.write('\n')
            file.write('Output File Name: '+self.OutputFileName)
            file.write('\n')
            file.write('Model Score: '+str(self.ModelCVScore))
            file.write('\n')
            file.write('Model Features:\n')
            file.write(str(self.ModelFeatures))
            file.write('\n')
            file.write('Model Hyper Parameters:\n')
            file.write(str(self.ModelHyperParameters))
            file.write('\n')
            file.write('Model Preprocessing Comments:\n')
            file.writelines(self.ModelPreProcessingSteps)
            file.write('\n')
            file.write('Model Comments:\n')
            file.writelines(self.Comments)
            file.write('\n')
            file.write('Git Comment ID:\n')
            file.write(str(self.GitComment))
            file.write('\n')
            file.write('------------------------------------------------------------------------------------')
            file.write('\n')
            file.write('------------------------------------------------------------------------------------')
        print('\nLog Saved!!!')


if __name__ == '__main__':
    mdlName = 'Test'
    mdlFeatures = ['A', 'B']
    mdlHyperParameters = {'A': 5, 'B': 8}
    mdlFileName = 'Test'
    mdlOutputFileName = ''
    mdlCVScore = 5
    preProcessingComment = ['1.Test\n', '2.Test2']
    mdlComment = 'Test'
    modelLog = clsLogFile(ModelName=mdlName,
                          ModelFeatures=mdlFeatures,
                          ModelHyperParameters=mdlHyperParameters,
                          ModelCVScore=mdlCVScore,
                          ModelFileName=mdlFileName,
                          OutputFileName=mdlFileName,
                          ModelPreProcessingSteps=preProcessingComment,
                          comments=mdlComment,
                          gitCommentID=000)
    modelLog.funcLogging()
