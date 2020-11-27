# Great Indian Hiring Hackathon

## Overview
This competition was organized by <b>Machine Hack</b> platform in collaboration with 12 companies as a screening round for potential candidates. The link for the competition can be found 
[here](https://www.machinehack.com/hackathons/retail_price_prediction_mega_hiring_hackathon/overview).

The task was to predict <b>UnitPrices</b> for a number of Items using the below features
* InvoiceNo
* StockCode
* Description
* Quantity
* InvoiceDate
* UnitPrice
* CustomerID
* Country

With this approach I received a private leaderboard rank of <b>272/1343</b> and my main intention was to create a template that can be used for all machine learning hackathons that involve regression problems.

### Project overview
* constants.py - Constants are defined in this project specific file.
* config.py - Train and Test classes are defined in this project specific file.
* featurepreprocessing.py - Variable preprocessing logic and feature engineering steps are defined in this project specific file.
* featureselection.py - Feature Selection logic is implemented in this file.
* GPMinimize.py - GP Minimize logic for hyperparameter tuning is implemented in this file.
* GPMinimizeConstants.py - GPMinimize constants are defined in this file.
* logging.py - Logging function is defined in this file.
* main.py - ML Model's main file.
* mainNeuralNetwork.py - Neural Network main file.
* models.py - Functionality for comparing various regression models without hyperparameter tuning is implemented in this file.
* prediction.py - Predicting outputs.
* preprocessing.py - This file defines preprocessing logic for train and test dataset.
* projectutils.py - Project specific helper functions.
* utils.py - Common helper functions.
* train.py - This file helps us in training a model.
* twilliowhatsapp.py - This file helps us to send a WhatsApp message after the model training is done.
* requirements.txt - Contains all the necessary libraries used for this project.

<ins>Note</ins><br />
Save the train and test csv file in the input folder before running the program.
