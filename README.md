# houseprice
In this research, I was asked to give a model to predict the house prices in Brooklyn, using the features in the given dataset.
I built a model to train the data, and use this model to predict the future mean price for one month and one year forward.

# data
Download data from https://www.kaggle.com/tianhwu/brooklynhomes2003to2017, and put the data in path "data/".

# file instruction
* Feature_generate.py: This file generates features from data set, build train and test set and also shows some basic information of the data set. You should run this file first.
* Model_train.py: This file trains the model, and tuning the parameters for the models.
* Model_test.py: This file validates the model in the test set.
* Predict.py: This file uses the model to predict the monthly and yearly forward house prices.
* Parameters.py: This file contains the parameters used in the machine learning models.
* Config.py: This file contains some configurations of the models and data set. 

