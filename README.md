# CancerData
Machine Learning (Python) for working with breast cancer data.

This breast cancer domain was obtained from the University Medical Centre, Institute of Oncology, Ljubljana, Yugoslavia. Thanks go to M. Zwitter and M. Soklic for providing the data.

## Environment Setup
This repo requires the following modules to be installed:
* Python 3.6 or above
* numpy
* pandas
* sci-kit learn (sklearn)

## Included Files

The data comes as a single csv file, available from https://datahub.io/machine-learning/breast-cancer/r/breast-cancer.csv

The code is separated into 4 files: 
* cancerdata_processing.py converts the breast-cancer csv into a pandas dataframe with only numerical values
* cancerdata_multiple_ML_classifiers.py leverages the cancerdata_processing module to train and test 9 different ML algorithms, available through scikit-learn. Available options are to display general performance statistics ( overview() ) or to perform a test with unseen data
* cancerdata_multiple_NN_classifiers.py works similarly to cancerdata_multiple_ML_classifiers except using the 3 different modes of multi-layer perceptron neural nets available through scikit-learn.
* cancerdata_validation.py is used to allow the above classifiers to utilise user-input data as a test, rather than a random selection from the csv file. This would be useful in a future context (e.g. in a deployed clinical setting)

Overall performance is poor, but work on optimising the settings for each classifier, plus a larger data set would improve performance.

## Default Settings
Scaling: sklearn.preprocessing.MinMaxScaler
Train/Test split: 80:20
Number of folds: 20
Confidence values: True/False within classification (e.g. True Positive/False Positive or True Negative/False Negative)
