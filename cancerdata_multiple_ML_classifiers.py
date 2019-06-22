"""
Bespoke preprocessing for cancer dataset
"""
from cancerdata_processing import preprocess
from cancerdata_validation import test_data_validation_np as tdv
"""
Generic SciKitLearn modules
"""
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler
"""
SciKitLearn classifiers
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
"""
SciKitLearn confusion matrix, classification report
"""
from sklearn.metrics import confusion_matrix as c_m
"""
to tabulate predictions
"""
import pandas as pd

def _choose_classifiers():
    """
    Specify names and classifiers to be used in pairs (matched by order)
    """
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "AdaBoost",
             "Naive Bayes", "QDA"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.020),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=20, max_features=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]
    return names, classifiers

def _sample_data():
    """
    Create representative numerical values,
    Drop any records including nan fields,
    Return pd into a numpy array
    """
    dataset = preprocess("breast-cancer_csv.csv").dropna().values

    """
    Separate classification (y) from the remainder of the data
    Split the set into a training and test set (80:20)
    Scale the data using the Standard Scaler
    """
    X, y = dataset[:, :-1], dataset[:, -1]
    #sc = StandardScaler()
    sc = MinMaxScaler()
    X = sc.fit_transform(X)
    X = sc.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)
    return X, y, X_train, X_test, y_train, y_test, sc

def _get_statistics(name, classifier, X, y, X_test, y_test):
    num_folds = 4
    score_names = ["accuracy", "precision", "recall", "F1-score"]
    scoring_systems = ["accuracy", "precision_weighted",
                       "recall_weighted", "f1_weighted"]
    print(f"The scores for the {name} classifiers are:")
    for score_name, scoring_system in zip(score_names, scoring_systems):
        # Cross validation score for K folds
        score = cross_val_score(classifier,
                                X, y,
                                scoring=scoring_system,
                                cv=num_folds)
        print(f"\t {score_name}: \t\t {round(100*score.mean(),2)}%")

def _train_all(names, classifiers,
               X, y, X_train, X_test, y_train, y_test,
               stats=True, predict=""):
    """
    Train each of the classifiers, and either output score or the predictions
    """
    ## ignore numpy warnings
    from warnings import filterwarnings
    filterwarnings('ignore')
    ## cycle around each classifier
    classes = {1:"LIKELY", -1:"UNLIKELY"}
    score = {1:0, -1:0}
    trusts = {}
    predictions = {}
    for name, classifier in zip(names, classifiers):
        ## train each classifier
        classifier.fit(X_train, y_train)
        if stats == True:
            _get_statistics(name, classifier,  X, y, X_test, y_test)
        if predict != "":
            ## Make prediction
            prediction = classifier.predict(predict)[0]

            ## Increment counter for relevant score
            score[prediction] += 1
            predictions.update({name:prediction})
            """
            reveal expected   true negatives, false positives,
                              false negatives, true positives
            """
            tn, fp, fn, tp = c_m(y_test, classifier.predict(X_test)).ravel()
            ## trust is the amount of time that the prediction was correct
            trust_score = tp/(tp + fp) if prediction == 1 else tn/(tn + fn)
            trust_score = round((trust_score * 100), 2)
            trusts.update({name:trust_score})
    if predict != "":
        scores = pd.DataFrame({'Recurrence':predictions,
                               'Confidence':trusts})
        pred_weight = scores.Recurrence * scores.Confidence
        weights = pd.DataFrame({'Weights':pred_weight})
        scores['Recurrence'] = scores['Recurrence'].apply(lambda x: classes[x])
        print(scores)
        classification = 1 if weights.Weights.mean() > 0 else -1
        print(f"\nRecurrence judged {classes[classification]} at \
{round(abs(weights.Weights.mean()),2)} % confidence")
        print(f"Poll of classifiers results:")
        for index in score:print(f"{classes[index]}:  \t\t{score[index]}")
        

def overview():
    names, classifiers = _choose_classifiers()
    X, y, X_train, X_test, y_train, y_test, sc = _sample_data()
    _train_all(names, classifiers, X, y, X_train, X_test, y_train, y_test)

def test():
    names, classifiers = _choose_classifiers()
    X, y, X_train, X_test, y_train, y_test, sc = _sample_data()
##    ## test on a random entry from the dataset
##    from random import randint
##    _train_all(names, classifiers,
##               X, y, X_train, X_test, y_train, y_test,
##               False, X[-randint(0,32):])
    ## test on a hypothetical case (user input)
    test_data = sc.transform(tdv()[:,:-1])
    _train_all(names, classifiers,
               X, y, X_train, X_test, y_train, y_test,
               False, test_data)

if __name__ == "__main__":
    test()
