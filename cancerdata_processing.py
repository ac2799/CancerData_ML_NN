"""
Challenge B (Higher reward):
Download the "breast-cancer.csv" file from this site.
Using python, create the best performing machine-learning algorithm you can to
predict recurrence rates based upon the variables provided.
"""
import numpy as np
import sklearn as sk
import pandas as pd

def _preprocess_range(fieldinput, separator, opt=0):
    ## field input as string; split to give list of 2 values
    values = fieldinput.split(separator)
    try:
        if opt == 0:
            ## OPTION 1: average out the two values to give midpoint in range
            values = (int(values[0]) + int(values[-1])) / 2
        elif opt == 1:
            ## OPTION 2: bottom of range
            values = int(values[0])
        elif opt == 2:
            ## OPTION 3: top of range
            values = int(values[-1])
    except ValueError:
        values = np.nan
    return values

def _preprocess_pos_neg(fieldinput, pos="yes", neg="no", neu=""):
    ## field input as a yes/no string (various cases); return 1/0 or NaN
    if fieldinput == None:
        return 0
    else:
        fieldinput = str(fieldinput)
        if fieldinput.upper() == pos.upper():
            return 1
        elif fieldinput.upper() == neg.upper():
            return -1
        elif fieldinput.upper() == neu.upper():
            return 0
        else:
            return np.nan

def preprocess(csv_file):
    """
    Process the data to obtain a pandas dataframe with only numerical data.
    Field List:
    [0]     AGE as a 10 yr range separated by a hyphen
    [1]     MENOPAUSAL as 3 values; premenopausal, ge40 or lt40
            this field is not independent from field [0] in the case of those
            pre-menopausal who are over 40
    [2]     TUMOR-SIZE as a 5mm range separated by a hyphen
    [3]     INV-NODES as a 3 item range separated by a hyphen
    [4]     NODE-CAPS as a yes/no field; some blanks
    [5]     DEG-MALIG as an integer between 1 and 3
    [6]     BREAST as left or right
    [7]     BREAST-QUART as a five way grid of left/right | up/low | central
            may split to 2 fields (left/centre/right | up/centre/low)
    [8]     IRRADIAT as yes no field; no blanks
    [9]     CLASS as recurrence-events or no-recurrence-events
    """
    
    ## open the file
    ## return the file as an array
    dataframe = pd.read_csv(csv_file, header=0)
    """
    [0]     AGE as a 10 yr range separated by a hyphen
    """
    dataframe['age'] = dataframe['age'].apply(lambda x: _preprocess_range(x, "-"))
    """
    [1]     MENOPAUSAE as 3 values; premenopausal, ge40 or lt40
            this field is not independent from field [0] in the case of those
            pre-menopausal who are over 40
    """
    menopausal_option = {"premeno": 0, "ge40":1, "lt40":2}
    dataframe['menopause'] = dataframe['menopause'].apply(lambda x: menopausal_option[x])
    """
    [2]     TUMOR-SIZE as a 5mm range separated by a hyphen
    """
    dataframe['tumor-size'] = dataframe['tumor-size'].apply(lambda x: _preprocess_range(x, "-"))
    """
    [3]     INV-NODES as a 3 item range separated by a hyphen
    """
    dataframe['inv-nodes'] = dataframe['inv-nodes'].apply(lambda x: _preprocess_range(x, "-"))
    
    """
    [4]     NODE-CAPS as a yes/no field; some blanks
    """
    dataframe['node-caps'] = dataframe['node-caps'].apply(lambda x: _preprocess_pos_neg(x))
    """
    [5]     DEG-MALIG as an integer between 1 and 3
    """
    pass
    """
    [6]     BREAST as left or right
    """
    dataframe['breast'] = dataframe['breast'].apply(lambda x: _preprocess_pos_neg(x,"left","right"))
    """
    [7]     BREAST-QUAD as a five way grid of left/right | up/low | central
            ==> split to 2 fields (left/central/right | up/central/low)
    """
    dataframe['breast-quad'] = dataframe['breast-quad'].mask(dataframe['breast-quad'] == "central", "central_central")
    dataframe.insert(7, 'breast_quad_lr', np.NaN)
    dataframe.insert(7, 'breast_quad_ud', np.NaN)
    dataframe[['breast_quad_lr','breast_quad_ud']] = dataframe['breast-quad'].str.split("_", n=1, expand = True)
    dataframe.drop(columns = ["breast-quad"], inplace = True)
    dataframe['breast_quad_lr'] = dataframe['breast_quad_lr'].apply(lambda x: _preprocess_pos_neg(x,"left","right","central"))
    dataframe['breast_quad_ud'] = dataframe['breast_quad_ud'].apply(lambda x: _preprocess_pos_neg(x,"up","low","central"))
    """
    [8]     IRRADIAT as yes no field; no blanks
    """
    dataframe['irradiat'] = dataframe['irradiat'].apply(lambda x: _preprocess_pos_neg(x))
    """
    [9]     CLASS as recurrence-events or no-recurrence-events
    """
    dataframe['Class'] = dataframe['Class'].apply(lambda x: _preprocess_pos_neg(x, "recurrence-events", "no-recurrence-events"))
    return dataframe

if __name__ == "__main__":
    print(f"Partial data\n{preprocess('breast-cancer_csv.csv').iloc[0:4, 0:9]}")
