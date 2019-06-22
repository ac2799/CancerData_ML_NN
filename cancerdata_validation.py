from cancerdata_processing import preprocess
import pandas as pd

def _headers():
    """
    return the cancerdata headers
    """
    return ['age',
            'menopause',
            'tumor-size',
            'inv-nodes',
            'node-caps',
            'deg-malig',
            'breast',
            'breast-quad',
            'irradiat',
            'Class']

def _integer_range(field_name):
    while True:
        print(f"Enter the {field_name} as an integer, or a range separated by a hyphen(-) : ")
        val = input()
        try:
            test = val.split("-")
            if len(test[0]) + len("-") + len(test[-1]) == len(val) or len(test) == 1:
                int(test[0])
                int(test[-1])
                return val
        except:
            print("Your input did not match the field criteria. Try again.")

def _set_values(field_name, choices):
    while True:
        print(f"Enter the {field_name}; available choices are: ")
        for choice in choices: print(f"\t {choice.upper()}")
        val = input().lower()
        if val in choices:
            return val
        else:
            print("Your input did not match the field criteria. Try again.")

def _integer(field_name):
    while True:
        print(f"Enter the {field_name} as an integer: ")
        val = input()
        try:
            int(val)
            return val
        except:
            print("Your input did not match the field criteria. Try again.")

def _test_data_validation():
    """
    Gets user data through inputs, applies appropriate validation
    Returns as a pandas dataframe
    """
    dataframe = pd.DataFrame({'age':[_integer_range('age')],
                              'menopause':[_set_values('menopause',["premeno",
                                                                   "ge40",
                                                                   "lt40"])],
                              'tumor-size':[_integer_range('tumor-size')],
                              'inv-nodes':[_integer_range('inv-nodes')],
                              'node-caps':[_set_values('node-caps',["yes","no"])],
                              'deg-malig':[_integer('deg-malig')],
                              'breast':[_set_values('breast',["left","right"])],
                              'breast-quad':[_set_values('breast-quad',["left_up",
                                                                       "left_low",
                                                                       "central",
                                                                       "right_up",
                                                                       "right_low"])],
                              'irradiat':[_set_values('irradiat',["yes","no"])],
                              'Class':[0]})
    print(dataframe)
    return dataframe

def _test_data_validation_csv(filename):
    """
    Takes the pandas dataframe and returns as a csv
    """
    _test_data_validation().to_csv(filename, header=_headers(), index=False)

def test_data_validation_np():
    """
    Takes the pandas dataframe and returns as numpy array
    Publically accessed for user generated test data
    """
    filename = "test_data.csv"
    _test_data_validation_csv(filename)
    return preprocess(filename).values

if __name__ == "__main__":
    _test_data_validation_csv("test_data.csv")
