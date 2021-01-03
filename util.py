import json
import pickle

import pandas as pd

__columns = None
__model = None


def get_predict_is_loan_approved(data):
    x = pd.DataFrame(columns=__columns)
    x.loc[0] = data
    return __model.predict(x)[0]


def load_model():
    global __columns
    if __columns is None:
        with open("artifacts/columns.json", "rb") as f:
            __columns = json.load(f)['data_columns']

    global __model
    if __model is None:
        with open("artifacts/loan_prediction_model.pickle", 'rb') as f:
            __model = pickle.load(f)
