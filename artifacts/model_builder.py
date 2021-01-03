import json
import pickle

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

loan_data = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv")
loan_data.drop('Unnamed: 0', axis=1, inplace=True)

"""# SPLIT """

X = loan_data.drop('Loan_Status', axis=1)
y = loan_data['Loan_Status']
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=1234, stratify=y)

"""# CLEANING DATA

## CLEAN TRAIN DATA
"""

imp = SimpleImputer(strategy="most_frequent")
cleaned_X_train = pd.DataFrame(imp.fit_transform(X_train), columns=loan_data.columns[:len(loan_data.columns) - 1])
cleaned_X_train.drop('Loan_ID', axis=1, inplace=True)

int_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Credit_History']
for col in int_columns:
    cleaned_X_train[col] = cleaned_X_train[col].apply(lambda r: int(r))

cleaned_X_train['Loan_Amount_Term'] = cleaned_X_train['Loan_Amount_Term'].apply(lambda r: str(int(r)))
cleaned_X_train.head()

"""## CLEAN VALIDATION DATA"""

imp_val = SimpleImputer(strategy="most_frequent")
cleaned_X_test = pd.DataFrame(imp_val.fit_transform(X_test), columns=loan_data.columns[:len(loan_data.columns) - 1])
cleaned_X_test.drop('Loan_ID', axis=1, inplace=True)
for col in int_columns:
    cleaned_X_test[col] = cleaned_X_test[col].apply(lambda r: int(r))
cleaned_X_test['Loan_Amount_Term'] = cleaned_X_test['Loan_Amount_Term'].apply(lambda r: str(int(r)))
cleaned_X_test.head()

"""# ENCODE DATA

## ENCODE TRAIN DATA
"""
cleaned_X_train['Gender'].unique()

cat_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Loan_Amount_Term', 'Property_Area']
cat_values = [cleaned_X_train[cat].unique() for cat in cat_features]
encoder = OneHotEncoder(categories=cat_values, dtype=np.int64, sparse=False, handle_unknown='ignore')
encoder.fit(cleaned_X_train[cat_features])

cat_dummies = pd.DataFrame(encoder.transform(cleaned_X_train[cat_features]))
cat_dummies.head()

num_features = ['ApplicantIncome', 'CoapplicantIncome']
cleaned_X_train[num_features].head()

sc = StandardScaler()
sc.fit(cleaned_X_train[num_features])

num_dummies = pd.DataFrame(sc.transform(cleaned_X_train[num_features]))
num_dummies.head()

ready_X_train = pd.merge(num_dummies, cat_dummies, left_index=True, right_index=True)
ready_X_train.head()

"""## ENCODE VALIDATION DATA"""

ready_X_test = pd.merge(pd.DataFrame(sc.transform(cleaned_X_test[num_features])),
                        pd.DataFrame(encoder.transform(cleaned_X_test[cat_features])), left_index=True,
                        right_index=True)
ready_X_test.head()

"""# HANDLE IMBALANCED DATA"""

sm = SMOTE(random_state=1234)
X_train_res, y_train_res = sm.fit_resample(ready_X_train, y_train)

pd.DataFrame(X_train_res)

"""# CREATE MODEL """

tree_model = DecisionTreeClassifier(min_samples_split=0.05, random_state=1234)
tree_model.fit(X_train_res, y_train_res)
f1_score(y_train_res, tree_model.predict(X_train_res)), f1_score(y_test, tree_model.predict(ready_X_test))

dt_params = {
    'min_samples_split': [.001, .005, .01, .05, .1]
}

tree_cv = GridSearchCV(DecisionTreeClassifier(random_state=1234), dt_params, scoring='f1', cv=5, n_jobs=-1, verbose=2)
tree_cv.fit(X_train_res, y_train_res)
f1_score(y_train_res, tree_cv.predict(X_train_res)), f1_score(y_test,
                                                              tree_cv.predict(ready_X_test)), tree_cv.best_params_

"""# MAKE PIPELINE"""

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('encoder', OneHotEncoder(dtype=np.int64, sparse=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ], n_jobs=-1, remainder='drop'
)

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=1234)),
    ('classifier', tree_cv.best_estimator_)
])

"""# PREDICT ON TEST DATA"""

clf.fit(cleaned_X_train, y_train)
f1_score(y_train, clf.predict(cleaned_X_train)), f1_score(y_test, clf.predict(cleaned_X_test))

"""# EXPORT PIPELINE"""

with open('loan_prediction_model.pickle', 'wb') as f:
    pickle.dump(clf, f)

columns = {
    'data_columns': [col for col in cleaned_X_train.columns]
}

with open('columns.json', 'w') as f:
    f.write(json.dumps(columns))
