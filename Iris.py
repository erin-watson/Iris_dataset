#Iris Classification
from pprint import pprint
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

all_data = load_iris(return_X_y=False, as_frame=True)
pprint(all_data)

# I will now make training and test data.
# train_test_split could be used on the 'frame' which contains the 'data' and
# 'target' however for practice I will ignore the 'frame' and work only
# with 'data' and 'target'.

print(all_data.keys()) # 'Keys' represent the data sections withn 'all_data'

features = all_data['data']
labels = all_data['target']

X, y = features, labels

# randomized and strategically split reain and test data.
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.90,
                                                    random_state=42,
                                                    stratify=y)


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                    train_size=0.85,
                                                    random_state=42,
                                                    stratify=y_train)

print(f"Train labels:\n{y_train}")
print(f"Val labels:\n{y_val}")
print(f"Test labels:\n{y_test}")

Train_Labels = y_train
Val_Labels = y_val
Test_Labels = y_test

print(f"Train features:\n{X_train}")
print(f"Val features:\n{X_val}")
print(f"Test features:\n{X_test}")

Train_Features = X_train
Val_Features = X_val
Test_Features = X_test

# Checking all flower types exist in eaqual preportions in the train and test set.
print(y_train.value_counts(normalize=True))
print(y_val.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))

# Print NaN sum for eaach column.
for column in features.columns:
    print(features[column].isna().sum())
# As below, this is how to print NaN sum for data not technically in a column.
print(labels.isna().sum())

rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf.fit(Train_Features, Train_Labels)
print(rf)
importances = rf.feature_importances_
print(importances)

predict = rf.predict(Val_Features)
print(predict)

print(Val_Labels.values)

print('accuracy_score: ', accuracy_score(Val_Labels, predict))
