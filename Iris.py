#Iris Classification
from pprint import pprint
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier

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


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.85,
                                                    random_state=42,
                                                    stratify=y)

print(f"Train labels:\n{y_train}")
print(f"Test labels:\n{y_test}")

print(f"Train features:\n{X_train}")
print(f"Test features:\n{X_test}")

# Checking all flower types exist in eaqual preportions in the train and test set.
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))