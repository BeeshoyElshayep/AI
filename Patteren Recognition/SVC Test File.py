import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split , KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import joblib

testDataset = pd.read_csv('heart_test.csv')
indextest = testDataset.iloc[:, 0].to_numpy()


loaded_model = joblib.load("SVC Model.sav")
result = loaded_model.predict(testDataset)
print(result)
sample = pd.DataFrame()
sample['index'] = indextest
sample['target'] = result
print(sample)
sample.to_csv('sample.csv', index=False)
