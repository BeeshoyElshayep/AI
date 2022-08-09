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
import time as time
import joblib 
import pickle as pk


#Read The Training data
testDataset = pd.read_csv('C:\\Users\\Eng Beeshoy\\Downloads\\heart_test.csv')
datset=pd.read_csv('C:\\Users\\Eng Beeshoy\\Downloads\\heart_train.csv')
indextest = testDataset.iloc[:, 0].to_numpy()

#Sacatter Data
X = datset.iloc[:, :-1]
y = datset.iloc[:, -1].to_numpy()
"""
#Best features for training
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Feature', 'Score']  # naming the dataframe columns
print(featureScores.nlargest(12, 'Score'))  # print 10 best features
Xtest = featureScores.nlargest(12, 'Score').iloc[0:4, 0].to_numpy()
print(Xtest)
X = X.loc[:, [x for x in Xtest]]

#Best features for test
testDataset = testDataset.loc[:, [x for x in Xtest]]
"""
#fixed random seed
randomSeed = 5


# scaling the data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
testDataset = scaler.transform(testDataset)


#PCA
pt=pd.DataFrame(X)
#print(pt)
pca=PCA(random_state=randomSeed,n_components=.9)
Zpca=pca.fit_transform(pt)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("NO of components")
plt.ylabel("cum explained data")
#plt.show()
print(Zpca.shape)

datatest2 = pd.DataFrame(testDataset)

Zpca2=pca.transform(datatest2)

pk.dump(pca, open("pca.pkl", "wb"))



# split the train and test data
X_train, X_test, y_train, y_test = train_test_split(
    Zpca, y, test_size=0.2, random_state=randomSeed)


#---------- USING PCA----------
#Decision Tree
clf = DecisionTreeClassifier(
    max_depth=6, random_state=randomSeed, splitter='random', min_samples_leaf=2)
timetest1 = time.time()
clf.fit(X_train, y_train)
timetest2 = time.time()
DTreeTraining = timetest2-timetest1

timetest1 = time.time()
y_prediction = clf.predict(X_test)
timetest2 = time.time()
DTreeTest=timetest2-timetest1
filename="Decision Tree Model.sav"
joblib.dump(clf,filename)
accuracy = np.mean(y_prediction == y_test)*100
print("The achieved accuracy using Decision Tree is " + str(accuracy))

DTreeAccuracy=accuracy

#KNN
Knn = KNeighborsClassifier(
    n_neighbors=94, weights='distance', algorithm='auto')
timetest1 = time.time()
Knn.fit(X_train, y_train)
timetest2 = time.time()

KnnTraining = timetest2-timetest1

timetest1 = time.time()
y_prediction = Knn.predict(X_test)
timetest2 = time.time()
KnnTest = timetest2-timetest1
filename = "KNN Model.sav"
joblib.dump(Knn, filename)

accuracy = np.mean(y_prediction == y_test)*100
print("The achieved accuracy using KNN is " + str(accuracy))
KnnAccuracy=accuracy

#SVC USING KFOLD
kf = KFold(random_state=randomSeed, shuffle=True)

timetest1 = time.time()
for trainIndex, testIndex in kf.split(X):
    X_train, X_test = X[trainIndex], X[testIndex]
    y_train, y_test = y[trainIndex], y[testIndex]
    SVmodel = SVC(verbose=True, random_state=randomSeed, gamma='auto', kernel='rbf', decision_function_shape='ovr'
                  ).fit(X_train, y_train)
timetest2 = time.time()

SVCTraining = timetest2-timetest1


timetest1 = time.time()
y_prediction = SVmodel.predict(X_test)
timetest2 = time.time()
filename = "SVC Model.sav"
joblib.dump(Knn, filename)

SVCTest = timetest2-timetest1
accuracy = np.mean(y_prediction == y_test)*100
print("The achieved accuracy using SVC is " + str(accuracy))

SVCAccuracy=accuracy


Models=["Decision Tree","KNN","SVC"]
TrainingTime = [DTreeTraining, KnnTraining, SVCTraining]

plt.plot(Models,TrainingTime)
plt.show()

plt.figure(figsize=(9, 3))

plt.subplot(131)
plt.bar(Models , TrainingTime)
plt.subplot(132)
plt.scatter(Models, TrainingTime)
plt.subplot(133)
plt.plot(Models, TrainingTime)
plt.suptitle('Training Time')
plt.show()

TestingTime=[DTreeTest,KnnTest,SVCTest] 

plt.plot(Models, TestingTime)
plt.show()

plt.figure(figsize=(9, 3))

plt.subplot(131)
plt.bar(Models, TestingTime)
plt.subplot(132)
plt.scatter(Models, TestingTime)
plt.subplot(133)
plt.plot(Models, TestingTime)
plt.suptitle('Testing Time')
plt.show()

AccuracyData=[DTreeAccuracy,KnnAccuracy,SVCAccuracy]
plt.figure(figsize=(9, 3))

plt.subplot(131)
plt.bar(Models, AccuracyData)
plt.subplot(132)
plt.scatter(Models, AccuracyData)
plt.subplot(133)
plt.plot(Models, AccuracyData)
plt.suptitle('Accuracy')
plt.show()



