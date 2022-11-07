import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from Perceptron import Perceptron

# Handle dataset and create a map for the Features and Classes
dataset = pd.read_csv('processed-penguins.csv')
col = list(dataset.columns.values)
dataset = dataset[col[1:]+[col[0]]]
Features = {1:'bill_length_mm',2:'bill_depth_mm',3:'flipper_length_mm',4:'gender',5:'body_mass_g'}
Species = {1:'Adelie',2:'Gentoo',3:'Chinstrap'}


# Split the Data into 0.6 training and 0.4 testing (the required ratio)
# Class1 and class2 are integers that map to its real class in Species map
def CreateTrainAndTestDataframes(class1, class2):
    df1 = dataset[dataset['species'] == Species[class1]]
    df2 = dataset[dataset['species'] == Species[class2]]

    df1_train = df1.sample(frac=0.6, random_state=1)
    df2_train = df2.sample(frac=0.6, random_state=1)

    df1_test = df1.loc[~df1.index.isin(df1_train.index)]
    df2_test = df2.loc[~df2.index.isin(df2_train.index)]

    result = df1_train.append(df2_train, ignore_index=True)
    df_train = result.sample(frac=1, random_state=1).reset_index()

    Result = df1_test.append(df2_test, ignore_index=True)
    df_test = Result.sample(frac=1, random_state=1).reset_index()

    return df_train, df_test

# Split the Data into (X_train & Y_train) and (X_test & Y_test)
# f1 and f2 are integers that map to its real feature in features map
def SelectFeatures(f1, f2,train,test):
    X_train = train[[Features[f1], Features[f2]]]
    X_test = test[[Features[f1], Features[f2]]]

    Y_train = train[['species']]
    Y_test = test[['species']]

    return X_train, Y_train, X_test, Y_test

# part of testing metrices
# needing to conver y_true into into 1 and -1 np array
def accuracy(y_true, y_predict):
    acccuray= np.sum(y_true==y_predict)/ len(y_true)
    return acccuray

# taking classes and features from user
c1 = input("enter class number: ")
c2 = input("enter class number: ")
train, test = CreateTrainAndTestDataframes(int(c1),int(c2))
f1 = input("enter feature number: ")
f2 = input("enter feature number: ")
X_train, Y_train,X_test, Y_test= SelectFeatures(int(f1),int(2),train,test)

# creating perceptron object
p = Perceptron(1,2)
p.fit(X_train,Y_train,Species)
predictions = p.predict(X_test)

# testing and calculating accuracy
y_converted = Y_test.to_numpy()
y=np.array([1 if i==Species[1] else -1 for i in y_converted])
print("aacuracy: ", accuracy(y,predictions))


# Plotting the graph
xtrain = X_train.to_numpy()
y_converted = Y_train.to_numpy()
y = np.array([1 if i == Species[1] else -1 for i in y_converted])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(xtrain[:, 0], xtrain[:, 1], marker='o', c=y)

x01 = np.amin(xtrain[:, 0])
x02 = np.amax(xtrain[:, 0])

x11 = (-p.weights[0] * x01 - p.bias) / p.weights[1]
x12 = (-p.weights[0] * x02 - p.bias) / p.weights[1]

ax.plot([x01, x02], [x11, x12], 'k')

ymin = np.amin(xtrain[:, 1])
ymax = np.amax(xtrain[:, 1])

ax.set_ylim([ymin - 3, ymax + 3])

plt.show()