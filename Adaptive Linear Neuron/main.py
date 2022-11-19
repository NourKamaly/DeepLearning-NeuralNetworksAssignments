import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from Perceptron import Perceptron
import pandas as pd
from tkinter import *
from tkinter import messagebox
# import warnings
# warnings.filterwarnings('ignore')


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
    fp = 0
    fn = 0
    tp = 0
    tn = 0

    for actual_value, predicted_value in zip(y_true, y_predict):
        if predicted_value == actual_value:
            if predicted_value == 1:
                tp += 1
            else:
                tn += 1
        else:
            if predicted_value == 1:
                fp += 1
            else:
                fn += 1

    Confusion_matrix = [[tn, fp],[fn, tp]]

    our_confusion_matrix = np.array(Confusion_matrix)
    Confusion_matrix

    acccuray= (tp + tn)/(tp+tn+fp+fn)
    return acccuray,Confusion_matrix



############################################ GUI #########################################################################

#creating form and put its dimension
form = Tk()
form.geometry("800x500")

#label for classes
var = StringVar()
label1 = Label( form, textvariable=var )
var.set("Select Two Classes:")
label1.pack(anchor = W )

#radio button for classes
classesNumbers = IntVar()
R1 = Radiobutton(form, text="Adelie & Gentoo", variable=classesNumbers, value=1)
R1.pack( anchor = W )

R2 = Radiobutton(form, text="Adelie & Chinstrap", variable=classesNumbers, value=2)
R2.pack( anchor = W )

R3 = Radiobutton(form, text="Gentoo & Chinstrap", variable=classesNumbers, value=3)
R3.pack( anchor = W)


#label for features
var = StringVar()
label2 = Label( form, textvariable=var)
var.set("Select Two Features:")
label2.pack(anchor = W )

#check boxes for features
features=[IntVar(),IntVar(),IntVar(),IntVar(),IntVar()]
selectedFeatures=[]
C1 = Checkbutton(form, text = "bill_length_mm", variable = features[0],onvalue = 1, offvalue = 0)
C2 = Checkbutton(form, text = "bill_depth_mm", variable = features[1],onvalue = 2, offvalue = 0)
C3 = Checkbutton(form, text = "flipper_length_mm", variable = features[2],onvalue = 3, offvalue = 0)
C4 = Checkbutton(form, text = "gender", variable = features[3],onvalue = 4, offvalue = 0)
C5 = Checkbutton(form, text = "body_mass_g", variable = features[4],onvalue = 5, offvalue = 0)
C1.pack(anchor = W)
C2.pack(anchor = W)
C3.pack(anchor = W)
C4.pack(anchor = W)
C5.pack(anchor = W)


#label for learning rate
var = StringVar()
label3 = Label( form, textvariable=var)
var.set("Enter Learning Rate(eta):")
label3.pack(anchor = W )

#textBox for learning rate
learningRate = DoubleVar()
E1 = Entry(form, bd =5,textvariable=learningRate)
E1.pack(anchor=W)


#label for epochs
var = StringVar()
label4 = Label( form, textvariable=var)
var.set("Number Of Epochs(m):")
label4.pack(anchor = W )

#textBox for epochs
numberOfEpochs = IntVar()
E2 = Entry(form, bd =5,textvariable=numberOfEpochs)
E2.pack(anchor=W)

#label for threshold
var = StringVar()
label5 = Label( form, textvariable=var)
var.set("Enter Threshold(MSE):")
label5.pack(anchor = W )

#textBox for threshold
threshold = DoubleVar()
E3 = Entry(form, bd =5,textvariable=threshold)
E3.pack(anchor=W)

#checkbox for bias
isBias=IntVar()
C6 = Checkbutton(form, text = "bias", variable = isBias,onvalue = 1, offvalue = 0)
C6.pack(anchor = W)


def startPrediction():
    # taking classes and features from user
    if(classesNumbers.get()==1):
        c1=1
        c2=2
    elif(classesNumbers.get()==2):
        c1=1
        c2=3
    else:
        c1=2
        c2=3


    train, test = CreateTrainAndTestDataframes(int(c1),int(c2))
    selectFeaures=[]
    for i in range(5):
        if(features[i].get()!=0):
            selectFeaures.append(i)
    f1=selectFeaures[0]+1
    f2=selectFeaures[1]+1

    print("c1: " + str(c1))
    print("c2: " + str(c2))
    print("f1: " + str(f1))
    print("f2: " + str(f2))
    print("learning rate: " + str(learningRate.get()))
    print("epochs: "+ str(numberOfEpochs.get()))
    print("bias: "+str(isBias.get()==1))

    X_train, Y_train,X_test, Y_test= SelectFeatures(int(f1),int(f2),train,test)

    # creating perceptron object
    p = Perceptron(c1,c2,learningRate.get(),numberOfEpochs.get(),isBias.get()==1,threshold.get())
    p.fit(X_train,Y_train,Species)
    predictions = p.predict(X_test)

    # testing and calculating accuracy
    y_converted = Y_test.to_numpy()
    y=np.array([1 if i==Species[p.class1] else -1 for i in y_converted])

    acc1,confusion_matrix= accuracy(y,predictions)
    print("aacuracy: ", str(acc1))
    print("Confusion Matrix: ", str(confusion_matrix))
    print("MSE: ",p.calculateMSE(y,predictions,y.shape[0]))
    messagebox.showinfo("Accuracy","Accuracy Is: " + str(acc1))

    # Plotting the graph
    xtrain = X_train.to_numpy()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(xtrain[:, 0], xtrain[:, 1], marker='o',c=p.Y_encoded)



    x01 = np.amin(xtrain[:, 0])
    x02 = np.amax(xtrain[:, 0])

    x11 = (-p.weights[0] * x01 - p.bias) / p.weights[1]
    x12 = (-p.weights[0] * x02 - p.bias) / p.weights[1]

    ax.plot([x01, x02], [x11, x12], 'k')

    ymin = np.amin(xtrain[:, 1])
    ymax = np.amax(xtrain[:, 1])

    ax.set_ylim([ymin - 3, ymax + 3])

    plt.show()


#start Button
button=Button(form,text="Start",width=10,command=startPrediction)
button.pack()

#main loop
form.mainloop()

