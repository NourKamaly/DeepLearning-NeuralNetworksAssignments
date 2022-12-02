import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tkinter import *
from tkinter import messagebox
from Perceptron import *

dataset = pd.read_csv('processed-penguins.csv')
train_data = pd.read_csv("mnist_train.csv")
test_data = pd.read_csv("mnist_test.csv")

def Train_Test_Dataframes():
    df_train = dataset.sample(frac=0.6, random_state=1)
    df_test = dataset.loc[~dataset.index.isin(df_train.index)]

    X_train = df_train.drop(['species'], axis=1)
    X_test = df_test.drop(['species'], axis=1)

    Y_train = pd.get_dummies(df_train.species)
    Y_test = pd.get_dummies(df_test.species)

    return X_train, Y_train, X_test, Y_test





def Train_Test_Dataframe_Bonus():
    X_train = (train_data.iloc[:, 1:].values).astype('float32')
    y_train = train_data.iloc[:, 0].values.astype('int32')

    X_test = (test_data.iloc[:, 1:].values).astype('float32')
    y_test = test_data.iloc[:, 0].values.astype('int32')

    X_train=X_train/255
    X_test=X_test/255

    y_train=pd.get_dummies(y_train)
    y_test=pd.get_dummies(y_test)

    return X_train, y_train, X_test, y_test



def ConvertArray(a):
  return (a[0] & a[1] & a[2])

def arg_max_value_func(a):
  index = ['Adelie','Chinstrap','Gentoo']
  if a[0] == 1:
    return index[0]
  if a[1] == 1:
    return index[1]
  if a[2] == 1:
    return index[2]


def accuracy(y_true, y_predict):
    l = (y_predict == y_true)
    l = np.apply_along_axis(ConvertArray, 1, l)

    acccuray = np.sum(l == True) / len(y_true)
    return acccuray

def ConfusionMatrix(y_test, predictions, class_1, class_2, class_3):
  cm = confusion_matrix(y_test, predictions)
  cm_df = pd.DataFrame(cm,index = [class_1,class_2,class_3],
                          columns = [class_1,class_2,class_3])
  plt.figure(figsize=(6,6))
  sns.heatmap(cm_df, annot=True)
  plt.title('Confusion Matrix')
  plt.ylabel('Actal Values')
  plt.xlabel('Predicted Values')
  plt.show()


#creating form and put its dimension
form = Tk()
form.geometry("800x500")
#

# label for choose Dataset
var = StringVar()
label7 = Label(form, textvariable=var)
var.set("Select Mandatory Or Bonus Dataset:")
label7.pack(anchor=W)


# radio button for mandatory or bonus
InputDatasetChoice = IntVar()
R3 = Radiobutton(form, text="Penguins", variable=InputDatasetChoice, value=1)
R3.pack(anchor=W)

R4 = Radiobutton(form, text="MNIST", variable=InputDatasetChoice, value=2)
R4.pack(anchor=W)


#label for Hidden Layers
var = StringVar()
label1 = Label( form, textvariable=var )
var.set("Enter Number of Hidden Layers:")
label1.pack(anchor = W )


#textBox for Hidden Layers
InputHiddenLayers = IntVar()
E1 = Entry(form, bd =5,textvariable=InputHiddenLayers)
E1.pack(anchor=W)


#label for neurons
var = StringVar()
label2 = Label( form, textvariable=var )
var.set("Enter number of neurons in each hidden layers:")
label2.pack(anchor = W )


#textBox for neurons
InputNeurons = StringVar()
E2= Entry(form, bd =5,textvariable=InputNeurons)
E2.pack(anchor=W)


#label for epochs
var = StringVar()
label4 = Label( form, textvariable=var )
var.set("Enter the number of epochs:")
label4.pack(anchor = W )


#textBox for epochs
epochs = IntVar()
E4= Entry(form, bd =5,textvariable=epochs)
E4.pack(anchor=W)


#label for activation function
var = StringVar()
label5 = Label( form, textvariable=var )
var.set("Select One Activation Function:")
label5.pack(anchor = W )



#radio button for activation Function
InputActivationFunction = IntVar()
R1 = Radiobutton(form, text="Sigmoid", variable=InputActivationFunction, value=1)
R1.pack( anchor = W )


R2 = Radiobutton(form, text="Hyperbolic tangent", variable=InputActivationFunction, value=2)
R2.pack( anchor = W )


#label for learning rate
var = StringVar()
label3 = Label( form, textvariable=var,)
var.set("Enter Learning Rate(eta):")
label3.pack(anchor = W )


#textBox for learning rate
LearningRate = DoubleVar()
E3 = Entry(form, bd =5,textvariable=LearningRate)
E3.pack(anchor=W)


#checkbox for bias
isBias=IntVar()
C1 = Checkbutton(form, text = "bias", variable = isBias,onvalue = 1, offvalue = 0)
C1.pack(anchor = W)







def startPrediction():
    if(InputDatasetChoice.get()==1):
        neurons = [5]
        if (InputNeurons.get().find(',') == 1):
            neurons.append(int(InputNeurons.get()))
        else:
            mylist = InputNeurons.get().split(',')
            for i in mylist:
                neurons.append(int(i))
        neurons.append(3)
        X_train, Y_train, X_test, Y_test = Train_Test_Dataframes()

    else:
        neurons = [784]
        if(InputNeurons.get().find(',')==1):
            neurons.append(int(InputNeurons.get()))
        else:
            mylist = InputNeurons.get().split(',')
            for i in mylist:
                neurons.append(int(i))

        neurons.append(10)
        X_train, Y_train, X_test, Y_test = Train_Test_Dataframe_Bonus()
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)

    L = InputHiddenLayers.get()
    L += 2

    activationFunction = ""
    if (InputActivationFunction.get() == 1):
        activationFunction = "sigmoid"
    else:
        activationFunction = "tanh"

    print("hidden layers: " + str(L))
    print("neurons: " + str(neurons))
    print("activation function: " + activationFunction)
    print("learning rate: " + str(LearningRate.get()))
    print("epochs: " + str(epochs.get()))
    print("bias: " + str(isBias.get() == 1))

    p = MLP(L, neurons,isBias.get()==1, activationFunction, LearningRate.get(), epochs.get())
    p.fit(X_train, Y_train)

    predictions_test = p.predict(X_test)
    predictions_train = p.predict(X_train)

    y_test = Y_test.to_numpy()
    y_train = Y_train.to_numpy()

    predict_test = np.array(predictions_test)
    predict_train = np.array(predictions_train)

    messagebox.showinfo("Accuracy for Training", "Accuracy: " + str(accuracy(y_train, predict_train)))
    messagebox.showinfo("Accuracy for Testing", "Accuracy: " + str(accuracy(y_test, predict_test)))

    if(InputDatasetChoice.get()==1):
    #Traning
        y_t = np.apply_along_axis(arg_max_value_func, 1, y_train)
        p_t = np.apply_along_axis(arg_max_value_func, 1, predict_train)
        ConfusionMatrix(y_t, p_t, 'Adelie', 'Chinstrap', 'Gentoo')

    #Test
        y_t = np.apply_along_axis(arg_max_value_func, 1, y_test)
        p_t = np.apply_along_axis(arg_max_value_func, 1, predict_test)
        ConfusionMatrix(y_t, p_t, 'Adelie', 'Chinstrap', 'Gentoo')

#start Button
button=Button(form,text="Start",width=10,command=startPrediction)
button.pack()

#main loop
form.mainloop()

'''
Mandatory
3 - [5,4,3] - True  - "sigmoid" - 0.1 - 500   --> (98)
3 - [5,4,3] - False - "sigmoid" - 0.1 - 500   --> (100)

3 , [5,4,3] , True , "tanh" , 0.01 , 500 --> (98)
3 , [5,4,3] , False , "tanh" , 0.01 , 500 --> (100)
'''
'''
Bonus
3 - [784,16,10] - True - "tanh" - 0.1 - 100   --> (96)
'''
