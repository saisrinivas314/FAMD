
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from skfeature.function.information_theoretical_based import FCBF
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgbm

main = tkinter.Tk()
main.title("FAMD: A Fast Multifeature Android Malware Detection Framework, Design, and Implementation") #designing main screen
main.geometry("1300x1200")

global filename
global dataset
global X, Y
global X_train, X_test, y_train, y_test
global precision, recall, accuracy, fscore

def upload(): 
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    dataset  = pd.read_csv(filename)
    label = dataset.groupby('label').size()
    label = label[0:20]
    label.plot(kind="bar")
    plt.title("Top 20 Malware Attack found in dataset")
    plt.show()

def preprocessDataset():
    global X, Y
    global dataset
    X = []
    Y = []
    text.delete('1.0', END)
    temp = dataset['label']
    for i in range(len(temp)):
        if temp[i] == 'Benign':
            Y.append(0)
        else:
            Y.append(1)
    Y = np.asarray(Y)
    k = 0
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            if k > 0:
                words = ''
                arr = line.split(",")
                for i in range(len(arr)-1):
                    words+=arr[i]+" "
                X.append(words)
            k = k + 1    
    f.close()
    X = np.asarray(X)
    text.insert(END,"Total android apps found in dataset : "+str(len(X))+"\n")
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    #here X contains all dataset features
    X = X[indices]
    #Y contains class label as malware or benign
    Y = Y[indices]
    #here in vector object will be created with NGRAM RANGE
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), decode_error='replace')
    #to vector object we will input X features to generate NGRAM from X features
    tfidf = tfidf_vectorizer.fit_transform(X).toarray()
    #NGRAM features will be converted into vector frame
    df = pd.DataFrame(tfidf, columns=tfidf_vectorizer.get_feature_names())
    print(str(df))
    print(df.shape)
    df1 = df.values
    X = df1[:, 0:df.shape[1]]
    text.insert(END,"Total features found in each android app : "+str(X.shape[1])+"\n\n")
    text.insert(END,str(df))
    
def executeFCBF():
    text.delete('1.0', END)
    global X, Y
    global X_train, X_test, y_train, y_test
    text.insert(END,"Total features found in each android app before applying FCBF : "+str(X.shape[1])+"\n\n")
    #creating FCBF object and input X and Y values and then FCBF will return index of important features as idx variable
    idx = FCBF.fcbf(X,Y, n_selected_features=60)
    #from X we will chose only important idx features
    X = X[:, idx[0:60]]
    text.insert(END,"Total features found in each android app after applying FCBF : "+str(X.shape[1])+"\n\n")
    X = X[0:600]
    Y = Y[0:600]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

def executeKNN():
    global X, Y
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    global precision, recall, accuracy, fscore
    precision = []
    recall = []
    accuracy = []
    fscore = []
    cls = KNeighborsClassifier()
    cls.fit(X_train,y_train)
    predict = cls.predict(X_test)
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    accuracy.append(a)
    text.insert(END,"KNN Accuracy  : "+str(a)+"\n")
    text.insert(END,"KNN Precision : "+str(p)+"\n")
    text.insert(END,"KNN Recall    : "+str(r)+"\n")
    text.insert(END,"KNN FSCORE    : "+str(f)+"\n\n")


def executeRandomForest():
    global X, Y
    global X_train, X_test, y_train, y_test
    cls = RandomForestClassifier()
    cls.fit(X_train,y_train)
    predict = cls.predict(X_test)
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    accuracy.append(a)
    text.insert(END,"Random Forest Accuracy  : "+str(a)+"\n")
    text.insert(END,"Random Forest Precision : "+str(p)+"\n")
    text.insert(END,"Random Forest Recall    : "+str(r)+"\n")
    text.insert(END,"Random Forest FSCORE    : "+str(f)+"\n\n")


def executeXGBoost():
    global X, Y
    global X_train, X_test, y_train, y_test
    cls = XGBClassifier()
    cls.fit(X_train,y_train)
    predict = cls.predict(X_test)
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    accuracy.append(a)
    text.insert(END,"XGBoost Accuracy  : "+str(a)+"\n")
    text.insert(END,"XGBoost Precision : "+str(p)+"\n")
    text.insert(END,"XGBoost Recall    : "+str(r)+"\n")
    text.insert(END,"XGBoost FSCORE    : "+str(f)+"\n\n")


def executeLGBM():
    global X, Y
    global X_train, X_test, y_train, y_test
    cls = lgbm.LGBMClassifier()
    cls.fit(X_train,y_train)
    predict = cls.predict(X_test)
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    accuracy.append(a)
    text.insert(END,"LightGBM Accuracy  : "+str(a)+"\n")
    text.insert(END,"LightGBM Precision : "+str(p)+"\n")
    text.insert(END,"LightGBM Recall    : "+str(r)+"\n")
    text.insert(END,"LightGBM FSCORE    : "+str(f)+"\n\n")
    

def executeCatBoost():
    global X, Y
    global X_train, X_test, y_train, y_test
    #creating object of CATBOOST algorithm
    cls = CatBoostClassifier(iterations=50, learning_rate=0.1, custom_loss=['AUC', 'Accuracy'])
    #start training CatBoost with X and Y features
    cls.fit(X,Y)
    #performing prediction on test data using CatBoost trained object
    predict = cls.predict(X_test)
    predict = predict.flatten()
    y_test = y_test.flatten()
    pred = []
    for i in range(len(predict)):
        pred.append(int(predict[i]))
    predict = np.asarray(pred)
    print(predict)
    print(y_test)
    #calculating accuracy between original test values and predicted values
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    accuracy.append(a)
    text.insert(END,"CatBoost Accuracy  : "+str(a)+"\n")
    text.insert(END,"CatBoost Precision : "+str(p)+"\n")
    text.insert(END,"CatBoost Recall    : "+str(r)+"\n")
    text.insert(END,"CatBoost FSCORE    : "+str(f)+"\n\n")
    

def graph():
    df = pd.DataFrame([['KNN','Precision',precision[0]],['KNN','Recall',recall[0]],['KNN','F1 Score',fscore[0]],['KNN','Accuracy',accuracy[0]],
                       ['Random Forest','Precision',precision[1]],['Random Forest','Recall',recall[1]],['Random Forest','F1 Score',fscore[1]],['Random Forest','Accuracy',accuracy[1]],
                        
                       ['XGBoost','Precision',precision[2]],['XGBoost','Recall',recall[2]],['XGBoost','F1 Score',fscore[2]],['XGBoost','Accuracy',accuracy[2]],
                       ['Light GBM','Precision',precision[3]],['Light GBM','Recall',recall[3]],['Light GBM','F1 Score',fscore[3]],['Light GBM','Accuracy',accuracy[3]],
                       ['CatBoost','Precision',precision[4]],['CatBoost','Recall',recall[4]],['CatBoost','F1 Score',fscore[4]],['CatBoost','Accuracy',accuracy[4]],
                                              
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='FAMD: A Fast Multifeature Android Malware Detection Framework, Design, and Implementation')
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Drebin Malware Dataset", command=upload)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

ngramButton = Button(main, text="Preprocess with NGram Technique", command=preprocessDataset)
ngramButton.place(x=320,y=550)
ngramButton.config(font=font1) 

fcbfButton = Button(main, text="Apply FCBF Feature Selection Algorithm", command=executeFCBF)
fcbfButton.place(x=620,y=550)
fcbfButton.config(font=font1) 

knnButton = Button(main, text="Execute KNN Algorithm", command=executeKNN)
knnButton.place(x=960,y=550)
knnButton.config(font=font1)

rfButton = Button(main, text="Execute Random Forest Algorithm", command=executeRandomForest)
rfButton.place(x=50,y=600)
rfButton.config(font=font1)

xgboostButton = Button(main, text="Execute XGBoost Algorithm", command=executeXGBoost)
xgboostButton.place(x=320,y=600)
xgboostButton.config(font=font1)

lgbmButton = Button(main, text="Execute LGBM Algorithm", command=executeLGBM)
lgbmButton.place(x=620,y=600)
lgbmButton.config(font=font1)

catboostButton = Button(main, text="Execute CatBoost Algorithm", command=executeCatBoost)
catboostButton.place(x=960,y=600)
catboostButton.config(font=font1)

graphButton = Button(main, text="Accuracy & Precision Graph", command=graph)
graphButton.place(x=50,y=650)
graphButton.config(font=font1)

main.config(bg='turquoise')
main.mainloop()
