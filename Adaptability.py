import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import tkinter as tk
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler
from tkinter import *
from tkinter import filedialog

# input dataset
def selectfile():
    # open file
    filepath = filedialog.askopenfilename(title='Choose a file', initialdir='/')

    # read file and remove row with Enrolled as Target
    global data
    global df
    data = pd.read_csv(filepath)
    df = data
    
    # show file name on GUI
    filename = os.path.basename(filepath)
    label2.config(text = filename)


def predict():
     # rename column
     col_names = ['gender', 'age', 'educational_level', 'institution_type', 'it_student', 'location', 'load_shedding', 'financial_condition', 'internet_type', 'network_type', 'class_duration', 'self_lms', 'device', 'adaptivity_level']
     df.columns = col_names
     col_names
     # print('Students Adaptability Level Online Education.csv (Col Renamed)')
     # print(df)
     # print('')

     # # frequency distribution of values in variables
     # print('Distribution of values in variables :')
     # for col in col_names:
     #      print(df[col].value_counts())  
     # print('')

     # # check missing values in variables
     # print('Missing values in variables:')
     # print(df.isnull().sum())
     # print('')

     # set target variabel
     X = df.drop(['adaptivity_level'], axis=1)
     y = df['adaptivity_level']

     # split X and y into training and testing sets
     from sklearn.model_selection import train_test_split
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

     # # check the shape of X_train and X_test
     # print('dimension X train & test :')
     # print(X_train.shape, X_test.shape)
     # print('')

     # # check data types in X_train
     # print('check data types in X_train :')
     # print(X_train.dtypes)
     # print('')

     # Encode categorical variables
     import category_encoders as ce
     encoder = ce.OrdinalEncoder(cols=['gender', 'age', 'educational_level', 'institution_type', 'it_student', 'location', 'load_shedding', 'financial_condition', 'internet_type', 'network_type', 'class_duration', 'self_lms', 'device'])
     X_train = encoder.fit_transform(X_train)
     X_test = encoder.transform(X_test)

     # DECISION TREE BY ENTROPY
     if radiobtn.get() == '0':
          # instantiate the DecisionTreeClassifier model with criterion entropy
          clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
          clf_en.fit(X_train, y_train)

          # Predict the Test set results with criterion entropy
          y_pred_en = clf_en.predict(X_test)

          # Check accuracy score with criterion entropy
          from sklearn.metrics import accuracy_score
          print('==============================================================================')
          print('')
          print('ENTROPY MODEL')
          print('Model accuracy score with criterion entropy: {0:0.4f}'. format(accuracy_score(y_test, y_pred_en)))

          # Compare the train-set and test-set accuracy
          y_pred_train_en = clf_en.predict(X_train)
          print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_en)))

          # Check for overfitting and underfitting
          # print the scores on training and test set
          print('Training set score: {:.4f}'.format(clf_en.score(X_train, y_train)))
          print('Test set score: {:.4f}'.format(clf_en.score(X_test, y_test)))

          # Visualize decision-trees
          from sklearn import tree
          fn=['gender', 'age', 'educational_level', 'institution_type', 'it_student', 'location', 'load_shedding', 'financial_condition', 'internet_type', 'network_type', 'class_duration', 'self_lms', 'device']
          cn= y_train.unique()
          fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (5,5), dpi=300)
          tree.plot_tree(clf_en.fit(X_train, y_train),
                         feature_names = fn, 
                         class_names=cn,
                         filled = True)
          fig.savefig('Etropy.png')
          plt.show()

          # Confusion matrix
          from sklearn.metrics import confusion_matrix
          cm = confusion_matrix(y_test, y_pred_en)
          print(cm)
          print('\nTrue Positives(TP) = ', cm[0,0])
          print('\nTrue Negatives(TN) = ', cm[1,1] + cm[1,2] + cm[2,1] + cm[2,2])
          print('\nFalse Positives(FP) = ', cm[1,0] + cm[2,0])
          print('\nFalse Negatives(FN) = ', cm[0,1] + cm[0,2])
          print('')

          # Classification Report
          from sklearn.metrics import classification_report
          print(classification_report(y_test, y_pred_en))
          print('')

     # DECISION TREE BY GINI
     elif radiobtn.get() == '1':
          # instantiate the DecisionTreeClassifier model with criterion gini index
          clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
          clf_gini.fit(X_train, y_train)

          # Predict the Test set results with criterion gini index
          y_pred_gini = clf_gini.predict(X_test)

          #Check accuracy score with criterion gini index
          from sklearn.metrics import accuracy_score
          print('==============================================================================')
          print('')
          print('GINI MODEL')
          print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))

          # Compare the train-set and test-set accuracy
          y_pred_train_gini = clf_gini.predict(X_train)
          print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_gini)))

          # Check for overfitting and underfitting
          # print the scores on training and test set
          print('Training set score: {:.4f}'.format(clf_gini.score(X_train, y_train)))
          print('Test set score: {:.4f}'.format(clf_gini.score(X_test, y_test)))
          print('')

          # Visualize decision-trees
          from sklearn import tree
          fn=['gender', 'age', 'educational_level', 'institution_type', 'it_student', 'location', 'load_shedding', 'financial_condition', 'internet_type', 'network_type', 'class_duration', 'self_lms', 'device']
          cn= y_train.unique()
          fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (5,5), dpi=300)
          tree.plot_tree(clf_gini.fit(X_train, y_train),
                         feature_names = fn, 
                         class_names=cn,
                         filled = True)
          fig.savefig('Gini.png')
          plt.show()

          # Confusion matrix
          from sklearn.metrics import confusion_matrix
          cm = confusion_matrix(y_test, y_pred_gini)
          print(cm)
          print('\nTrue Positives(TP) = ', cm[0,0])
          print('\nTrue Negatives(TN) = ', cm[1,1] + cm[1,2] + cm[2,1] + cm[2,2])
          print('\nFalse Positives(FP) = ', cm[1,0] + cm[2,0])
          print('\nFalse Negatives(FN) = ', cm[0,1] + cm[0,2])
          print('')

          # Classification Report
          from sklearn.metrics import classification_report
          print(classification_report(y_test, y_pred_gini))
          print('')

     
     # NAIVE BAYES GAUSSIAN
     elif radiobtn.get() == '2':
          print('==============================================================================')

          # Feature Scaling
          cols = X_train.columns
          
          scaler = RobustScaler()
          X_train = scaler.fit_transform(X_train)
          X_test = scaler.transform(X_test)
          X_train = pd.DataFrame(X_train, columns=[cols])
          X_test = pd.DataFrame(X_test, columns=[cols])
          print(X_train)

          # Model training
          # train a Gaussian Naive Bayes classifier on the training set
          from sklearn.naive_bayes import GaussianNB
          gnb = GaussianNB()
          gnb.fit(X_train, y_train)

          # Predict the results
          y_pred = gnb.predict(X_test)

          #Check accuracy score
          from sklearn.metrics import accuracy_score
          print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
          y_pred_train = gnb.predict(X_train)
          print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

          # print the scores on training and test set
          print('Training set score: {:.4f}'.format(gnb.score(X_train, y_train)))
          print('Test set score: {:.4f}'.format(gnb.score(X_test, y_test)))
          print('')

          # Classification Report
          from sklearn.metrics import classification_report
          print(classification_report(y_test, y_pred))
          print('')

          # Confusion matrix
          from sklearn.metrics import confusion_matrix
          cm = confusion_matrix(y_test, y_pred)

          print(cm)
          print('\nTrue Positives(TP) = ', cm[0,0])
          print('\nTrue Negatives(TN) = ', cm[1,1] + cm[1,2] + cm[2,1] + cm[2,2])
          print('\nFalse Positives(FP) = ', cm[1,0] + cm[2,0])
          print('\nFalse Negatives(FN) = ', cm[0,1] + cm[0,2])
     else:
          print('Please Choose Algorithm')

# gui
gui = tk.Tk()
gui.title('Students Adaptability Level in Online Education')
radiobtn = tk.StringVar()

frame = tk.Frame(gui, width=500, height=350)
frame.pack()

# input dir
label = tk.Label(gui, text="Dataset :", fg='black', font=("Helvetica", 15))
label.place(x=50, y=40)

label2 = tk.Label(gui, text="", fg='black', font=("Helvetica", 15))
label2.place(x=140, y=40)

btn = tk.Button(gui, text="Choose Dataset File", fg='black', font=("Helvetica", 15), command=selectfile)
btn.place(x=50, y=70)

# choose algorithm
label3 = tk.Label(gui, text="Choose an Algorithm :", fg='black', font=("Helvetica", 15))
label3.place(x=50, y=130)

list1 = tk.Radiobutton(gui, text="Decision Tree by Entropy", variable=radiobtn, value="0", fg='black', font=("Helvetica", 15))
list1.place(x=50, y=160)

list2 = tk.Radiobutton(gui, text="Decision Tree by Gini", variable=radiobtn, value="1", fg='black', font=("Helvetica", 15))
list2.place(x=50, y=190)

list3 = tk.Radiobutton(gui, text="Naive Bayes Gaussian", variable=radiobtn, value="2", fg='black', font=("Helvetica", 15))
list3.place(x=50, y=220)

# run
btn1 = tk.Button(gui, text="Run", fg='black', font=("Helvetica", 15), command=predict)
btn1.place(x=50, y=280)

gui.mainloop()