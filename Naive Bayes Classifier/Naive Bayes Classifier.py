# Imported libraries
import pandas as pd
import numpy as np
import time as tm
from os import system, name
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# This class houses all the labels and informational data used during interaction.
class Info_Pool:
    Info1 = {'Inf1': "\nOption #1: Naive Bayes Classifier\n",
             'Inf2': "This is a simple  Naive Bayes classifier that uses the load_brest_cancer dataset from "
                     "URL: https://scikit-learn.org/stable/datasets/toy_dataset.html\n"}

    Interactive = {'Data1': "1. Gaussian Naive Bayes Prediction & accuracy",
                   'Data2': "2. Multinomial Naive Bayes Prediction & accuracy",
                   'Data3': "3. Posterior probability by converting the dataset into a frequency table.",
                   'Data4': "4. Likelihood table w/ relevant probabilities",
                   'Data5': "5. Probability errors correction w/ Laplacian correction",
                   'Data6': "6. Quit\n",
                   'Data7': "Enter Option: "}

    classes = {'0': "Malignant = 0", '1': "Benign = 1"}

    sys1 = 'nt'
    sys2 = 'clear'
    sys3 = 'cls'


# This function clears the console terminal.
def clr_T():
    if name == Info_Pool.sys1:
        _ = system(Info_Pool.sys3)
    else:
        _ = system(Info_Pool.sys2)


# This class loads the dataset and trains the model with the same.
class mainDataSet:
    mainDat_Set = load_breast_cancer()
    X = mainDat_Set.data
    y = mainDat_Set.target
    Data_info = Info_Pool.classes['0']
    Data_info_2 = Info_Pool.classes['1']
    Info1 = f'\nClasses/ features Information: {Data_info, Data_info_2}\n'
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# This class houses the naive bayes gaussian classifier.
class naive_bayes_Gaussian:
    gau = GaussianNB()
    gau.fit(mainDataSet.X_train, mainDataSet.y_train)
    prediction_y = gau.predict(mainDataSet.X_test)
    accuracy_1 = accuracy_score(mainDataSet.y_test, prediction_y)
    accuracy_Data1 = f'\nAccuracy_Gaussian: {accuracy_1:.2f}\n'
    prediction_Data1 = f'\nGaussian Prediction: {prediction_y}\n'


# This class houses the naive bayes Multinomial classifier.
class naive_bayes_Multinomial:
    multi_nom = MultinomialNB()
    multi_nom.fit(mainDataSet.X_train, mainDataSet.y_train)
    predictor2_y = multi_nom.predict(mainDataSet.X_test)
    accuracy_2 = accuracy_score(mainDataSet.y_test, predictor2_y)
    accuracy_Data2 = f'\nAccuracy_MultinomialNB: {accuracy_2:.2f}\n'
    prediction_Data2 = f'\nMultinomial Prediction: {predictor2_y}\n'


# This class houses the posterior probability frequency table.
class posterior_probability_dataFrame:
    posterior_probs = naive_bayes_Gaussian.gau.predict_proba(mainDataSet.X_test)
    posteriorProb_df = pd.DataFrame(data=posterior_probs, columns=naive_bayes_Gaussian.gau.classes_)
    posteriorProb_df_2 = posteriorProb_df.rename(columns={0: "Malignant", 1: "Benign"})
    posteriorProb_Data = f'\nPosterior Probability frequency table:\n{posteriorProb_df_2}\n'


# This class houses the Likelihood table that generates the relevant probability.
class likelihood_table_dataframe:
    num_classes = len(np.unique(mainDataSet.y_test))
    num_features = mainDataSet.X.shape[0]
    likelihood_table = np.zeros((num_classes, num_features))

    for table in range(num_classes):
        class_indices = np.where(mainDataSet.y_test == table)[0]
        X_class = mainDataSet.X_test[class_indices]

        means = X_class.mean(axis=0)
        stds = X_class.std(axis=0)

        likelihood_table = means

    likelihood_table_Data = f'\nLikelihood table:\n{likelihood_table}\n'


# This class contains the Zero Probability errors correction that uses Laplacian correction.
class Laplacian_correction:
    clf = naive_bayes_Multinomial.multi_nom
    clf.fit(mainDataSet.X_train, mainDataSet.y_train)

    prediction = clf.predict(mainDataSet.X_test)
    prediction_Data = f'\nPrediction w/Laplacian correction:\n{prediction}\n'


# This function activates the gaussian prediction and accuracy measurement when selected.
def gaussian():
    clr_T()
    dat_set1 = mainDataSet.Info1
    dat_set2 = naive_bayes_Gaussian.prediction_Data1
    dat_set3 = naive_bayes_Gaussian.accuracy_Data1

    print(dat_set1, dat_set2, dat_set3)
    tm.sleep(2)


# This function activates the multinomial prediction and accuracy measurement when selected.
def multinomialNB():
    clr_T()
    d_set = mainDataSet.Info1
    d_set1 = naive_bayes_Multinomial.prediction_Data2
    d_set2 = naive_bayes_Multinomial.accuracy_Data2

    print(d_set, d_set1, d_set2)
    tm.sleep(2)


# This function generates the posterior table when selected.
def posterior_prob():
    clr_T()
    post_prob1 = posterior_probability_dataFrame.posteriorProb_Data

    print(post_prob1)
    tm.sleep(2)


# This function generates the likelihood table when selected.
def likelihood_table():
    clr_T()
    table = likelihood_table_dataframe.likelihood_table_Data
    print(table)
    tm.sleep(2)


# This activates the zero probability errors correction
def correction():
    clr_T()
    L_Info = mainDataSet.Info1
    L_correction = Laplacian_correction.prediction_Data

    print(L_Info, L_correction)
    tm.sleep(2)


# This class houses the interactive selection menu.
class usr_Interaction:
    while True:
        clr_T()
        print(Info_Pool.Info1['Inf1'])
        print(Info_Pool.Info1['Inf2'])
        print(Info_Pool.Interactive['Data1'])
        print(Info_Pool.Interactive['Data2'])
        print(Info_Pool.Interactive['Data3'])
        print(Info_Pool.Interactive['Data4'])
        print(Info_Pool.Interactive['Data5'])
        print(Info_Pool.Interactive['Data6'])
        print(Info_Pool.Interactive['Data7'], end='')

        user_input = input()

        if user_input == '1':
            gaussian()

        elif user_input == '2':
            multinomialNB()
        elif user_input == '3':
            posterior_prob()
        elif user_input == '4':
            likelihood_table()
        elif user_input == '5':
            correction()
        elif user_input == '6':
            quit()


# this function runs the entire algorithm.
if __name__ == "main":
    usr_Interaction()
