"""
The following code is not completely my own:
    PCA Analysis - Oluwafemi Oyedeji
    Histogram - Connor Curtis
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)


def fraud_hist(df, bin_cnt):
    """
    Create histograms for 'Amount' seperated by fraud, non-fraud, and total
    :param df: Credit card transaction dataframe
    :param bin_cnt: Number of bins in histograms
    :return: NULL
    """
    fraud = df[df['Class'] == 1]
    notFraud = df[df['Class'] == 0]

    fig, axes = plt.subplots(1, 3)
    ax = axes.ravel()
    ax[0].hist(df['Amount'], bins=bin_cnt)
    ax[0].set_title("All transactions")
    ax[1].hist(fraud['Amount'], bins=bin_cnt)
    ax[1].set_title("Fraud transactions")
    ax[2].hist(notFraud['Amount'], bins=bin_cnt)
    ax[2].set_title("Non-Fraud transactions")

    plt.tight_layout()


def fraud_model(V, y, rep=10, breakdown=False, alph=0, threshold=0.5, ridge=False, ):
    '''
    Create a number of models and find the average
    :param V: Known parameters
    :param y: Target parameter
    :param rep: Number of repititions
    :param breakdown: Boolean condition to print true/false positive/negative
    :param alph: lambda value
    :param threshold: where to round initial output up to 1
    :param ridge: boolean condition to use ridge or LASSO regression
    :return: NULL
    '''
    model_type = ["linear", "ridge", "LASSO"]

    acc = []
    tpr = []
    tnr = []
    fpr = []
    fnr = []

    for i in range(rep):
        V_train, V_test, y_train, y_test = train_test_split(V, y, test_size=0.2)

        if alph == 0:
            res = LinearRegression().fit(V_train, y_train)
            moddex = 0
        elif ridge:
            res = Ridge(alph).fit(V_train, y_train)
            moddex = 1
        else:
            res = Lasso(alph).fit(V_train, y_train)
            moddex = 2

        predictions = res.predict(V_test)
        predictions = (predictions >= threshold).astype(int)
        cnt = np.sum(predictions == y_test)
        acc.append(cnt / len(predictions))

        # Breakdown of True/False Positive/Negative Rate
        zeros = np.zeros(len(predictions))
        ones = zeros + 1

        p = np.sum(predictions == ones)
        n = np.sum(predictions == zeros)
        tp = np.sum((predictions == ones) & (predictions == y_test))
        tn = np.sum((predictions == zeros) & (predictions == y_test))
        fp = p - tp
        fn = n - tn
        tpr.append(tp / (tp + fn))
        tnr.append(tn / (tn + fp))
        fpr.append(fp / (fp + tn))
        fnr.append(fn / (fn + tp))

    if breakdown:
        print("Mean", model_type[moddex], "Regression Model accuracy:", (round(np.mean(acc) * 100, 1)), '%')
        print("True positive rate:", round(np.mean(tpr) * 100, 1), '%')
        print("True negative rate:", round(np.mean(tnr) * 100, 1), '%')
        print("False positive rate:", round(np.mean(fpr) * 100, 1), '%')
        print("False negative rate:", round(np.mean(fnr) * 100, 1), '%')
        print()
    else:
        print("Mean", model_type[moddex], "regression model accuracy:", (round(np.mean(acc) * 100, 1)), '%')
        print()


#
def max_lambda(V, y):
    """
    Finding largest lamda for sparse LASSO regression model while maintaining above a 90% accuracy
    :param V: features for model training and testing input
    :param y: target for model training and testing output
    :return: NULL
    """
    alph = 0.1
    acc = 1
    while acc >= 0.90:
        V_train, V_test, y_train, y_test = train_test_split(V, y, test_size=0.2)
        alph += 0.1
        res = Lasso(alpha=alph).fit(V_train, y_train)

        predictions = res.predict(V_test)
        predictions = (predictions >= 0.5).astype(int)

        cnt = np.sum(predictions == y_test)

        acc2 = cnt / (len(y_test))
        if acc2 < 0.90:
            alph -= 0.1
            break
        else:
            acc = acc2

    print("Accuracy:", round(acc, 3) * 100, '%')
    print("Lamda:", alph)
