"""
This project was worked on by Brandon Miner and Connor Payne (2023)

The following code is not completely our own:
    PCA Analysis - Oluwafemi Oyedeji
"""
# Data Manip
import pandas as pd
import numpy as np

# Visuals
import matplotlib.pyplot as plt

# PCA Analysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Modeling
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

# Options
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

# Global Variables
df = None
feature_names = None
pca = None
X_pca = None
num_components = None


def set_dataframe(dat):
    """
    Sets the global variable `df` to dat for access in other functions.

    :param dat (pd.DataFrame): DataFrame of data of interest.

    :return: None
    """

    df = dat


def features(target):
    """
    Gets a list of feature column names, excluding the target, from a generic DataFrame.
    Sets the list of features to the gobal variable `feature_names`.
    Used in this project to obtain a list of features from the credit_card_fraud_2023.csv dataset.

    :param target (str): The column name in the data frame that is the target variable.

    :return: None
    """

    feature_list = [
        name
        for name in df.columns
            if name not in target
    ]
    feature_names = feature_list


def prelim_pca():
    """
    Perform preliminary Principal Component Analysis (PCA) on the standardized features.

    This function scales the input features using StandardScaler, 
    applies PCA to reduce dimensionality, and determines the number 
    of principal components based on an eigenvalue threshold of 1.

    This function relies on globally defined variables:
    - DataFrame `df`
    - list `feature_names`

    This function sets three globally defined variables:
            - pca (PCA): The fitted PCA object.
            - X_pca (pd.DataFrame): A DataFrame of the transformed features 
              in the principal component space, with columns named 'PC1', 'PC2', etc.
            - num_components (int): The number of principal components selected 
              based on the explained variance threshold.

    :param: None

    :return: None
    """

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
      scaler.fit_transform(df[feature_names]),
      columns=feature_names
    )

    # Perform preliminary PCA
    pca = PCA()
    X_pca = pd.DataFrame(
        pca.fit_transform(X_scaled),
        columns=[f'PC{i + 1}' for i in range(X_scaled.shape[1])]
    )

    # Choose number of components based on eigenvalues threshold of 1
    num_components = np.argmax(pca.explained_variance_ < 1)
    assert num_components > 0, 'Number of components must be > 0'


def setup(dat, target):
    """
    Defines all global variables by calling 3 functions:
    - set_dataframe(dat) to define `df`
    - features(target) to define `feature_names`
    - prelim_pca() to define `pca`, `X_pca`, and `num_components`

    :param dat (pd.DataFrame): DataFrame of data of interest. 
    :param target (str): The column name in the data frame that is the target variable. 

    :return: None
    """

    set_dataframe(dat)
    features(target)
    prelim_pca()


def fraud_hist(bin_cnt):
    """
    Create histograms for 'Amount' separated by fraud, non-fraud, and total transactions.
    Number of bins of histogram specified by input from user.
    Plots three histograms side-by-side.

    :param bin_cnt (int): Number of bins to use in the histograms.
    
    :return: None 
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
    plt.show()


def scree_plt():
    """
    Create and plot a Scree-plot of Principal Components of data set.

    :param: None 
    
    :return: None 
    """ 

    # Visualize scree plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), tight_layout=True)

    # Explained Variance Ratio for Each Principal Component
    ax1.bar(
        x=range(1, len(pca.explained_variance_ratio_[:num_components]) + 1),
        height=pca.explained_variance_ratio_[:num_components],
        width=0.5,
    )

    # Set axes properties
    ax1.set_ylim(0.0, pca.explained_variance_ratio_[0] * 1.1)
    ax1.set_xlabel('Principal Components')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Scree Plot - Explained Variance Ratio')

    # Cumulative explained variance ratio plot
    ax2.plot(
        range(1, len(pca.explained_variance_ratio_[:num_components]) + 1),
        np.cumsum(pca.explained_variance_ratio_[:num_components]),
        'o-'
    )
    # Set axes properties
    ax2.set_ylim(
        bottom=pca.explained_variance_ratio_[0] * 0.9,
        top=np.cumsum(pca.explained_variance_ratio_[:num_components])[-1] * 1.1
    )
    ax2.set_xlabel('Principal Components')
    ax2.set_ylabel('Cumulative Explained Variance Ratio')
    ax2.set_title('Cumulative Explained Variance')

    # Display the plots
    plt.show()
    print(pca.explained_variance_ratio_)


def pca_biplot():
    """
    Create and plot a Biplot of Principal Component 2's effect of Principal Component 1 (check for error)

    :param: None 
    
    :return: None 
    """

    # Visualize PCA biplot
    fig, ax1 = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    # Iterate over features to plot arrows and labels on the biplot
    for i, feature in enumerate(feature_names):
        # Plot arrows representing feature contributions to PC1 and PC2
        ax1.arrow(
            x=0,
            y=0,
            dx=pca.components_[0, i],
            dy=pca.components_[1, i],
            color='r',
            alpha=0.5,
            head_width=0.01,
            head_length=0.01,
        )
        # Annotate each arrow with the corresponding feature name
        ax1.text(
            pca.components_[0, i] * 1.3,
            pca.components_[1, i] * 1.3,
            feature,
            color='g'
        )
    # Set axes properties
    ax1.set_xlim(-0.4, 0.4)
    ax1.set_ylim(-0.8, 0.8)
    ax1.set_xlabel('Principal Component 1')
    ax1.set_ylabel('Principal Component 2')
    ax1.set_title('PCA Biplot')

    # Display the biplot
    plt.show()


def pca_scatter():
    """
    Plot scatter plot of transactions projected to the first two principal components.
    Points colored based on whether the transaction was fraudulent or not.

    :param: None 
    
    :return: None 
    """

    # Map class labels to colors
    colors = df['Class'].apply(lambda x: '#FF6F69' if x == 1 else '#608654')

    # Handle plot layout
    fig, ax = plt.subplots(figsize=(6, 4))

    # Create a scatter plot using Matplotlib
    scatter = ax.scatter(
        X_pca['PC1'], 
        X_pca['PC2'], 
        c=colors, 
        edgecolor='w', 
        linewidth=0.5, 
        s=25, 
        alpha=0.8
    )

    # Set labels for axes
    ax.set_xlabel('Principal component 1')
    ax.set_ylabel('Principal component 2')
    ax.set_xlim(-10, 70)
    ax.set_ylim(-30, 70)

    # Add a custom legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6F69', markersize=10, label='Fraud'),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#608654', markersize=10, label='Non-Fraud')]
    ax.legend(handles=handles, loc='upper left')

    # Display the plot
    plt.show()


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

    :return: None
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

