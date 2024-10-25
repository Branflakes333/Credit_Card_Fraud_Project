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
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

# Options
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

# Global Variables
df = None
target = None
features = None
pca = None
X_pca = None
num_components = None
V_train = None
V_test = None 
y_train = None 
y_test = None


def set_dataframe(dat):
    """
    Sets the global variable `df` to `dat` for access in other functions.

    :param dat (pd.DataFrame): DataFrame of data of interest.

    :return: None
    """
    global df
    df = dat


def set_target_and_features(tar):
    """
    Sets the global variable `target` to `tar` for access in other functions.

    :param tar (pd.Column): Column of target variable

    :return: None
    """
    global target
    target = tar

    global features
    features = df.columns.drop([target]).tolist()


def get_features():
    return features


def prelim_pca():
    """
    Perform preliminary Principal Component Analysis (PCA) on the standardized features.

    This function scales the input features using StandardScaler, 
    applies PCA to reduce dimensionality, and determines the number 
    of principal components based on an eigenvalue threshold of 1.

    This function relies on globally defined variables:
    - DataFrame `df`
    - list `features`

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
      scaler.fit_transform(df[features]),
      columns=features
    )

    # Perform preliminary PCA
    global pca
    pca = PCA()
    global X_pca
    X_pca = pd.DataFrame(
        pca.fit_transform(X_scaled),
        columns=[f'PC{i + 1}' for i in range(X_scaled.shape[1])]
    )

    # Choose number of components based on eigenvalues threshold of 1
    global num_components
    num_components = np.argmax(pca.explained_variance_ < 1)
    assert num_components > 0, 'Number of components must be > 0'


def setup(dat, target):
    """
    Defines all global variables by calling 3 functions:
    - set_dataframe(dat) to define `df`
    - features(target) to define `features`
    - prelim_pca() to define `pca`, `X_pca`, and `num_components`

    :param dat (pd.DataFrame): DataFrame of data of interest. 
    :param target (str): The column name in the data frame that is the target variable. 

    :return: None
    """

    set_dataframe(dat)
    set_target_and_features(target)
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

    fig, axes = plt.subplots(3, 1)
    ax = axes.ravel()
    ax[0].hist(df['Amount'], bins=bin_cnt, density = True)
    ax[0].set_title("All transactions")
    ax[1].hist(fraud['Amount'], bins=bin_cnt, density = True)
    ax[1].set_title("Fraud transactions")
    ax[2].hist(notFraud['Amount'], bins=bin_cnt, density = True)
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


def pca_biplot():
    """
    Create and plot a Biplot of Principal Component 2's effect of Principal Component 1 (check for error)

    :param: None 
    
    :return: None 
    """

    # Visualize PCA biplot
    fig, ax1 = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    # Iterate over features to plot arrows and labels on the biplot
    for i, feature in enumerate(features):
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


def fraud_model(V, y):
    '''
    Create a number of models and find the average

    :V dataframe: Dataframe of all feature data
    :y dataframe: Dataframe of all target data

    :return: None
    '''
    # Create Logistic Regression Model
    logistic = SGDClassifier(
        loss = 'log_loss', 
        max_iter = 10000,
        learning_rate = 'optimal'
    )

    # Create training testing split for features and target data
    global V_train, V_test, y_train, y_test
    V_train, V_test, y_train, y_test = train_test_split(V, y, test_size=0.2, random_state = 333)

    # Train the logistic model on training data
    logistic.fit(V_train, y_train)
    return logistic


def calculate_metrics(model, V, y, cutoff):
    '''
    Calculate accuracy, False Positive Rate, and False Negative Rate
    :model: logistic regression model
    :V: features data frame
    :y: target data frame
    :cutoff: Prediction probability rounding number 

    :return accuracy, FPR, FNR: accuracy, false positive rate, and false negative rate of model given data set
    '''
    proba = model.predict_proba(V)[:, 1]  # Probabilities for the positive class (1)
    predictions = (proba >= cutoff).astype(int).astype(bool)
        
    # Accuracy
    accuracy = np.mean(predictions == y)
        
    # Confusion matrix components
    TP = np.sum((predictions == 1) & (y == 1))  # True Positives
    TN = np.sum((predictions == 0) & (y == 0))  # True Negatives
    FP = np.sum((predictions == 1) & (y == 0))  # False Positives
    FN = np.sum((predictions == 0) & (y == 1))  # False Negatives
        
    # False Positive Rate (FPR) and False Negative Rate (FNR)
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
        
    return accuracy, FPR, FNR
    

def model_accuracy(model, cutoff=0.5):
    '''
    Table of model accuracy, false negative rate, and false positive rate for training and testing split
    :model: logistic regression model
    :cutoff: Prediction probability rounding number

    :return metrics_df: data frame of metrics for model
    '''
    # Calculate metrics for training and testing datasets
    train_accuracy, train_FPR, train_FNR = calculate_metrics(model, V_train, y_train, cutoff)
    test_accuracy, test_FPR, test_FNR = calculate_metrics(model, V_test, y_test, cutoff)
    
    # Create a DataFrame to display the results
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'False Positive Rate (FPR)', 'False Negative Rate (FNR)'],
        'Training Data': [f'{train_accuracy * 100:.2f}%', f'{train_FPR * 100:.2f}%', f'{train_FNR * 100:.2f}%'],
        'Testing Data': [f'{test_accuracy * 100:.2f}%', f'{test_FPR * 100:.2f}%', f'{test_FNR * 100:.2f}%']
    })
    
    return metrics_df


def accuracy_FNR_plt(model):
    '''
    Plots the train and test accuracies, and False Negative Rates (FNRs)
    for different cutoff values, with train and test data on the same graphs.
    
    :param model: Trained model

    :return: None
    '''
    # Setup lists
    train_accuracies_list = []
    train_FNRs_list = []
    test_accuracies_list = []
    test_FNRs_list = []
    cutoffs = np.arange(0.001, 0.50, 0.005)

    # Calculate metrics for each cutoff and store in lists
    for co in cutoffs:
        train_accuracy, train_FPR, train_FNR = calculate_metrics(model, V_train, y_train, co)
        test_accuracy, test_FPR, test_FNR = calculate_metrics(model, V_test, y_test, co)

        train_accuracies_list.append(train_accuracy)
        train_FNRs_list.append(train_FNR)
        test_accuracies_list.append(test_accuracy)
        test_FNRs_list.append(test_FNR)

    # Create a 1x2 grid for subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot accuracy on the first subplot
    axs[0].plot(cutoffs, train_accuracies_list, color='blue', label='Train Accuracy')
    axs[0].plot(cutoffs, test_accuracies_list, color='green', label='Test Accuracy')
    axs[0].axhline(y=0.90, color='gray', linestyle='--', linewidth=1)  # Add horizontal dashed line at y=0.90
    axs[0].axvline(x=0.072, color='gray', linestyle='--', linewidth=1)  # Add vertical dashed line at x=0.072
    axs[0].set_title('Train and Test Accuracy vs Cutoff')
    axs[0].set_xlabel('Cutoff')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    # Plot FNR on the second subplot
    axs[1].plot(cutoffs, train_FNRs_list, color='red', label='Train FNR')
    axs[1].plot(cutoffs, test_FNRs_list, color='purple', label='Test FNR')
    axs[1].set_title('Train and Test FNR vs Cutoff')
    axs[1].set_xlabel('Cutoff')
    axs[1].set_ylabel('FNR')
    axs[1].legend()

    plt.tight_layout()  # Adjusts subplot layout for better spacing
    plt.show()
