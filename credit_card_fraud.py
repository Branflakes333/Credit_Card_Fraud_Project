"""
This project was worked on by Brandon Miner and Connor Payne (2023)

The following code is not completely our own:
    PCA Analysis - Oluwafemi Oyedeji
"""
# Data Manip
import pandas as pd
import numpy as np
import seaborn as sns

# Visuals
import matplotlib.pyplot as plt

# PCA Analysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

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


def covariance_plt():
    """
    Calculate and plot covarianve matrix.
    Points colored based on Severity of variance/covariance

    :param: None 
    
    :return: None 
    """

    # Calculate covariance matrix
    cov_matrix = df.cov()
    
    # Set up the matplotlib figure with a larger size for more variables
    plt.figure(figsize=(15, 12))
    
    # Use seaborn's heatmap to plot
    sns.heatmap(cov_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, 
                cbar_kws={"shrink": .7}, xticklabels=cov_matrix.columns, yticklabels=cov_matrix.columns)
    
    # Rotate the x and y labels to prevent overlap
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    # Add title
    plt.title('Covariance Matrix')
    
    # Show plot
    plt.tight_layout()  # Adjust layout to make room for labels
    plt.show()


def fraud_model(V, y):
    '''
    Create a logistic regression model using SGDClassifier with log loss.

    :V: DataFrame of all feature data
    :y: DataFrame of all target data

    :return: Trained logistic regression model
    '''
    logistic = SGDClassifier(
        loss='log_loss',
        max_iter=10000,
        learning_rate='optimal',
        random_state=333
    )

    # Training/testing split
    global V_train, V_test, y_train, y_test
    V_train, V_test, y_train, y_test = train_test_split(V, y, test_size=0.2, random_state=333)

    logistic.fit(V_train, y_train)
    return logistic


def calculate_metrics(model, V, y, cutoff):
    '''
    Calculate accuracy, FPR, and FNR using predictions based on a cutoff.

    :model: Logistic regression model
    :V: Features data
    :y: Target data
    :cutoff: Cutoff threshold for classifying positive predictions

    :return: Accuracy, FPR, and FNR
    '''
    proba = model.predict_proba(V)[:, 1]  # Probability for the positive class
    predictions = (proba >= cutoff)
    
    accuracy = accuracy_score(y, predictions)
    tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()

    FPR = fp / (fp + tn) if (fp + tn) > 0 else 0
    FNR = fn / (fn + tp) if (fn + tp) > 0 else 0

    return accuracy, FPR, FNR


def model_accuracy(model, cutoff=0.5):
    '''
    Display model accuracy, FPR, and FNR for train and test data.

    :model: Trained logistic regression model
    :cutoff: Cutoff threshold for positive predictions

    :return: DataFrame with metrics for training and testing sets
    '''
    train_metrics = calculate_metrics(model, V_train, y_train, cutoff)
    test_metrics = calculate_metrics(model, V_test, y_test, cutoff)

    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'False Positive Rate (FPR)', 'False Negative Rate (FNR)'],
        'Training Data': [f'{train_metrics[0] * 100:.2f}%', f'{train_metrics[1] * 100:.2f}%', f'{train_metrics[2] * 100:.2f}%'],
        'Testing Data': [f'{test_metrics[0] * 100:.2f}%', f'{test_metrics[1] * 100:.2f}%', f'{test_metrics[2] * 100:.2f}%']
    })

    return metrics_df


def accuracy_FNR_plt(model):
    '''
    Plot train and test accuracies, and FNRs across cutoff thresholds.

    :model: Trained model

    :return: None
    '''
    cutoffs = np.arange(0.001, 0.50, 0.005)
    train_accuracies, train_FNRs, test_accuracies, test_FNRs = [], [], [], []

    for cutoff in cutoffs:
        train_metrics = calculate_metrics(model, V_train, y_train, cutoff)
        test_metrics = calculate_metrics(model, V_test, y_test, cutoff)

        train_accuracies.append(train_metrics[0])
        train_FNRs.append(train_metrics[2])
        test_accuracies.append(test_metrics[0])
        test_FNRs.append(test_metrics[2])

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy plot
    axs[0].plot(cutoffs, train_accuracies, color='blue', label='Train Accuracy')
    axs[0].plot(cutoffs, test_accuracies, color='green', label='Test Accuracy')
    axs[0].axhline(y=0.90, color='gray', linestyle='--')
    axs[0].axvline(x=0.078, color='gray', linestyle='--')
    axs[0].set_title('Train and Test Accuracy vs Cutoff')
    axs[0].set_xlabel('Cutoff')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    # FNR plot
    axs[1].plot(cutoffs, train_FNRs, color='red', label='Train FNR')
    axs[1].plot(cutoffs, test_FNRs, color='purple', label='Test FNR')
    axs[1].set_title('Train and Test FNR vs Cutoff')
    axs[1].set_xlabel('Cutoff')
    axs[1].set_ylabel('FNR')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def calculate_aic(model):
    '''
    Calculate the Akaike Information Criterion (AIC) for a logistic regression model.

    :model: Trained logistic regression model

    :return: AIC value
    '''
    # Step 1: Predicted probabilities for the positive class
    proba = model.predict_proba(V_train)[:, 1]

    # Step 2: Log-likelihood calculation
    epsilon = 1e-10  # To avoid log(0)
    log_likelihood = np.sum(y_train * np.log(proba + epsilon) + (1 - y_train) * np.log(1 - proba + epsilon))

    # Step 3: Number of parameters
    k = V_train.shape[1] + 1  # Number of features + intercept

    # Step 4: AIC formula
    aic_train = 2 * k - 2 * log_likelihood

    # Step 1: Predicted probabilities for the positive class
    proba = model.predict_proba(V_test)[:, 1]

    # Step 2: Log-likelihood calculation
    epsilon = 1e-10  # To avoid log(0)
    log_likelihood = np.sum(y_test * np.log(proba + epsilon) + (1 - y_test) * np.log(1 - proba + epsilon))

    # Step 3: Number of parameters
    k = V_test.shape[1] + 1  # Number of features + intercept

    # Step 4: AIC formula
    aic_test = 2 * k - 2 * log_likelihood

    return aic_train, aic_test
