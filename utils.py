#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import libraries
import numpy as np 
import time
import scipy.io as sio
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler,SMOTE
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score,auc,RocCurveDisplay,precision_recall_curve,PrecisionRecallDisplay,plot_precision_recall_curve
from sklearn.metrics import balanced_accuracy_score,plot_confusion_matrix,precision_score,recall_score
from imblearn.over_sampling import RandomOverSampler,SMOTE
from sklearn.metrics import precision_score,recall_score,f1_score,  precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings('ignore')
from matplotlib.ticker import NullFormatter
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
import matplotlib.colors
import matplotlib.colors as mcolors

# To plot consistent and pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
mpl.rcParams['font.family'] = 'times new roman'
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['figure.dpi'] = 120
mpl.rcParams['savefig.dpi'] = 120

# The code contains a dimensionality reduction function (DR) that can be used to reduce the feature space of a dataset to 2 dimensions using various methods such as t-SNE, PCA, or LDA. The names list contains the names of different classifiers, and the classifiers list contains instances of those classifiers. The min_max_scaler is used to scale the features, and the smote is used for handling class imbalance using the SMOTE technique. The variable n_components is set to 2 as the default number of components after dimensionality reduction. The DR function applies the chosen dimensionality reduction method to the data (X) and returns the transformed data (X) and the target labels (y).

# Dimensionality reduction function for reducing the feature space to 2 dimensions
def DR(X, y, method, n_components=2):
    # Apply t-SNE for dimensionality reduction if the method is 'TSNE'
    if method == 'TSNE':
        X = TSNE(n_components=n_components, perplexity=100).fit_transform(X) #method='exact',
    # Apply PCA for dimensionality reduction if the method is 'PCA'
    if method == 'PCA':
        X = PCA(n_components=n_components).fit_transform(X)
    # Apply LDA for dimensionality reduction if the method is 'LDA'
    if method == 'LDA':
        X = LinearDiscriminantAnalysis(n_components=n_components).fit(X, y).transform(X)
    return X, y

# List of classifier names
names = ["KNN", "SVM", "DT", "RF", "MLP", "AdaBoost"]

# List of classifiers
classifiers = [
    KNeighborsClassifier(3),
    SVC(gamma=2, C=1, probability=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
]

# Min-Max Scaler to scale the features in the range [0, 1]
min_max_scaler = MinMaxScaler()

# SMOTE (Synthetic Minority Over-sampling Technique) for handling class imbalance
smote = SMOTE(random_state=0)

# Number of components after dimensionality reduction
n_components = 2

# Dimensionality reduction function for reducing the feature space to 2 dimensions
def DR(X, y, method, n_components=n_components):
    # Apply t-SNE for dimensionality reduction if the method is 'TSNE'
    if method == 'TSNE':
        X = TSNE(n_components=n_components, perplexity=100).fit_transform(X) #method='exact',
    # Apply PCA for dimensionality reduction if the method is 'PCA'
    if method == 'PCA':
        X = PCA(n_components=n_components).fit_transform(X)
    # Apply LDA for dimensionality reduction if the method is 'LDA'
    if method == 'LDA':
        X = LinearDiscriminantAnalysis(n_components=n_components).fit(X, y).transform(X)
    return X, y

# These functions (option1, option2, option3, and option4) perform different classification approaches and evaluate the performance of classifiers based on various metrics. Option 1 involves classification without any preprocessing or dimensionality reduction. Option 2 applies the Synthetic Minority Over-sampling Technique (SMOTE) to handle class imbalance. Option 3 performs classification after applying dimensionality reduction using the specified method (t-SNE, PCA, or LDA). Option 4 combines both preprocessing and dimensionality reduction before classification. The functions return a list of results, where each result contains the classifier name and the average performance metrics over multiple trials.

# Option 1: Perform classification without any data preprocessing or dimensionality reduction.
def option1(X, y):
    # Use preprocessing (MinMaxScaler or some other techniques.)
    X = min_max_scaler.fit_transform(X) 
    results = []
    for name, clf in zip(names, classifiers):
        res = np.zeros((n_trial, 6))
        for i in range(n_trial):
            start = time.time()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)
            accuracy = balanced_accuracy_score(y_test, y_pred).round(2)
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted').round(2)
            pre = precision_score(y_test, y_pred, average='weighted').round(2)
            rec = recall_score(y_test, y_pred, average='weighted').round(2)
            f1 = f1_score(y_test, y_pred, average='weighted').round(2)
            endtime = time.time() - start
            res[i, :] = [pre, rec, f1, accuracy, auc, endtime]
        results.append([name, res.mean(axis=0).round(2)])
    return results

# Option 2: Perform classification with Synthetic Minority Over-sampling Technique (SMOTE) for handling class imbalance.
def option2(X, y):
    # Use preprocessing (MinMaxScaler or some other techniques.)
    X = min_max_scaler.fit_transform(X) 
    results = []
    for name, clf in zip(names, classifiers):
        res = np.zeros((n_trial, 6))
        for i in range(n_trial):
            start = time.time()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
            # Apply Synthetic Minority Over-sampling Technique on the training data
            X_train, y_train = smote.fit_resample(X_train, y_train)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)
            accuracy = balanced_accuracy_score(y_test, y_pred).round(2)
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted').round(2)
            pre = precision_score(y_test, y_pred, average='weighted').round(2)
            rec = recall_score(y_test, y_pred, average='weighted').round(2)
            f1 = f1_score(y_test, y_pred, average='weighted').round(2)
            endtime = time.time() - start
            res[i, :] = [pre, rec, f1, accuracy, auc, endtime]
        results.append([name, res.mean(axis=0).round(2)])
    return results

# Option 3: Perform classification after applying dimensionality reduction using the specified method.
def option3(X, y, method, n_components=n_components):
    # Use preprocessing (MinMaxScaler or some other techniques.) 
    X = min_max_scaler.fit_transform(X)
    results = []
    for name, clf in zip(names, classifiers):
        res = np.zeros((n_trial, 6))
        for i in range(n_trial):
            start = time.time()
            # Dimensionality reduction
            X, y = DR(X, y, method, n_components=n_components)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)
            accuracy = balanced_accuracy_score(y_test, y_pred).round(2)
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted').round(2)
            pre = precision_score(y_test, y_pred, average='weighted').round(2)
            rec = recall_score(y_test, y_pred, average='weighted').round(2)
            f1 = f1_score(y_test, y_pred, average='weighted').round(2)
            endtime = time.time() - start
            res[i, :] = [pre, rec, f1, accuracy, auc, endtime]
        results.append([name, res.mean(axis=0).round(2)])
    return results

# Option 4: Perform classification after applying both preprocessing and dimensionality reduction.
def option4(X, y, method, n_components=n_components):
    # Use preprocessing (MinMaxScaler or some other techniques.) 
    X = min_max_scaler.fit_transform(X)
    results = []
    for name, clf in zip(names, classifiers):
        res = np.zeros((n_trial, 6))
        for i in range(n_trial):
            start = time.time()
            # Dimensionality reduction
            X, y = DR(X, y, method, n_components=n_components)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
            # Apply Synthetic Minority Over-sampling Technique on the training data
            X_train, y_train = smote.fit_resample(X_train, y_train)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)
            accuracy = balanced_accuracy_score(y_test, y_pred).round(2)
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted').round(2)
            pre = precision_score(y_test, y_pred, average='weighted').round(2)
            rec = recall_score(y_test, y_pred, average='weighted').round(2)
            f1 = f1_score(y_test, y_pred, average='weighted').round(2)
            endtime = time.time() - start
            res[i, :] = [pre, rec, f1, accuracy, auc, endtime]
        results.append([name, res.mean(axis=0).round(2)])
    return results

