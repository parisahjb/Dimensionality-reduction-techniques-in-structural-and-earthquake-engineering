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
# from imblearn.metrics import classification_report_imbalanced,sensitivity_specificity_support,sensitivity_score,specificity_score,geometric_mean_score,macro_averaged_mean_absolute_error,make_index_balanced_accuracy 
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

# dimensionality reduction
def DR(X,y,method,n_components=2):
    if method=='TSNE':
        X = TSNE(n_components=n_components, method="exact", perplexity=100).fit_transform(X)
    if method=='PCA':
        X = PCA(n_components=n_components).fit_transform(X)
    if method=='LDA':
        X = LinearDiscriminantAnalysis(n_components=n_components).fit(X, y).transform(X)
    return X,y
#Different Classifiers

names = ["KNN","SVM", "DT","RF","MLP","AdaBoost"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(gamma=2, C=1, probability=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
]

min_max_scaler = MinMaxScaler()
smote = SMOTE(random_state = 0)
# dimensionality reduction
n_components=2
def DR(X,y,method,n_components=n_components):
    if method=='TSNE':
        X = TSNE(n_components=n_components, perplexity=100).fit_transform(X) #method='exact',
    if method=='PCA':
        X = PCA(n_components=n_components).fit_transform(X)
    if method=='LDA':
        X = LinearDiscriminantAnalysis(n_components=n_components).fit(X, y).transform(X)
    return X,y
def option1(X,y):

# use preprocessing (MinMaxScaler or some other techniques.) 
    X = min_max_scaler.fit_transform(X) 
    # split data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    results=[]
    for name, clf in zip(names, classifiers):
        res=np.zeros((n_trial,6))
        for i in range(n_trial):
            start = time.time()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
            clf.fit(X_train, y_train)
            y_pred=clf.predict(X_test)
            y_prob=clf.predict_proba(X_test)
            accuracy = (balanced_accuracy_score(y_test, y_pred)).round(2)
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted').round(2)
            pre=precision_score(y_test, y_pred, average='weighted').round(2)
            rec=recall_score(y_test, y_pred, average='weighted').round(2)
            f1=f1_score(y_test, y_pred, average='weighted').round(2)
            endtime=time.time() - start
            res[i,:]=[pre,rec,f1,accuracy,auc,endtime]          
        results.append([name,res.mean(axis=0).round(2),])
    return results

def option2(X,y):
# use preprocessing (MinMaxScaler or some other techniques.)
    X = min_max_scaler.fit_transform(X) 
    results=[]
    for name, clf in zip(names, classifiers):
        res=np.zeros((n_trial,6))
        for i in range(n_trial):
            start = time.time()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
            #Synthetic Minority Over-sampling Technique on train data
            X_train, y_train = smote.fit_resample(X_train, y_train)
            clf.fit(X_train, y_train)
            y_pred=clf.predict(X_test)
            y_prob=clf.predict_proba(X_test)
            accuracy = (balanced_accuracy_score(y_test, y_pred)).round(2)
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted').round(2)
            pre=precision_score(y_test, y_pred, average='weighted').round(2)
            rec=recall_score(y_test, y_pred, average='weighted').round(2)
            f1=f1_score(y_test, y_pred, average='weighted').round(2)
            endtime=time.time() - start
            res[i,:]=[pre,rec,f1,accuracy,auc,endtime]          
        results.append([name,res.mean(axis=0).round(2),])
    return results
 
def option3(X,y,method,n_components=n_components):
    # use preprocessing (MinMaxScaler or some other techniques.) 
    X = min_max_scaler.fit_transform(X)
    results=[]
    for name, clf in zip(names, classifiers):
        res=np.zeros((n_trial,6))
        for i in range(n_trial):
            start = time.time()
            # dimensionality reduction
            X,y=DR(X,y,method,n_components=n_components)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
            clf.fit(X_train, y_train)
            y_pred=clf.predict(X_test)
            y_prob=clf.predict_proba(X_test)
            accuracy = (balanced_accuracy_score(y_test, y_pred)).round(2)
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted').round(2)
            pre=precision_score(y_test, y_pred, average='weighted').round(2)
            rec=recall_score(y_test, y_pred, average='weighted').round(2)
            f1=f1_score(y_test, y_pred, average='weighted').round(2)
            endtime=time.time() - start
            res[i,:]=[pre,rec,f1,accuracy,auc,endtime]          
        results.append([name,res.mean(axis=0).round(2),])
    return results    
def option4(X,y,method,n_components=n_components):
    # use preprocessing (MinMaxScaler or some other techniques.) 
    X = min_max_scaler.fit_transform(X) #X_normalized
    results=[]
    for name, clf in zip(names, classifiers):
        res=np.zeros((n_trial,6))
        for i in range(n_trial):
            start = time.time()
            # dimensionality reduction
            X,y=DR(X,y,method,n_components=n_components)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
            #Synthetic Minority Over-sampling Technique on train data
            X_train, y_train = smote.fit_resample(X_train, y_train)
            clf.fit(X_train, y_train)
            y_pred=clf.predict(X_test)
            y_prob=clf.predict_proba(X_test)
            accuracy = (balanced_accuracy_score(y_test, y_pred)).round(2)
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted').round(2)
            pre=precision_score(y_test, y_pred, average='weighted').round(2)
            rec=recall_score(y_test, y_pred, average='weighted').round(2)
            f1=f1_score(y_test, y_pred, average='weighted').round(2)
            endtime=time.time() - start
            res[i,:]=[pre,rec,f1,accuracy,auc,endtime]          
        results.append([name,res.mean(axis=0).round(2),])
    return results    

