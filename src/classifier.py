# coding: utf-8

""" MIT License """
'''
    Axel Masquelin & Sami Connolly  
    Andrea Elhajj  & Thayer Alshaabi
    ---
    Copyright (c) 2018 
'''

# libraries and dependencies
# ---------------------------------------------------------------------------- #
import utils
import itertools
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# ---------------------------------------------------------------------------- #


def find_best_config(X, y, clf, folds):
    '''
        Run an exhaustive search over the 
        specified parameter values to find the best configuration 
        for either 
        -   rf:  RandomForestClassifier or 
        -   svm: SVMClassifier
    '''
    if clf == 'rf':
        classifier = RandomForestClassifier(random_state=2018)

        param_grid = { 
            'n_estimators': [100, 300, 500],
            'max_features': ['auto', 'sqrt'],
            'criterion'   : ['gini', 'entropy'],
            'class_weight': ['balanced', 'balanced_subsample'],
        }
    elif clf == 'svm':
        classifier = SVC(random_state=2018)

        c = [0.1, 1, 10, 50, 75, 100]
        param_grid = [
            {
                'kernel':['linear'], 
                'class_weight': ['balanced'], 
                'C':c
            },
            {
                'kernel':['poly'],   
                'class_weight': ['balanced'], 
                'C':c, 
                'degree':[2,3,4,5]
            },
            {
                'kernel':['rbf'],    
                'class_weight': ['balanced'], 
                'C':c, 
                'gamma':[1e-1, 1e-2, 1e-3]
            },
            {
                'kernel':['sigmoid'],
                'class_weight': ['balanced'], 
                'C':c
            }
        ]
    else:
        print('Error: unknown classifier')

    # Find the best parameters with GridSearchCV 
    # for (parallel 5 folds cross-validation)
    optimizer = GridSearchCV(
        classifier, param_grid=param_grid, cv=folds, 
        scoring='roc_auc', verbose=10, n_jobs=-1)

    optimizer.fit(X, y)

    return optimizer


def eval(
    X,              # data matrix
    y,              # target vvector
    clf,            # classifier to use ('rf', 'svm')
    params,         # configuration(parameters) to initialize the classifier
    folds=10,       # number of folds for cross-validation
    plot_roc=False  
):
    '''
        Run classifier with specified cross-validation folds(cv) 
        and claculate the accuracy of the model based 
        on True-Positive and False-Positive predictions

        clf -- 
        -   'rf':  RandomForestClassifier or 
        -   'svm': SVMClassifier

        params -- 
        -   'auto': run an exhaustive search to find the best configuration
        -   dict of classifier configuration 
    '''
    cv = StratifiedKFold(
        n_splits=folds, shuffle=True, 
        random_state=2018
    ).split(X, y)

    if clf == 'rf':
            classifier = RandomForestClassifier(**params)

    elif clf == 'svm':
            classifier = SVC(**params, probability=True)

    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    
    if plot_roc:
        plt.figure(2, figsize=(10,8))

    for itr, (train, test) in enumerate(cv):
        utils.progressBar(itr+1, folds) 
        
        # get predictions
        probs = classifier.fit(X[train], y[train]).predict_proba(X[test])
        
        # compute AUC
        fpr, tpr, _ = roc_curve(y[test], probs[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        if plot_roc:
            plt.plot(
                fpr, tpr, lw=1, alpha=0.5,
                label='ROC fold %d (AUC = %0.2f)' % (itr+1, roc_auc)
            )
  
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    if plot_roc:
        # plot mean score of all runs
        plt.plot(
            mean_fpr, mean_tpr, color='k',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc)
        )
        # plot std
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(
            mean_fpr, tprs_lower, tprs_upper,
            color='grey', alpha=.2, label=r'$\pm$ std.'
        )

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', fontsize=18)
        plt.ylabel('True Positive Rate', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title('ROC Curve', fontsize=20)
        plt.legend(loc="lower right", fontsize=14)
        plt.show()

    return mean_auc


def plot_confmtx(X,y, clf_params, folds=10):
    '''
        Run a RandomForestClassifier with cross-validation 
        and plot a confusion matrix for each fold
    '''
    cv = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=2018
    ).split(X, y)

    classifier = RandomForestClassifier(**clf_params)
    classes = np.unique(y)

    fig = plt.figure(figsize=(10,8))
    for itr, (train, test) in enumerate(cv):
        utils.progressBar(itr+1, folds) 
            
        # get predictions
        y_pred = classifier.fit(X[train], y[train]).predict(X[test])
        
        # compute confusion matrix
        matrix = confusion_matrix(y[test], y_pred)
        
        # plot confusion matrix
        ax = fig.add_subplot(np.ceil(folds/2),np.ceil(folds/2),itr+1)
        ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.GnBu)
        
        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            plt.text(
                j, i, format(matrix[i, j], 'd'),
                horizontalalignment="center",
                fontsize=14, color="white" if matrix[i, j] > matrix.max()/2 else "black"
            )
        
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, fontsize=14)
        plt.yticks(tick_marks, classes, fontsize=14)
        plt.ylabel('True label', fontsize=18)
        plt.xlabel('Predicted label', fontsize=18)
        plt.title('Fold %d' % (itr+1), fontsize=20)

    plt.tight_layout()
    plt.show()