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
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix
# ---------------------------------------------------------------------------- #


def find_best_config(X, y, clf, folds):
    '''
        Run an exhaustive search over the 
        specified parameter values to find the best configuration 
    '''
    if isinstance(clf, RandomForestClassifier):
        param_grid = { 
            'n_estimators': [100, 300, 500],
            'max_features': ['auto', 'sqrt'],
            'criterion'   : ['gini', 'entropy'],
            'class_weight': ['balanced', 'balanced_subsample'],
        }
    elif isinstance(clf, SVC):
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
    optimizer = RandomizedSearchCV(
        clf, param_grid, cv=folds, 
        scoring='roc_auc', verbose=1, n_jobs=-1).fit(X, y)

    return optimizer


def plot_confmtx(matrix, classes, itr):
    '''
        Plot a confusion matrix for each fold
    '''        
    # plot confusion matrix
    plt.figure(itr+3, figsize=(10,8))
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.GnBu)
    
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(
            j, i, format(matrix[i, j], 'd'),
            horizontalalignment="center",
            fontsize=14, color="white" if matrix[i, j] > matrix.max()/2 else "black"
        )
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=14)
    plt.yticks(tick_marks, classes, fontsize=14)
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.title('Fold %d' % (itr+1), fontsize=20)
    plt.tight_layout()


def plot_roc_curve(tprs, mean_fpr, mean_tpr, mean_auc, std_auc):
    '''
        Plot roc curve per fold and mean/std score of all runs
    '''
    plt.figure(2, figsize=(10,8))

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


def eval(
    X,              # data matrix
    y,              # target vvector
    clf,            # classifier to use ('rf', 'svm')
    folds=10,       # number of folds for cross-validation
    plot_roc=False,
    plot_confusion_matrix=False,  
):
    '''
        Run classifier with specified cross-validation folds(cv) 
        and claculate the accuracy of the model based 
        on True-Positive and False-Positive predictions

        clf -- 
        -   'rf':  RandomForestClassifier or 
        -   'svm': SVMClassifier
    '''
    cv = StratifiedKFold(
        n_splits=folds, shuffle=True, 
        random_state=2018
    ).split(X, y)

    if clf == 'rf':
            clf = RandomForestClassifier(random_state=2018)

    elif clf == 'svm':
            clf = SVC(random_state=2018, probability=True)

    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    
    for itr, (train, test) in enumerate(cv):
        utils.progressBar(itr+1, folds) 
        
        # get predictions
        classifier = find_best_config(X[train], y[train], clf, folds//2)
        probs = classifier.predict_proba(X[test])
        
        if plot_confusion_matrix:
            classes = np.unique(y)

            # get predictions
            y_pred = classifier.predict(X)

            # compute confusion matrix
            matrix = confusion_matrix(y, y_pred)

            plot_confmtx(matrix, classes, itr)

        # compute AUC
        fpr, tpr, _ = roc_curve(y[test], probs[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        if plot_roc:
            plt.figure(2, figsize=(10,8))
            plt.plot(
                fpr, tpr, lw=1, alpha=0.5,
                label='ROC fold %d (AUC = %0.2f)' % (itr+1, roc_auc)
            )
  
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    if plot_roc:
        plot_roc_curve(tpr, mean_fpr, mean_tpr, mean_auc, std_auc)
        
    return mean_auc