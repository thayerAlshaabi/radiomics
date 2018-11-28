# coding: utf-8

""" MIT License """
'''
    Axel Masquelin & Sami Connolly  
    Andrea Elhajj  & Thayer Alshaabi
    ---
    Copyright (c) 10 
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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve
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
        scoring='roc_auc', verbose=0, n_jobs=-1).fit(X, y)

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


def eval(
    X_train, X_test,
    y_train, y_test,              
    clf, # classifier to use ('rf', 'svm')
    seed=10, # seed for the random generator
):
    '''
        Run classifier and claculate the accuracy of the model based 
        on True-Positive and False-Positive predictions

        clf -- 
        -   'rf':  RandomForestClassifier or 
        -   'svm': SVMClassifier
    '''
    if clf == 'rf':
            clf = RandomForestClassifier(random_state=seed)

    elif clf == 'svm':
            clf = SVC(random_state=seed, probability=True)
    
    # get predictions
    classifier = find_best_config(X_train, y_train, clf, 5)
    probs = classifier.predict_proba(X_test)
    
    # compute AUC
    fpr, tpr, _ = roc_curve(y_test, probs[:, 1])

    return (fpr, tpr)