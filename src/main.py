# coding: utf-8

# libraries and dependencies
# ---------------------------------------------------------------------------- #
import utils
import classifier
import sys, os
sys.path.append("..")
# ---------------------------------------------------------------------------- #

if __name__ == '__main__':
    
    df = utils.subsample_df(utils.import_data()) 

    X, y = utils.load_data(df, normalize=True)

    optimizer = classifier.find_best_config(X, y, clf='rf', folds=5)
    
    auc_score = classifier.fitness(
        X, y, 
        clf = 'rf',  
        params = optimizer.best_params_, 
        folds=10, plot_roc=True  
    )
    
    classifier.plot_confusion_matrix(
        X, y, clf_params=optimizer.best_params_, folds=3
    )
