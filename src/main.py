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
import classifier
from evolution import Evolution
import numpy as np
import sys, os
sys.path.append("..")
# ---------------------------------------------------------------------------- #

if __name__ == '__main__':
    
    # import data
    X, y = utils.load_data(
        filename='data_101718.csv', 
        clean=False,
        normalize=True,
        resample=2 # (2) to downsample the negative cases
    )  

    # concatenate selected features with their target values
    dataset = np.column_stack((X[:, :50], y))
    
    evo = Evolution(
        dataset = dataset.tolist(), # data samples 
        popsize = 100,              # initial population size
        hofsize = 1,                # the number of best individual to track
        cx = .5,                    # crossover rate
        mut = .2,                   # mutation rate
        maxgen = 50,                # max number of generations
    )  

    pop, stats, hof = evo.run()

    # get features of the best individual
    tree = evo.get_tree(hof[0])
    features_idx = [int(f[2:]) for f in tree.values() if str(f).startswith('f_')] 
    
    # filter out unselected features
    selected_features = X[:, features_idx] 

    if len(selected_features):
        print('Number of selected features: ', selected_features.shape[1])

        # evaluate selected features
        optimizer = classifier.find_best_config(X, y, clf='rf', folds=3)
        
        auc_score = classifier.eval(
            selected_features, y, 
            clf = 'rf',  
            params = optimizer.best_params_, 
            folds=10, # number of runs
            plot_roc=True  
        )

    else:
        print('No features were selected!!!')

    
