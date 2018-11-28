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
import parser
import operator
import classifier
import numpy as np
from deap import gp
import matplotlib.pyplot as plt
from evolution import Evolution
from sklearn.model_selection import StratifiedKFold
# ---------------------------------------------------------------------------- #

def run_deap(X, y):
    popsize = 500
    mutRate = 0.3 #If bloating control is removed use 0.3
    crRate = 0.5 #If bloating control removed use 0.5 (.7)
    GenMax = 250
    
    # concatenate selected features with their target values
    dataset = np.column_stack((X, y))

    evo = Evolution(
        dataset = dataset.tolist(),   # data samples 
        popsize = popsize,            # initial population size
        hofsize = 10,                 # the number of best individual to track
        cx = crRate,                  # crossover rate
        mut = mutRate,                # mutation rate
        maxgen = GenMax,              # max number of generations
    )

    pop, logbook, hof = evo.run(verbose=1)

    return evo, pop, logbook, hof


if __name__ == '__main__':
    seed = 2018
    folds = 5  

    # set a fix seed
    np.random.seed(seed)

    # import data
    X, y = utils.load_data(
        filename='data_trimmed.csv', 
        clean=False,
        normalize=True,
        resample=2 # (2) to downsample the negative cases
    )  

    # setup cross-validation 
    cv = StratifiedKFold(
        n_splits=folds, 
        shuffle=True, 
        random_state=seed
    ).split(X, y)

    fprs, tprs = [], []
    # run with specified cross-validation folds
    for itr, (train, test) in enumerate(cv):
        print('--Running fold ', itr+1)
        #utils.progressBar(itr+1, folds) 

        evo, pop, logbook, hof = run_deap(X[train], y[train])

        test_scores = np.zeros(len(hof))

        # evaluate trees in the current hof array on the testing set
        for i in range(len(hof)):
            tree = gp.compile(hof[i], evo.pset)
            dataset = np.column_stack((X[test], y[test]))
            test_scores[i] = parser.eval_tree(tree, dataset)
        
        # get top 5 trees based on their scores on the testing set
        hof_prime = [hof[i] for i in test_scores.argsort()[-5:][::-1]]

        # parse features from the selected trees
        features_idx = parser.parse_features(hof_prime)

        print('\nNumber of selected features: {}\n\n'.format(
            len(features_idx))
        )

        fp, tp = classifier.eval(
            X[train[:, None], features_idx], 
            X[test[:, None],  features_idx],
            y[train], y[test],  
            clf = 'rf', seed=seed
        )

        tprs.append(tp)
        fprs.append(fp)

    auc_scores = utils.calc_auc(fprs, tprs, plot_roc=True)
    print('kfolds Cross-Validation AUC: ', auc_scores)

    print('Done')
    plt.show() # show figures

