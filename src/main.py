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
import parsers
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
        hofsize = 25,                 # the number of best individual to track
        cx = crRate,                  # crossover rate
        mut = mutRate,                # mutation rate
        maxgen = GenMax,              # max number of generations
    )

    pop, logbook, hof = evo.run(verbose=1)

    return evo, pop, logbook, hof


if __name__ == '__main__':
    seed = 2018
    folds = 5 
    hofp_size = 10
<<<<<<< HEAD
    method = 'svm'
=======
    method = 'gp-svm'
>>>>>>> 25355d0a6378b4ae1a137c707da46cec6a8cf8dd
    reps = 10
    
    # import data
    X, y = utils.load_data(
        filename='data_trimmed.csv', 
        clean=False,
        normalize=True,
        resample=2 # (2) to downsample the negative cases
    )  

    cond = method.split('-')
    print('\n\nMethod: ', cond)

    auc_scores = np.zeros((reps, folds))

    for r in range(reps):
        print('\nRun #', r+1)
        print('-'*75)

        # set a fix seed
        np.random.seed(seed)

        # setup cross-validation 
        cv = StratifiedKFold(
            n_splits=folds, 
            shuffle=True, 
            random_state=seed
        ).split(X, y)

        fprs, tprs = [], []
        # run with specified cross-validation folds
        fig_counter = 2
        for itr, (train, test) in enumerate(cv):
            print('--Running fold ', itr+1)
            
            if cond[0] == 'gp':
                evo, pop, logbook, hof = run_deap(X[train], y[train])

                test_scores = np.zeros(len(hof))
                # evaluate trees in the current hof array on the testing set
                for i in range(len(hof)):
                    predictions = parsers.get_tree_predictions(
                        gp.compile(hof[i], evo.pset), 
                        np.column_stack((X[test], y[test]))
                    )
                    test_scores[i] = parsers.eval_tree(y[test], predictions)

                # get top trees based on their scores on the testing set
                hof_prime = [hof[i] for i in test_scores.argsort()[-hofp_size:][::-1]]

                # parse features from the selected trees
                features_idx, fig_counter = parsers.parse_features(hof_prime, fig_counter)

                print('\nNumber of selected features: {}\n\n'.format(
                    len(features_idx))
                )
                
                if len(cond) > 1:
                    if ((cond[1] == 'rf') or (cond[1] == 'svm')):
                        print(cond[1])
                        fp, tp = classifier.eval(
                            X[train[:, None], features_idx], 
                            X[test[:, None],  features_idx],
                            y[train], y[test],  
                            clf=cond[1], seed=seed
                        )
<<<<<<< HEAD
                else:     
                    fp, tp  = parsers.eval_hof(
=======
                else:
                    print(cond[0])     
                    fp, tp  = parser.eval_hof(
>>>>>>> 25355d0a6378b4ae1a137c707da46cec6a8cf8dd
                        [gp.compile(i, evo.pset) for i in hof],
                        X[test], y[test] 
                    )


            elif (cond[0] == 'rf') or (cond[0] == 'svm'):
                fp, tp = classifier.eval(
                    X[train], X[test],
                    y[train], y[test],  
                    clf=cond[0], seed=seed
                )

            tprs.append(tp)
            fprs.append(fp)

        auc_scores[r, :] = utils.calc_auc(fprs, tprs, plot_roc=True)
    
        print('-'*75)
    
    print('Done')
    utils.csv_save(method, auc_scores)
    plt.show() # show figures

