# coding: utf-8

""" MIT License """
'''
    Axel Masquelin & Sami Connolly  
    Andrea Elhajj  & Thayer Alshaabi
    ---
    Description: 
    main.py is the brain of the script, where all methodologies are defined for the run instance.
    Furthermore, main.py loads the desired dataset defined by "filename" for the user to be able
    to pull whichever dataset he/she desires.
    
    Deap Documentation: https://deap.readthedocs.io/en/master/

    Import Variable:
     (1) Method = ['gp', 'gp-svm', 'gp-rf', 'svm', 'rf'] | # Defines list of methodologies the system will take 
     (2) seed = 2018 | #Defines the see for the classifier function and not the seed for the evolution
     (3) folds = 5  | # Number of cross validation conducted
     (4) hofp_size = 10 | Number of individuals taken fromt he hall of fame population to evaluate
     (5)reps = 5 | # Number of repetition the code will run for.
     (6) filename = string | #Defines the dataset that you will want to utilize

    Function call:
        python main.py
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
    mutRate = 0.4 
    crRate = 0.6 
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
    method = ['gp-svm', 'gp-rf', 'gp', 'svm', 'rf']
    figure = 0
    reps = 5
    
    # import data
    x, y = utils.load_data(
        filename='data_trimmed.csv', 
        clean=False,
        normalize=True,
        resample=2 # (2) to downsample the negative cases
    )
    for n in range(len(method)):

        cond = method[n].split('-')
        print('\n\nMethod: ', cond)
        X = x[:,0:250]
        auc_scores = np.zeros((reps, folds))
        feat = []

        for r in range(reps):
            print('\nRun #', r+1)
            print('-'*75)

            # set a fix seed
            np.random.seed(r)

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
                    feat.append(features_idx)
                    print(feat)

                    if len(cond) > 1:
                        if ((cond[1] == 'rf') or (cond[1] == 'svm')):
                            fp, tp = classifier.eval(
                                X[train[:, None], features_idx], 
                                X[test[:, None],  features_idx],
                                y[train], y[test],  
                                clf=cond[1], seed=seed
                            )
                    else:     
                        fp, tp  = parsers.eval_hof(
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

            auc_scores[r, :] = utils.calc_auc(fprs, tprs, figure, plot_roc=True)
            plt.savefig(r"./results/images/TrimmedDataset/" + method[n] + "-AUC" + "_reps" + str(r))
            figure += 1
            print('-'*75)
        
        utils.csv_save(method[n], auc_scores)
        print(feat)
        
    print('Done')

    # plt.show() # show figures

