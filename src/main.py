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
import numpy as np
import classifier
from evolution import Evolution
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------------- #

if __name__ == '__main__':

    # set a fix seed
    np.random.seed(2018)

    # import data
    X, y = utils.load_data(
        filename='data_trimmed.csv', 
        clean=False,
        normalize=True,
        resample=2 # (2) to downsample the negative cases
    )  

    # concatenate selected features with their target values
    dataset = np.column_stack((X, y))
    
    popsize = 500
    mutRate = 0.3 #If bloating control is removed use 0.3
    crRate = 0.5 #If bloating control removed use 0.5 (.7)
    GenMax = 2

    evo = Evolution(
        dataset = dataset.tolist(),   # data samples 
        popsize = popsize,            # initial population size
        hofsize = 10,                 # the number of best individual to track
        cx = crRate,                  # crossover rate
        mut = mutRate,                # mutation rate
        maxgen = GenMax,              # max number of generations
    )

    pop, logbook, hof = evo.run(verbose=1)

    # get features of the best individual
    tree = evo.get_tree(hof[0], plot=True)
    features_idx = [int(f[2:]) for f in tree.values() if str(f).startswith('f_')] 

    # filter out unselected features
    selected_features = X[:, features_idx] 
    
    if len(selected_features):
        print('\nNumber of selected features: {}\n\n'.format(
            selected_features.shape[1])
        )

        # evaluate selected features
        print('Evaluating features...')
        auc_score = classifier.eval(
            selected_features, y, 
            clf = 'rf',  
            folds=10, 
            plot_roc=True,
            plot_confusion_matrix=True,   
        )

        # show figures
        print('Done')
        plt.show()

    else:
        print('No features were selected!!!')
