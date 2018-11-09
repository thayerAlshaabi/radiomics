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
import matplotlib.pyplot as plt
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

    
