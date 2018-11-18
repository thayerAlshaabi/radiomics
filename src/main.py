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
    mode = 'Optimize'
    # import data
    X, y = utils.load_data(
        filename='data_trimmed.csv', 
        clean=False,
        normalize=True,
        resample=2 # (2) to downsample the negative cases
    )  

    # concatenate selected features with their target values
    dataset = np.column_stack((X, y))
    
    if mode == 'Optimize':
        pop = [10, 50, 100, 500, 1000]
        mutRate = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        crRate = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        GenMax = [50, 100, 250, 500]
    else:
        pop = 100
        mutRate = 0.2
        crRate = 0.5
        GenMax = 50

    for i in range(len(pop)):
        evo = Evolution(
            dataset = dataset.tolist(), # data samples 
            popsize = pop[i],              # initial population size
            hofsize = 10,                # the number of best individual to track
            cx = crRate[i],                    # crossover rate
            mut = mutRate[i],                   # mutation rate
            maxgen = GenMax[i],                # max number of generations
        )  

        pop, stats, hof = evo.run()
        print("Statistics:", '\n', stats)
    #     plt.figure(i)
    #     # plt.plot(stats.gen, stats.avg)
    #     plt.xlabel("Generations")
    #     plt.ylabel("Fitness")
    #     plt.title("Fitness to Population Size")
    # plt.show()

    # get features of the best individual
    tree = evo.get_tree(hof[0], plot=True)
    features_idx = [int(f[2:]) for f in tree.values() if str(f).startswith('f_')] 
    
    if mode == 'Full':
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

    
