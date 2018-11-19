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
    
    # if mode == 'Optimize':
    #     popsize = [10, 50, 100, 500, 1000]
    #     mutRate = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #     crRate = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #     GenMax = [50, 100, 250, 500]
    # else:
    popsize = 500
    mutRate = 0.2
    crRate = 0.3
    GenMax = 250
    reps = 5

    evo = Evolution(
        dataset = dataset.tolist(), # data samples 
        popsize = popsize,              # initial population size
        hofsize = 10,                # the number of best individual to track
        cx = crRate,                    # crossover rate
        mut = mutRate,                   # mutation rate
        maxgen = GenMax,                # max number of generations
        )  
    
    if mode == 'Optimize':
        MaxLogs = np.zeros((reps, GenMax+1))

        for i in range (reps):
            pop, logbook, hof= evo.run(reps)
            MaxLogs[i][:] = logbook.select("max")
    
        MaxAvg = []
        for j in range(GenMax+1):
            total = 0
            for i in range(reps):
                total += MaxLogs[i][j]
            MaxAvg.append(total/reps)
        
        plt.figure(1)
        plt.plot(logbook.select("gen"), MaxAvg)
        plt.xlabel("Generations")
        plt.ylabel("Maximum Average Fitness")
        plt.title("Fitness to Population Size " + str(popsize) + " Nreps = " + str(reps))
        plt.show()
    
    else: 
        pop, logbook, hof= evo.run(reps)

    if mode == 'Full':
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

    
