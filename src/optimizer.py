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
        filename='data_trimmed.csv', 
        clean=False,
        normalize=True,
        resample=2 # (2) to downsample the negative cases
    )  

    # concatenate selected features with their target values
    dataset = np.column_stack((X, y))
    
    popsize = [100, 500, 1000]
    mutRate = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    crRate = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    GenMax = [50, 100, 250, 500]
    reps = 5
    states = 1

    for i in range(len(popsize)):
        evo = Evolution(
            dataset = dataset.tolist(),      # data samples 
            popsize = popsize[0],            # initial population size
            hofsize = 10,                    # the number of best individual to track
            cx = crRate[0],                  # crossover rate
            mut = mutRate[0],                # mutation rate
            maxgen = GenMax[0],              # max number of generations
            )
            
        MaxLogs = np.zeros((reps, GenMax[0]+1))
            
        for l in range(reps):
            pop, logbook, hof= evo.run(reps)    
            MaxLogs[l][:] = logbook.select("max")

        MaxAvg = []
        for n in range(GenMax[0]+1):
            total = 0
            for m in range(reps):
                total += MaxLogs[m][n]
            MaxAvg.append(total/reps) 

    print('Done')


    
