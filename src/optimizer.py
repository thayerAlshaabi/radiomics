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
from evolution import Evolution

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import classifier
import utils
import csv
import os

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
    
    popsize = [100, 250, 500, 1000]
    mutRate = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    crRate = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    GenMax = [50, 100, 250, 500]
    reps = 5
    states = 1


    evo = Evolution(
        dataset = dataset.tolist(),      # data samples 
        popsize = popsize[0],            # initial population size
        hofsize = 10,                    # the number of best individual to track
        cx = crRate[0],                  # crossover rate
        mut = mutRate[0],                # mutation rate
        maxgen = GenMax[0],              # max number of generations
        )
            

    logs = {} 
    cwd = os.getcwd()
    pth_to_save =  cwd + "/Results/test.csv"

    for l in range(reps):
        pop, logbook, hof= evo.run(reps)
        logs['reps'] = l
        logs['gen'] = logbook.select('gen')
        logs['nevals'] = logbook.select("nevals")
        logs['avg'] = logbook.select("avg")
        logs['min'] = logbook.select("min")
        logs['max'] = logbook.select("max")
        
        if not os.path.exists(pth_to_save):
            with open(pth_to_save, 'w') as csvfile:
                dict_writer = csv.DictWriter(csvfile, fieldnames=logs.keys())
                dict_writer.writeheader()
                dict_writer.writerow(logs)
        else:
            with open(pth_to_save, 'a') as csvfile:
                dict_writer = csv.DictWriter(csvfile, fieldnames=logs.keys())
                dict_writer.writerow(logs)

    print('Done')


    
