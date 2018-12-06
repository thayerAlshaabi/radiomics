# coding: utf-8

""" MIT License """
'''
    Axel Masquelin & Sami Connolly  
    Andrea Elhajj  & Thayer Alshaab
    ---
    Description: 
    Function designed to evaluate all parameters provided to the gp and identify the best parameters.
    Saves all fitness of individuals by logging them into csv files which will then be evaluated on plots.py

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
import random
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
    GenMax = [50, 100, 250, 500]
    mutRate = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    crRate = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]# 0.7, 0.8, 0.9]
    reps = 5
    states = 1

    i = 6
    evo = Evolution(
        dataset = dataset.tolist(),      # data samples 
        popsize = popsize[3],            # initial population size
        hofsize = 10,                    # the number of best individual to track
        cx = crRate[i],                  # crossover rate
        mut = mutRate[6],                # mutation rate
        maxgen = GenMax[2],              # max number of generations
        )
            
    logs = pd.DataFrame()

    gen = np.zeros((reps, GenMax[2]+1))
    nevals = np.zeros((reps, GenMax[2]+1))
    avg = np.zeros((reps, GenMax[2]+1))
    mini = np.zeros((reps, GenMax[2]+1))    
    maxi = np.zeros((reps, GenMax[2]+1))
    
    for l in range(reps):
        np.random.seed(reps)

        pop, logbook, hof= evo.run()
        gen[l][:] = logbook.select('gen')
        nevals[l][:] = logbook.select('nevals')
        avg[l][:] = logbook.select('avg')
        mini[l][:] = logbook.select('min')
        maxi[l][:] = logbook.select('max')
        
    AvgEval = []
    Avg = []
    AvgMin = []
    AvgMax = []
    
    for n in range(GenMax[2]+1):
        totalEval = 0
        totalAvg = 0
        totalMin = 0
        totalMax = 0
        
        for m in range(reps):
            totalEval += nevals[m][n]
            totalAvg += avg[m][n]
            totalMin += mini[m][n]
            totalMax += maxi[m][n]
        
        AvgEval.append(totalEval/reps)
        Avg.append(totalAvg/reps)
        AvgMin.append(totalMin/reps)
        AvgMax.append(totalMax/reps)

    logs['gen'] = gen[l][:]
    logs['nEval'] = AvgEval
    logs['Avg Fitness'] = Avg
    logs['Avg Min'] = AvgMin
    logs['Avg Max'] = AvgMax
            
    #print(logs)

    cwd = os.getcwd()
    pth_to_save =  cwd + "/results/mutEphemeralAll.6_cxOnePoint_.6_selDoubleTournament_codeBloatOn.csv"
    logs.to_csv(pth_to_save)

    print('Done')