# coding: utf-8

""" MIT License """
'''
    Axel Masquelin & Sami Connolly  
    Andrea Elhajj  & Thayer Alshaabi
    ---
    Description: 
    Utilized to generate plots of the max fitness of individuals based on the optimizer.py function
    ---
    Copyright (c) 2018 
'''

# libraries and dependencies
# ---------------------------------------------------------------------------- #
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os
import re
# ---------------------------------------------------------------------------- #

cwd = os.getcwd()
base = []
for root, dirs, files in os.walk(cwd + "/results/", topdown = True):
    plt.figure(1)
    for name in files:
        if ((re.fullmatch('mutUniform', re.split(r'\W+', name)[0])) and name.endswith(".csv")):
            filename = os.path.join(root,name)
            base.append(os.path.splitext(os.path.basename(filename))[0])
            df = pd.read_csv(filename)
            plt.plot(df['gen'], df['Avg Max'])



plt.xlabel('Generations')
plt.ylabel('Average Fitness')
base = np.asarray(base)        
plt.title("Average Maximum Fitness over Generations")
plt.legend(base)
#plt.legend(('Pop 100', 'Pop 250', 'Pop 500', 'Pop 1000'))
plt.show()
        

