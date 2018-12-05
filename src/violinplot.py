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
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import seaborn as sns
import pandas as pd
import numpy as np
import glob
import os
import re
# ---------------------------------------------------------------------------- #\

cwd = os.getcwd()
base = []
for root, dirs, files in os.walk(cwd + "/results/images/OriginalDataset", topdown = True):
    df = pd.DataFrame()
    plt.figure(1)
    for name in files:
        if (name.endswith(".csv")):
            if (len(name.split('_')) == 2):
                header = name.split('_')[0]
                mean_AUC = []
                filename = os.path.join(root,name)
                base.append(os.path.splitext(os.path.basename(filename))[0])
                aucs = pd.read_csv(filename)
                mean_AUC.append(np.mean(aucs['AUC1']))
                mean_AUC.append(np.mean(aucs['AUC2']))
                mean_AUC.append(np.mean(aucs['AUC3']))
                mean_AUC.append(np.mean(aucs['AUC4']))
                mean_AUC.append(np.mean(aucs['AUC5']))
                df[header] = mean_AUC
print(df)


labels = {'gp-rf', 'gp-svm', 'gp', 'rf', 'svm'}

#ax.Axes.violinplot(mean_AUC, showmeans = True, showmedians = True)
ax = sns.violinplot(x = labels, y = 'AUC', hue = "smoker", palette = "mutted", data = df)

# plt.xlabel('Generations')
# plt.ylabel('Average Fitness')
# base = np.asarray(base)        
# plt.title("Average Maximum Fitness over Generations")
# plt.legend(base)
# #plt.legend(('Pop 100', 'Pop 250', 'Pop 500', 'Pop 1000'))
# plt.show()