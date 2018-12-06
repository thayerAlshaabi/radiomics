# coding: utf-8

""" MIT License """
'''
    Axel Masquelin & Sami Connolly  
    Andrea Elhajj  & Thayer Alshaabi
    ---
    Description:
    Generates violin plots of AUC data stored in csv files, and runs a two way t-test to
    evaluate whether two approaches are statistically different from each other. 
    ---
    ---
    Copyright (c) 2018 
'''

# libraries and dependencies
# ---------------------------------------------------------------------------- #
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import seaborn as sns
import pandas as pd
from scipy import stats
import numpy as np
import glob
import csv
import os
import re
# ---------------------------------------------------------------------------- #\

cwd = os.getcwd()
base = []
for root, dirs, files in os.walk(cwd + "/results/images/TrimmedDataset", topdown = True):
    df = pd.DataFrame()
    plt.figure(1)
    for name in files:
        if (name.endswith(".csv")):
            if (len(name.split('_')) == 2):
                header = name.split('_')[0]
                mean_AUC = []
                filename = os.path.join(root,name)
                base.append(os.path.splitext(os.path.basename(filename))[0])
                #aucs = csv.read_csv(filename)
                with open(filename, newline = '') as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        for l in range(len(row)-1):
                            mean_AUC.append(float(row[l+1]))
                df[header] = mean_AUC

labels = {'gp-rf', 'gp-svm', 'gp', 'rf', 'svm'}
sns.violinplot(data = df, inner="quartile", bw=.15, fontsize = 12)
plt.xlabel('Methods', fontsize = 12)
plt.ylabel('AUC', fontsize = 12)

t,p = stats.ttest_ind(df['gp-rf'], df['gp-svm'], equal_var = False)
print("gp-rf, gp-svm: ", t,p)
t,p = stats.ttest_ind(df['gp-rf'], df['gp'], equal_var = False)
print("gp-rf, gp: ", t,p)
t,p = stats.ttest_ind(df['gp'], df['gp-svm'], equal_var = False)
print("gp, gp-svm: ", t,p)
t,p = stats.ttest_ind(df['gp'], df['svm'], equal_var = False)
print("gp, svm: ", t,p)
t,p = stats.ttest_ind(df['gp'], df['rf'], equal_var = False)
print("gp, rf: ", t,p)
t,p = stats.ttest_ind(df['gp-rf'], df['rf'], equal_var = False)
print("gp-rf, rf: ", t,p)
t,p = stats.ttest_ind(df['gp-svm'], df['svm'], equal_var = False)
print("gp-svm, svm: ", t,p)
#plt.show()

