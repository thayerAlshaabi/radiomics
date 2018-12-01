# coding: utf-8

""" MIT License """
'''
    Axel Masquelin & Sami Connolly  
    Andrea Elhajj  & Thayer Alshaabi
    ---
    Copyright (c) 10 
'''

# libraries and dependencies
# ---------------------------------------------------------------------------- #
import sys, os, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
# ---------------------------------------------------------------------------- #


def progressBar(value, endvalue, bar_length=50):
    ''' A simple function to print a progressBar on the screen '''
    percent = float(value) / endvalue
    arrow = '=' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r[{0}] {1}%\t".format(arrow + spaces, int(round(percent * 100))))


def create_df(filename, clean=False):
    '''
        Import dataset into a pandas dataframe 
        and filter out empty data entires
    '''
    df = pd.read_csv(
        os.path.join(os.path.abspath(os.pardir), 
        'src/dataset', filename), low_memory=False
    )
    
    if clean:
        # Drop out the columns that have over 90% missing data points.
        df = df.dropna(thresh=len(df)*.90, axis=1)

        # Drop out any rows that are missing more than 50% of the required columns
        df = df.dropna(thresh=df.shape[1]*.5)

        # Drop out any rows that are missing the target value 
        df = df.dropna(axis=1, subset=df['ca'])

        # Fill the rest of missing entries with the mean of each col
        df = df.apply(lambda col: col.fillna(col.mean()))

    return df


def resample_df(df, method):
    '''
        Equalize the number of samples in each class:
        -- method = 1 - upsample the positive cases
        -- method = 2 - downsample the negative cases
    '''
    df_neg = df[df.ca==0]
    df_pos = df[df.ca==1]


    # Upsample the pos samples
    if method == 1:
        df_pos_upsampled = resample(
            df_pos,
            n_samples=len(df_neg),
            replace=True, 
            random_state=10
        )
        return pd.concat([df_pos_upsampled, df_neg])

    # Downsample the neg samples
    elif method == 2:
        df_neg_downsampled = resample(
            df_neg,
            n_samples=len(df_pos),
            replace=True, 
            random_state=10
        )
        return pd.concat([df_pos, df_neg_downsampled])

    else:
        print('Error: unknown method')
        

def load_data(
    filename, 
    clean=False, 
    normalize=False,
    resample=2,
):
    '''
        Returns:
            X: a matrix of features
            y: a target vector (ground-truth labels)

        normalize   - Option to normalize all features in the dataframe 
        clean       - Option to filter out empty data entires
        resample    - Method to use to resample the dataset
    '''

    df = resample_df(
        create_df(filename, clean),
        method=resample,
    ) 

    features = list(df.columns.values)
    features.remove('pid')
    features.remove('ca')

    X = StandardScaler().fit_transform(df[features])
    y = df.ca.values

    return X,y


def plot_roc_curve(tprs, mean_fpr, mean_tpr, mean_auc, std_auc):
    ''' Plot roc curve per fold and mean/std score of all runs '''

    plt.figure(1, figsize=(10,8))

    plt.plot(
        mean_fpr, mean_tpr, color='k',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc)
    )

    # plot std
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(
        mean_fpr, tprs_lower, tprs_upper,
        color='grey', alpha=.2, label=r'$\pm$ std.'
    )

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('ROC Curve', fontsize=20)
    plt.legend(loc="lower right", fontsize=14)


def calc_auc(fps, tps, plot_roc=False):
    ''' Calculate mean ROC/AUC for a given set of 
        true positives (tps) & false positives (fps) '''

    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    
    for itr, (_fp, _tp) in enumerate(zip(fps, tps)):
        tprs.append(np.interp(mean_fpr, _fp, _tp))
        tprs[-1][0] = 0.0
        roc_auc = auc(_fp, _tp)
        aucs.append(roc_auc)

        if plot_roc:
            plt.figure(1, figsize=(10,8))
            plt.plot(
                _fp, _tp, lw=1, alpha=0.5,
                label='ROC fold %d (AUC = %0.2f)' % (itr+1, roc_auc)
            )

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    if plot_roc:
        plot_roc_curve(tprs, mean_fpr, mean_tpr, mean_auc, std_auc)
        
    return aucs


def csv_save(method, auc):
    ''' Save AUCs scores to a csv file '''

    cols = ['AUC'+str(i+1) for i in range(auc.shape[1])]
    logs = pd.DataFrame(auc, columns=cols)    
    cwd = os.getcwd()
    pth_to_save =  cwd + "/results/" + method +  "_aucs.csv"
    logs.to_csv(pth_to_save)

    print(logs)
    