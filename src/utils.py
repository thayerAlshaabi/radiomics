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
import sys, os
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
# ---------------------------------------------------------------------------- #


def progressBar(value, endvalue, bar_length=50):
    '''
        A simple function to print a progressBar on the screen
    '''
    percent = float(value) / endvalue
    arrow = '=' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r[{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))


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
            random_state=2018
        )
        return pd.concat([df_pos_upsampled, df_neg])

    # Downsample the neg samples
    elif method == 2:
        df_neg_downsampled = resample(
            df_neg,
            n_samples=len(df_pos),
            replace=True, 
            random_state=2018
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