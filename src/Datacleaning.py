# coding: utf-8
# @author: axemasquelin
# date: 10/28/2018

# Libraries
# -------------------------------------------------#
import glob
import matplotlib.pyplot as plt
import math
import pandas as pd
import csv
from pandas import ExcelWriter
import os

# -------------------------------------------------#

def LoadFile(xl):
	df = pd.read_csv(xl)
	df.head()
	df = df.dropna(thresh=len(df)*.90, axis=1)
	df = df.dropna(thresh=df.shape[1]*.5)
	df = df.dropna(axis=1, subset=df['ca'])
	df = df.apply(lambda col: col.fillna(col.mean()))
	df.isnull().sum().any()
	return(df)

#Change working directory to dataset directory
if ((os.path.splitext(os.path.basename(os.getcwd()))[0]) != 'dataset'):
	os.chdir('./dataset')

#Finding all csv Files
csvFiles = glob.glob(os.getcwd() + '\*.csv')
filename = os.path.splitext(csvFiles[0])[0]
#Loading First CSV File
datframe = LoadFile(csvFiles[0])

#Saving over dataset after removing missing data
datframe.to_csv(filename + 'modified' + '.csv')