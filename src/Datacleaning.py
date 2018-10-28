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
	df1 = pd.read_csv(xl)
	df1.head()
	df1 = df1.dropna()
	return(df1)

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