# coding: utf-8
# @author: sami
# date: 10/30/2018

# Libraries
# -------------------------------------------------#
#from sklearn.preprocessing import StandardScalar
from deap import base, creator, gp, tools
#from sklearn.util import resample
import matplotlib.pyplot as plt
#import pygraphviz as pgv
#import pandas as pd
import operator
import glob
import math
import csv
import os
import random
# -------------------------------------------------#

numParameters = 3 #this is arbitrary - will change based on how many we decide to use

pset = gp.PrimitiveSet("main", numParameters)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
# add more operator types - keeping it at 3 for initial debugging

pset.addEphemeralConstant("name", lambda: random.uniform(-1, 1))
# the above line (about adding ephemerals) is buggy with my IDE (Spyder).
# if it's buggy for you, comment out, and instead use the below 2 lines
#pset.addTerminal(1,) #comment this line out when you test it unless ephemeral is buggy for you too
#pset.addTerminal(0,) #comment this line out when you test it unless ephemeral is buggy for you too

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.expr)

def main():
    
    #create a population of size 100
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    population = toolbox.population(n=100)

    # demonstrate how to create an individual and evaluate it:
    tree = toolbox.individual()
    fitnessOfTree = evaluateFitness(tree)
     
def evaluateFitness(individual):
    # currently i am just having the evaluate fitness function actually
    # solve the equation. to be implemented: compare to binary yes/no
    individualString = str(individual)
   # print(individualString) #uncomment to understand more about how this works
    function = gp.compile(individualString, pset)
    
  #  print(function(1, 2, 3)) #uncomment to understand more about how this works
   
    # currently have three parameters (1,2,3), because numParameters (line 21) = 3
    return function(1, 2, 3)

def importData(filename):
    # this is not finished (will look into this more with group)
    # https://docs.python.org/3/library/csv.html is very useful
    with open(filename, newline = '') as csvfile:
        raw = csv.reader(csvfile, delimiter = ' ', quotechar = '|')
        for row in raw:
            print(', '.join(row))
    
main()