# coding: utf-8
# @author: sami
# date: 10/30/2018

# Libraries
# -------------------------------------------------#
from sklearn.preprocessing import StandardScalar
from deap import base, creator, gp
from sklearn.util import resample
import matplotlib.pyplot as plt
#import pygraphviz as pgv
import pandas as pd
import operator
import glob
import math
import csv
import os
# -------------------------------------------------#


pset = gp.PrimitiveSet("MAIN", MUX_TOTAL_LINES, "IN")
pset.addPrimitive(operator.and_, 2)
pset.addPrimitive(operator.or_, 2)
pset.addPrimitive(operator.not_, 1)
pset.addPrimitive(operator.le, 2)
pset.addPrimitive(operator.ge, 2)

#pset.addPrimitive(if_then_else, 3)
# less than, greater than equal to, etc
pset.addTerminal(1)
pset.addTerminal(0)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# create individuals

# create population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.population(n=100)


###Crossover Function
def CrossEvol(parents):
	offpsrings = []
	for i in range(len(parents)/2):
		Xchild = deap.tools.cxOnePoint(parent[(2*i-1)], parent[(2*i)])
		offpsrings.append(child)

	return(offpsrings)

###Mutation Function
def Mutation(parents):
	offsprings = []
	for i in range(len(parents)):
		mChild = deap.gp.mutNodeReplacement(parents[i])
		offpsrings.append(mChild)

	return(offsprings)

###Selection Function - Offspring Selection?
def Tournament(population, rep, tourSize):
	deap.tools.selTournament(population, rep, tournSize, fit_attr = 'fitness')


def evalMultiplexer(individual):
    func = toolbox.compile(expr=individual)
    return sum(func(*in_) == out for in_, out in zip(inputs, outputs)),




pop = toolbox.population(n=300)
hof = tools.HallOfFame(5) #Best 5 Individuals to ever live
#Trying to figure out how to initialize/load population
pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,
                                   halloffame=hof, verbose=True)
