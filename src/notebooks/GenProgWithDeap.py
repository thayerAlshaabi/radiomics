# coding: utf-8
# @author: axemasquelin
# date: 10/28/2018

# Libraries
# -------------------------------------------------#
from sklearn.preprocessing import StandardScalar
from deap import base, creator, gp
from sklearn.util import resample
import matplotlib.pyplot as plt
#import pygraphviz as pgv
import pandas as pd
import operator
import random
import glob
import math
import csv
import os
# -------------------------------------------------#

def main():
	'''Evolution Parameters'''
	MaxGen = 100
	PopSize = 100
	Pcross = 0.8
	Pmut = 0.1 

	'''Seeding Experiments'''
	random.seed(1)
	'''Population Initialization'''
	pop = toolbox.population(PopSize)
	hof = tools.HallOfFame(5) #Store top 5 Individuals
		
	'''Model Defintion'''
	pset = gp.PrimitiveSet("MAIN", MUX_TOTAL_LINES, "IN")
	pset.addPrimitive(operator.and_, 2)
	pset.addPrimitive(operator.or_, 2)
	pset.addPrimitive(operator.not_, 1)
	pset.addPrimitive(if_then_else, 3)
	pset.addTerminal(1)
	pset.addTerminal(0)	

	#Trying to figure out how to initialize/load population
	pop, log = algorithms.eaSimple(pop, toolbox, Pcross, Pmut, MaxGen, stats=mstats,
	                                   halloffame=hof, verbose=True)

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

###Evaluation 
def evalMultiplexer(individual):
    func = toolbox.compile(expr=individual)
    return sum(func(*in_) == out for in_, out in zip(inputs, outputs)),


#Call main function
main()