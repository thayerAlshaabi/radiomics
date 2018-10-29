# coding: utf-8
# @author: axemasquelin
# date: 10/28/2018

# Libraries
# -------------------------------------------------#
from deap import base, creator, gp
import matplotlib.pyplot as plt
import operator
#import pygraphviz as pgv
import pandas as pd
import glob
import math
import csv
import os
# -------------------------------------------------#
def evalMultiplexer(individual):
    func = toolbox.compile(expr=individual)
    return sum(func(*in_) == out for in_, out in zip(inputs, outputs)),

pset = gp.PrimitiveSet("MAIN", MUX_TOTAL_LINES, "IN")
pset.addPrimitive(operator.and_, 2)
pset.addPrimitive(operator.or_, 2)
pset.addPrimitive(operator.not_, 1)
pset.addPrimitive(if_then_else, 3)
pset.addTerminal(1)
pset.addTerminal(0)
