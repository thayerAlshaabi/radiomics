# coding: utf-8

""" MIT License """
'''
    Axel Masquelin & Sami Connolly  
    Andrea Elhajj  & Thayer Alshaabi
    ---
    Copyright (c) 2018 
'''

# Libraries and dependencies
# -------------------------------------------------#
import random
import operator
import numpy as np
import sys, os
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

from deap import gp
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import warnings
warnings.filterwarnings("ignore")

from pathos.helpers import cpu_count
from pathos.multiprocessing import ProcessingPool
# -------------------------------------------------#

class Evolution:
    
    def __init__(self,  # Constructor
        dataset,        # numpy matrix 
        popsize = 100,  # initial population size
        hofsize = 1,    # the number of best individual to track
        cx = .5,        # crossover rate
        mut = .2,       # mutation rate
        maxgen = 50,    # max number of generations
    ):
        self.dataset = dataset
        self.popsize = popsize # set pop size
        self.hofsize = hofsize # set hof size
        self.cx = cx           # set crossover rate
        self.mut = mut         # set mutation rate
        self.maxgen = maxgen   # set max number of generations 
        self.nfeatures = len(dataset[0]) - 1 # number of features
        self.pset = self.build_primitive_set()
        self.toolbox = self.build_toolbox()


    def build_primitive_set(self):
        ''' Define operators/terminals set. '''

        # setup evolution env.
        pset = gp.PrimitiveSetTyped(
            'MAIN', # codename for the dataset
            itertools.repeat(float, self.nfeatures), # type and number of features
            bool, # type of the target value
            'f_', # a prefix to encode feature names
        )

        # boolean operators 
        pset.addPrimitive(operator.and_, [bool, bool], bool)
        pset.addPrimitive(operator.or_, [bool, bool], bool)
        pset.addPrimitive(operator.not_, [bool], bool)

        # floating point operators
        pset.addPrimitive(np.add, [float,float], float)
        pset.addPrimitive(np.subtract, [float,float], float)
        pset.addPrimitive(np.multiply, [float,float], float)
        pset.addPrimitive(self.div, [float,float], float)
        pset.addPrimitive(np.exp, [float], float)
        pset.addPrimitive(np.sin, [float], float)
        pset.addPrimitive(np.cos, [float], float)
        pset.addPrimitive(np.log, [float], float)

        # logic operators
        pset.addPrimitive(operator.gt, [float, float], bool)
        pset.addPrimitive(operator.lt, [float, float], bool)
        pset.addPrimitive(operator.eq, [float, float], bool)
        pset.addPrimitive(self.ifelse, [float, float], float)

        # terminals
        pset.addEphemeralConstant( # random constant
            "rand100", lambda: random.random() * 100, float
        )

        pset.addTerminal(False, bool)
        pset.addTerminal(True, bool)

        return pset


    def build_toolbox(self):
        ''' Define functions to use in the GP toolbox '''

        toolbox = base.Toolbox()

        creator.create("FitnessMax", 
                        base.Fitness, 
                        weights=(1.0,))

        creator.create("Individual",
                        gp.PrimitiveTree,
                        fitness=creator.FitnessMax,
                        pset=self.pset)

        toolbox.register("initializer",
                        gp.genHalfAndHalf,
                        pset=self.pset,
                        min_=1,
                        max_=2)

        toolbox.register("tree",
                        tools.initIterate,
                        creator.Individual,
                        toolbox.initializer)

        toolbox.register("population",
                        tools.initRepeat,
                        list,
                        toolbox.tree)

        toolbox.register("expr_mut",
                        gp.genHalfAndHalf,
                        min_=0,
                        max_=2)

        # Fitness function similar to Wu & Banzhaf, 2001.
        # Fi = TPRi x (1 - FPRi)^2
        toolbox.register("evaluate", self.fitness)

        # One-CrossoverPoint
        toolbox.register("mate", gp.cxOnePoint)

        # Uniform Mutation
        toolbox.register("mutate", 
                        gp.mutEphemeral,
                        mode="all")
           #             expr=toolbox.expr_mut, 
           #             pset=self.pset)
        '''
        toolbox.register("select", # selection function 
                tools.selTournament, 
                tournsize=3) 
        '''
        # DoubleTournament Selection uses the size of the individuals
        # in order to discriminate good solutions.
        
        toolbox.register("select", # selection function 
                        tools.selDoubleTournament, 
                        fitness_size = 7, # of individuals participating in each fitness tournament
                        parsimony_size = 1.8, # of individuals participating in each size tournament
                        fitness_first=True)
        
        # Control code-bloat: max depth of a tree
        toolbox.decorate("mate", 
                        gp.staticLimit(key=operator.attrgetter("height"), 
                        max_value=17))

        toolbox.decorate("mutate", 
                        gp.staticLimit(key=operator.attrgetter("height"), 
                        max_value=17))


        # enable multiprocessing 
        # pool = ProcessingPool(cpu_count())
        # toolbox.register('map', pool.map)
        
        return toolbox


    def fitness(self, individual):
        ''' Fitness function similar to Wu & Banzhaf, 2001. '''

        def eval(tree, samples):
            # evaluate the sum of correctly identified cases
            nTP = sum(
                bool(tree(*case[:self.nfeatures])) == bool(case[self.nfeatures]) \
                    for case in samples
            ) / len(samples)

            nFP = sum(
                bool(tree(*case[:self.nfeatures])) != bool(case[self.nfeatures]) \
                    for case in samples
            ) / len(samples)

            return nTP * pow((1 - nFP), 2)

        # transform the tree expression in a callable function
        tree = gp.compile(individual, self.pset)
        
        # randomly sample cases from the dataset to use as test cases
        samples = random.sample(self.dataset, len(self.dataset)//2)

        return eval(tree, samples),


    def assess(self, pop):
        ''' Evaluate fitness of individuals in the current population '''
        
        # find individuals that has not been evaluated
        individuals = [i for i in pop if not i.fitness.valid]

        # run fitness function on the selected individuals
        fitnesses = self.toolbox.map(self.toolbox.evaluate, individuals)
        
        # update fitness values
        for i, fit in zip(individuals, fitnesses):
            i.fitness.values = fit
        
        return individuals


    def run(self, verbose=0):
        ''' Run Evolution and return statistical logs and best individuals '''

        pop = self.toolbox.population(self.popsize)
        hof = tools.HallOfFame(self.hofsize)

        # create a logbook to keep track of generational updates
        logbook = tools.Logbook()
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('avg', np.mean)
        stats.register('std', np.std)
        stats.register('min', np.min)
        stats.register('max', np.max)
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # evaluate first set of individuals 
        individuals = self.assess(pop)

        # update the hall
        if hof is not None: hof.update(pop)
        
        # update logbook
        record = stats.compile(pop) if stats else {}
        logbook.record(gen=0, nevals=len(individuals), **record)
        
        if verbose: print(logbook.stream)

        # the generational loop
        for gen in range(1, self.maxgen + 1):
            np.random.seed(2018)
            # select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))

            # vary the pool of individuals
            offspring = algorithms.varAnd(offspring, self.toolbox, self.cx, self.mut)

            # evaluate current population
            individuals = self.assess(offspring)

            # update the hall of fame with the generated individuals
            if hof is not None: hof.update(offspring)

            # replace the current population by the offspring
            pop[:] = offspring

            # update logbook
            record = stats.compile(pop) if stats else {}
            logbook.record(gen=gen, nevals=len(individuals), **record)

            if verbose: print(logbook.stream)

        return pop, logbook, hof, 


    def get_tree(self, individual, plot=False):
        ''' Print tree structure '''

        nodes, edges, labels = gp.graph(individual)

        if plot:
            plt.figure(1, figsize=(10,8))
            tree = nx.Graph()
            tree.add_nodes_from(nodes)
            tree.add_edges_from(edges)
            pos = graphviz_layout(tree, prog="dot")

            nx.draw_networkx_nodes(tree, pos, node_color='white')
            nx.draw_networkx_edges(tree, pos)
            nx.draw_networkx_labels(tree, pos, labels)

            ax = plt.gca()
            ax.set_axis_off()

        return labels

    
    def div(self, numerator, denominator): 
        ''' Override division operator to handle division by zero '''

        try: 
            return numerator / denominator
        except ZeroDivisionError: 
            return 1


    def ifelse(self, feature1, feature2):
        ''' Define a new operator for if/else '''
        
        if feature1 < feature2:  
            return -feature1
        else: 
            return feature1