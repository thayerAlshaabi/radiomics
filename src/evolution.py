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
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from deap import gp
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
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
        self.numfeatures = len(dataset[0]) - 1 # number of features
        
        # setup evolution env.
        self.pset = gp.PrimitiveSetTyped(
            'MAIN', # codename for the dataset
            itertools.repeat(float, self.numfeatures), # type and number of features
            bool, # type of the target value
            'f_', # a prefix to encode feature names
        )

        self.toolbox = base.Toolbox()

        # set operator and terminal sets
        self.set_opts()      

        # set evolution functions
        self.populate()
        self.mutation()
        self.crossover()
        self.selection()
        self.fitness()  


    def run(self, reps):
        '''
            Run Evolution and return statistical logs and best individuals
        '''
        # set a fixed seed for the random number generator
        random.seed(reps*10)

        pop = self.toolbox.population(self.popsize)
        hof = tools.HallOfFame(self.hofsize)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop, logbook = algorithms.eaSimple(
            pop, 
            self.toolbox,
            cxpb = self.cx, 
            mutpb = self.mut, 
            ngen = self.maxgen,
            stats = stats, 
            halloffame = hof, 
            verbose = 1
        )
        return pop, logbook, hof, 


    def set_opts(self):
        '''
            Define operators set. 
        '''
        # boolean operators
        self.pset.addPrimitive(operator.and_, [bool, bool], bool)
        self.pset.addPrimitive(operator.or_, [bool, bool], bool)
        self.pset.addPrimitive(operator.not_, [bool], bool)

        # floating point operators
        self.pset.addPrimitive(operator.add, [float,float], float)
        self.pset.addPrimitive(operator.sub, [float,float], float)
        self.pset.addPrimitive(operator.mul, [float,float], float)
        self.pset.addPrimitive(self.div, [float,float], float)

        # logic operators
        #self.pset.addPrimitive(operator.gt, [float, float], bool)
        self.pset.addPrimitive(operator.lt, [float, float], bool)
        self.pset.addPrimitive(operator.eq, [float, float], bool)
        self.pset.addPrimitive(self.ifelse, [bool, float, float], float)

        # terminals
        self.pset.addEphemeralConstant( # random constant
            "rand100", lambda: random.random() * 100, float
        )
        self.pset.addTerminal(False, bool)
        self.pset.addTerminal(True, bool)


    def mutation(self,
        #individuals,
        #age
        ):
        '''
            Define a mutation function
        '''
        ''' Axel's Swap Mutation - Unsure how we are planning to store whether its an elif and all that. ???
        #Individuals (3D matrix - first layer of matrix is population/genome, second layer is age of individuals)
        #Mutation - holds index of individuals that will be mutated
        SwapSpring = np.zeros(len(individuals),len(mutation),2)
        for i in range(len(mutation)):
            location = mutation[i]
            rowi = individuals[1][i][:]
            a = np.random.permutation(len(individuals[1][i]))
            temp = rowi[a[2]]
            rowi[a[2]] = row[a[1]]
            rowi[a[1]] = temp
            SwapSpring[1][i][:] = rowi #Storing mutated genome
            SwapSpring[2][i] = individuals[2][i] #Storing Age of genome.
        '''

        self.toolbox.register("expr_mut", # mutation criteria
            gp.genFull, min_=0, max_=2
        )
        
        self.toolbox.register("mutate", # mutation function
            gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset
        )
    

    def crossover(self, ):
        '''
            Define a crossover function
        '''
        ''' Axel's CrossOver
        ****Check CycleCross.py in notebook folder
        '''

        self.toolbox.register("mate",  # crossover function
            gp.cxOnePoint #Cannot use tools as they work of sequences and are not compatible with GP inputs
        )


    def selection(self, ):
        '''
            Define a selection function
        '''
        self.toolbox.register("select", # selection function 
            tools.selTournament, tournsize=3
        )


    def fitness(self, ):
        '''
            Define a fitness function
        '''
        def eval(individual):
            nTP = 0
            nFP = 0
            nCases = 100
            # transform the tree expression in a callable function
            func = self.toolbox.compile(expr=individual)
            
            # randomly sample 100 cases from the dataset for testing
            test_samples = random.sample(self.dataset, nCases)
            
            #Fitnes of TPR and FPR

            # evaluate the sum of correctly identified cases
            nTP = sum(
                bool(func(*case[:self.numfeatures])) == bool(case[self.numfeatures]) \
                    for case in test_samples
            ) / nCases

            nFP = sum(
                bool(func(*case[:self.numfeatures])) != bool(case[self.numfeatures]) \
                    for case in test_samples
            ) / nCases

            result = nTP * pow((1 - nFP),2)
            return result,

        self.toolbox.register("evaluate", eval)


    def get_tree(self, individual, plot=False):
        '''
            Print tree structure
        '''
        expr = gp.genFull(self.pset, min_=1, max_=3)

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

    
    def populate(self, ):
        '''
            Define population scheme
        '''
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", 
            gp.PrimitiveTree, fitness=creator.FitnessMax
        )
        
        self.toolbox.register("expr", # create expression
            gp.genHalfAndHalf, pset=self.pset, min_=1, max_=2
        )
        
        self.toolbox.register("individual", # create individual (tree)
            tools.initIterate, creator.Individual, self.toolbox.expr
        )
        
        self.toolbox.register("population",  # create a population
            tools.initRepeat, list, self.toolbox.individual
        )
        
        self.toolbox.register("compile", 
            gp.compile, pset=self.pset
        )


    def div(self, numerator, denominator): 
        '''
            Override division operator to handle division by zero
        '''
        try: 
            return numerator / denominator
        except ZeroDivisionError: 
            return 1


    def ifelse(self, condition, case1, case2):
        '''
            Define a new operator for if/else 
        '''
        if condition:  
            return case1
        else: 
            return case2