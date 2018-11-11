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
######## Matlab Script ###########
#def crossover(self, ):
    '''
        Define a crossover function
    '''
    ''' Axel's Cycle CrossOver
    CycleSprings = zeros(size(crossover,2)/2, size(population,2)); %Memory allocation for offsprings
	for i = 1 : size(crossover,2)/2
    parent1 = population(crossover(2*i-1),:);
    parent2 = population(crossover(2*i),:);
    
    k = 1;
    indeces = [];
    stopCycle = parent1(1);
    lastvalue = parent2(1);
    indeces(k) = 1;
    while lastvalue ~= stopCycle
        for n=1:length(parent1)
            if parent1(n) == lastvalue
                k = k + 1;
                indeces(k) = n;
                lastvalue = parent2(n);
                break
            end
        end
    end
    for n=1:length(parent1)
        for m=1:length(indeces)
            if n == indeces(m)
                flag = 1;
                break
            else
                flag = 0;
            end
        end
        
        if flag == 1
            CycleSprings(i,n) = parent1(n);
        else 
            CycleSprings(i,n) = parent2(n);
        end
        
    end

end


    '''

#    self.toolbox.register("mate",  # crossover function
#        gp.cxOnePoint
#    )