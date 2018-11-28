import classifier
from deap import gp
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np


def get_tree(individual, plot=False, fig_number=-1):
    ''' Get and print tree structure '''

    nodes, edges, labels = gp.graph(individual)

    if plot:
        plt.figure(fig_number, figsize=(10,8))
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


def eval_tree(tree, samples):
    ''' Evaluate the sum of correctly identified cases '''

    nfeatures = len(samples[0]) - 1

    nTP = sum(
        bool(tree(*case[:nfeatures])) == bool(case[nfeatures]) \
            for case in samples
    ) / len(samples)

    nFP = sum(
        bool(tree(*case[:nfeatures])) != bool(case[nfeatures]) \
            for case in samples
    ) / len(samples)

    return nTP * pow((1 - nFP), 2)


def parse_features(hof, condition = 'GP', count = 0):
    ''' Get the indecies of the selected features '''

    ''' 
        we need to loop through each guy in HOF 
        and get a set of their 'unique' features  
        -- make sure to pass a figure number to plot trees in different figures
    '''

    featureArray = []
    # featureArrayTemporary = []

    for i in range(len(hof)):
        tree = get_tree(hof[i], plot=True, fig_number=count)
        features = [int(f[2:]) for f in tree.values() if str(f).startswith('f_')]

        # for element in range(len(features)):
        for j in range(len(features)):
            featureArray.append(features[j])
        print("Features: ", featureArray)
        count += 1


    uniques = np.unique(featureArray)
    print('Unique: ', uniques)

        
    return np.unique(featureArray), count

    # plt.figure(i)
    # features = get_tree(hof[0], plot=True)
    # features_idx = [int(f[2:]) for f in features.values() if str(f).startswith('f_')]

    #return features_idx


'''
we need to have the following options:
if 'GP':            use trees as classifiers 
elif 'rf':          use ranfom forest only 
elif 'svm':         use svm only 
elif 'gp-rf':       get features w/ deap then run a random forest
elif 'gp-svm':      get features w/ deap then run a SVM
else:               raise a warning 

we also need a script to save the results to a csv file
to plot these on figure to compare them all together 
'''