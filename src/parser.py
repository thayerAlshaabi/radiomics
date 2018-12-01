import classifier
from deap import gp
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np
import utils
from sklearn.metrics import roc_curve


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


def get_tree_predictions(tree, samples):
    ''' Run a tree on a testing set and get its predictions '''

    predictions = []
    nfeatures = len(samples[0]) - 1

    for case in samples:
        if tree(*case[:nfeatures]):
            predictions.append(1)
        else:
            predictions.append(0)
    
    return predictions


def eval_tree(groud_truth, predictions):
    ''' Evaluate the ratio of correctly identified cases '''

    correct_cases = 0

    for i in range(len(predictions)):
        if groud_truth[i] == predictions[i]:
            correct_cases += 1
    
    return correct_cases / len(groud_truth)


def eval_hof(hof, X_test, y_test):
    ''' Evaluate trees in HOF on the testing set '''

    votes = np.zeros((len(hof), len(y_test)))
    samples = np.column_stack((X_test, y_test))

    for i in range(len(hof)):
        votes[i, :] = get_tree_predictions(hof[i], samples)
    
    y_pred = votes.mean(axis=0)

    fpr, tpr, _ = roc_curve(y_test, y_pred)

    return (fpr, tpr)


def parse_features(hof, fig_counter=2):
    ''' Get the indecies of the selected features '''

    featureArray = []

    for i in range(len(hof)):
        tree = get_tree(hof[i], plot=False, fig_number=fig_counter)
        features = [int(f[2:]) for f in tree.values() if str(f).startswith('f_')]

        # for element in range(len(features)):
        for j in range(len(features)):
            featureArray.append(features[j])
        #print("Features: ", featureArray)
        fig_counter += 1


    uniques = np.unique(featureArray)
    print('Selected Features: ', uniques)
  
    return np.unique(featureArray), fig_counter