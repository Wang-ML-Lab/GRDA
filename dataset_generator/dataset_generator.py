import matplotlib
import collections
import csv
import shutil
import os
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
# from utils import read_pickle
# from utils import write_pickle
from scipy.ndimage.interpolation import zoom
import re
import pickle
import networkx as nx
from draw_graph_utils import draw


def read_pickle(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

def write_pickle(data, name):
    with open(name,'wb') as f:
        # the default protocol level is 4
        pickle.dump(data, f)

def show_graph_with_labels(adjacency_matrix, my_angles):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)

    pos = nx.kamada_kawai_layout(gr) # good, littel better than spring

    num_domain = adjacency_matrix.shape[0]

    # expand the graph in horizontal
    for i in range(num_domain):
        pos[i][0] *= 1.4
        pos[i][1] *= 0.8


    labels = dict()
    for i in range(num_domain):
        labels[i] = i

    # use self defined picture drawing picture.
    fig, ax = plt.subplots(1, 1)
    draw(gr, pos, node_radius=0.077, font_color='white', node_angles=my_angles, labels=labels, with_labels=True, ax=ax)
    ax.set_aspect("equal")
    plt.show()

# generate data given the mean/std, radius and number
def generate_data(mean, std, radius, num):
    dim = mean.shape[0]
    m_data = np.random.randn(num, dim)
    print('bingo', m_data.shape)
    m_data *= std[None, :]
    m_radius = m_data[:, 0] ** 2 + m_data[:, 1] ** 2
    m_data += mean[None, :]
    m_data = m_data[m_radius <= radius ** 2, :]

    # random choice
    choice = np.random.choice(m_data.shape[0], size=50, replace=False)
    print(choice)
    m_data = m_data[choice, :]

    print('num of data points within radius', radius, ':', m_data.shape[0])
    return m_data

# generate label for circle-shape data
def generate_label(m_data, radius):
    m_radius = m_data[:, 0] ** 2 + m_data[:, 1] ** 2
    m_label = np.zeros((m_data.shape[0],))
    m_label[m_radius > radius ** 2] = 1
    print("=============")
    print("label 0's num: {}".format(np.sum(m_label == 0)))
    print("label 1's num: {}".format(np.sum(m_label == 1)))
    return m_label

# create dataset, circle-shape domain manifold
def create_toy_data():
    fname = 'toy_d60_spiral.pkl'
    num_domain = 60
    # fname = 'toy_d30_spiral.pkl'
    # l_angle = np.random.rand(15) * np.pi / 2
    # l_angle = np.random.rand(num_domain) * np.pi * 2
    l_angle = np.random.rand(num_domain) * np.pi / 30 * num_domain

    radius_start = 1
    std_small = 1
    radius_step = 0.1
    radius_small = 1

    lm_data = []
    l_domain = []
    l_label = []
    for i, angle in enumerate(l_angle):
        # radius = radius_start + angle / (np.pi / 2) * radius_step
        # radius = radius_start + angle / (np.pi) * radius_step
        radius = radius_start + angle / (np.pi / 30 * num_domain) * radius_step * 60
        mean = np.array([np.cos(angle), np.sin(angle)]) * radius
        std = np.ones((2,)) * std_small
        m_data = generate_data(mean, std, radius_small, 300)
        m_data = np.append(m_data, generate_data(-mean, std, radius_small, 300), axis=0)
        m_label = np.ones(m_data.shape[0],)
        m_label[0:int(0.5 * m_data.shape[0])] = 0
        # m_data = generate_data(mean, std, radius_small, 300)
        # m_label = generate_label(m_data, ra)
        l_label.append(m_label)
        lm_data.append(m_data)
        l_domain.append(np.ones(m_data.shape[0],) * i)
    
    angle_all = np.array(l_angle)
    data_all = np.concatenate(lm_data, axis=0)
    domain_all = np.concatenate(l_domain, axis=0)
    label_all = np.concatenate(l_label, axis=0)# generate_label(data_all, radius_large)

    # generate A
    # A's generation:
    A = np.zeros((num_domain, num_domain))
    for i in range(num_domain):
        for j in range(i + 1, num_domain):
            p = np.cos(angle_all[i]) * np.cos(angle_all[j]) + np.sin(angle_all[i]) * np.sin(angle_all[j])
            
            if num_domain == 15:
                if p < 0.5:
                    c = 0
                else:
                    c = np.random.binomial(1, p)
            elif num_domain == 60:
                if p < 0.2:
                    c = 0
                else:
                    c = np.random.binomial(1, p)

            A[i][j] = c
            A[j][i] = c
    
    show_graph_with_labels(A, angle_all)
    print(angle_all)


    d_pkl = dict()
    d_pkl['data'] = data_all
    d_pkl['label'] = label_all
    d_pkl['domain'] = domain_all
    d_pkl['A'] = A
    d_pkl['angle'] = angle_all
    write_pickle(d_pkl, fname)

    l_style = ['k*', 'r*', 'b*', 'y*', 'k.', 'r.']
    for i in range(2):
        data_sub = data_all[label_all == i, :]
        plt.plot(data_sub[:, 0], data_sub[:, 1], l_style[i])
    plt.grid()
    plt.show()

if __name__ == '__main__':
    create_toy_data()