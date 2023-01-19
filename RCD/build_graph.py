# -*- coding: utf-8 -*-

import dgl
import torch
import networkx as nx

import matplotlib.pyplot as plt


def build_graph(type, node, dir_path):
    # g = dgl.DGLGraph()
    # add 34 nodes into the graph; nodes are labeled from 0~33
    # g.add_nodes(node)
    edge_list = []
    if type == 'direct':
        with open(dir_path + 'graph/K_Directed.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))

        # add edges two lists of nodes: src and dst
        if len(edge_list) > 0:
            src, dst = tuple(zip(*edge_list))
            g = dgl.graph((src, dst), num_nodes=node)
            # g.add_edges(src, dst)
        return g
    elif type == 'undirect':
        with open(dir_path + 'graph/K_Undirected.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        # add edges two lists of nodes: src and dst
        if len(edge_list) > 0:
            src, dst = tuple(zip(*edge_list))
            g = dgl.graph((src, dst), num_nodes=node)
            # g.add_edges(src, dst)
            # edges are directional in DGL; make them bi-directional
            g.add_edges(dst, src)
        return g
    elif type == 'k_from_e':
        with open(dir_path + 'graph/k_from_e.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g = dgl.graph((src, dst), num_nodes=node)
        # g.add_edges(src, dst)
        return g
    elif type == 'e_from_k':
        with open(dir_path + 'graph/e_from_k.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g = dgl.graph((src, dst), num_nodes=node)
        # g.add_edges(src, dst)
        return g
    elif type == 'u_from_e':
        with open(dir_path + 'graph/u_from_e.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g = dgl.graph((src, dst), num_nodes=node)
        # g.add_edges(src, dst)
        return g
    elif type == 'e_from_u':
        # 일반 그래프
        with open(dir_path + 'graph/e_from_u.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g = dgl.graph((src, dst), num_nodes=node)

        # # Hetero 그래프
        # g = dgl.heterograph({
        #     ('node', 'a', 'node'): [],
        #     ('node', 'b', 'node'): [],
        #     ('node', 'c', 'node'): [],
        #     ('node', 'd', 'node'): []
        # })
        # g.add_nodes(node)
        # edge_list = [[], [], [], []]
        # with open(dir_path + 'graph/e_from_u.txt', 'r') as f:
        #     for line in f.readlines():
        #         line = line.replace('\n', '').split('\t')
        #         edge_list[int(line[2])].append((int(line[0]), int(line[1])))
        # # add edges two lists of nodes: src and dst
        # for i, option in zip(range(len(edge_list)), ['a', 'b', 'c', 'd']):
        #     src, dst = tuple(zip(*edge_list[i]))
        #     g.add_edges(src, dst, etype=('node', option, 'node'))
        return g


def get_option_matrix(student_n, exercise_n, dir_path):
    option_matrix = torch.zeros(student_n + exercise_n, exercise_n).to(torch.int64)
    with open(dir_path + 'graph/e_from_u.txt', 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '').split('\t')
            option_matrix[int(line[0])][int(line[1])] = int(line[2])
    return option_matrix