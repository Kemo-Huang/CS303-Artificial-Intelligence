import networkx as nx
import matplotlib.pyplot as plt
import CARP_solver


def draw_graph(e):
    G = nx.Graph()
    G.add_weighted_edges_from(e)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_size=30, with_labels=True)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)
    # nx.draw_networkx_labels(G, pos)
    plt.show()


if __name__ == '__main__':
    edges = []
    with open('sample/gdb10.dat', 'r') as file:
        for i in range(9):
            file.readline()
        line = file.readline()
        while line != 'END':
            edges.append(list(map(int, line.split()[:-1])))
            line = file.readline()
    draw_graph(edges)
