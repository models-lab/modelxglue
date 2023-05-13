from collections import defaultdict

from grakel import Graph
from grakel.kernels import WeisfeilerLehman


def get_pairwise_kernel(graphs_train, graphs_test, kernel='WL'):
    G_train = [to_grakel_graph(g) for g in graphs_train]
    G_test = [to_grakel_graph(g) for g in graphs_test]
    if kernel == 'WL':
        kernel = WeisfeilerLehman(n_iter=3)
    X_train = kernel.fit_transform(G_train)
    X_test = kernel.transform(G_test)
    return X_train, X_test


def to_grakel_graph(graph_data):
    edges_dic = defaultdict(list)
    for edge in graph_data["links"]:
        edges_dic[edge["source"]].append(edge["target"])

    node_labels = {}
    for node in graph_data["nodes"]:
        node_labels[node["id"]] = node["name"].lower() if ("name" in node) and (not node["name"] is None) else '<unk>'

    return Graph(dict(edges_dic), node_labels=node_labels)
