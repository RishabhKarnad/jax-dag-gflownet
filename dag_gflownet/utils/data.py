import pandas as pd
import urllib.request
import gzip

import numpy as np
import networkx as nx

from pathlib import Path
from numpy.random import default_rng
from pgmpy.utils import get_example_model
from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.factors.continuous import LinearGaussianCPD

from dag_gflownet.utils.graph import sample_erdos_renyi_linear_gaussian
from dag_gflownet.utils.sampling import sample_from_linear_gaussian


def download(url, filename):
    if filename.is_file():
        return filename
    filename.parent.mkdir(exist_ok=True)

    # Download & uncompress archive
    with urllib.request.urlopen(url) as response:
        with gzip.GzipFile(fileobj=response) as uncompressed:
            file_content = uncompressed.read()

    with open(filename, 'wb') as f:
        f.write(file_content)

    return filename


def load_data_from_file(directory):
    adjacency = np.load(f'{directory}/G.npy')

    nodes = list(map(str, range(adjacency.shape[0])))

    graph = nx.from_numpy_array(
        adjacency, create_using=LinearGaussianBayesianNetwork)
    mapping = dict(enumerate(nodes))
    nx.relabel_nodes(graph, mapping=mapping, copy=False)

    theta_true = np.load(f'{directory}/theta.npy')
    cov = np.load(f'{directory}/cov.npy')

    factors = []
    for node in graph.nodes:
        i = int(node)
        parents = list(graph.predecessors(node))

        parents_idx = list(map(int, parents))

        theta = theta_true[parents_idx, i]
        theta = np.insert(theta, 0, 0.0)

        obs_noise = cov[i, i] ** 0.5

        # Create factor
        factor = LinearGaussianCPD(node, theta, obs_noise, parents)
        factors.append(factor)

    graph.add_cpds(*factors)

    data = np.load(f'{directory}/data.npy')
    data = pd.DataFrame(data, columns=list(graph.nodes()))

    score = 'bge'

    return graph, data, score


def get_data(name, args, rng=default_rng()):
    if name == 'erdos_renyi_lingauss':
        graph = sample_erdos_renyi_linear_gaussian(
            num_variables=args.num_variables,
            num_edges=args.num_edges,
            loc_edges=0.0,
            scale_edges=1.0,
            obs_noise=0.1,
            rng=rng
        )
        data = sample_from_linear_gaussian(
            graph,
            num_samples=args.num_samples,
            rng=rng
        )
        score = 'bge'

    elif name == 'sachs_continuous':
        graph = get_example_model('sachs')
        filename = download(
            'https://www.bnlearn.com/book-crc/code/sachs.data.txt.gz',
            Path('data/sachs.data.txt')
        )
        data = pd.read_csv(filename, delimiter='\t', dtype=float)
        data = (data - data.mean()) / data.std()  # Standardize data
        score = 'bge'

    elif name == 'sachs_interventional':
        graph = get_example_model('sachs')
        filename = download(
            'https://www.bnlearn.com/book-crc/code/sachs.interventional.txt.gz',
            Path('data/sachs.interventional.txt')
        )
        data = pd.read_csv(filename, delimiter=' ', dtype='category')
        score = 'bde'

    else:
        raise ValueError(f'Unknown graph type: {name}')

    return graph, data, score
