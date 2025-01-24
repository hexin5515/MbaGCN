import torch
import math
import pickle
import os.path as osp
import numpy as np
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.datasets import HeterophilousGraphDataset
from torch_geometric.datasets import Actor
from torch_geometric.datasets import WebKB
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.nn import APPNP
from torch_sparse import coalesce
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import is_undirected, to_undirected
from torch_geometric.io import read_npz


class dataset_heterophily(InMemoryDataset):
    def __init__(self, root='data/', name=None,
                 p2raw=None,
                 train_percent=0.01,
                 transform=None, pre_transform=None):
        if name=='actor':
            name='film'
        existing_dataset = ['chameleon', 'film', 'squirrel']
        if name not in existing_dataset:
            raise ValueError(
                f'name of hypergraph dataset must be one of: {existing_dataset}')
        else:
            self.name = name

        self._train_percent = train_percent

        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(
                f'path to raw hypergraph dataset "{p2raw}" does not exist!')

        if not osp.isdir(root):
            os.makedirs(root)

        self.root = root

        super(dataset_heterophily, self).__init__(
            root, transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.data = Data.from_dict(self.data)
        self.train_percent = self.data.train_percent.item()

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        print('processing...')
        p2f = osp.join(self.raw_dir, self.name)
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


def DataLoader(name):

    name = name.lower()
    if name in ['cora', 'citeseer', 'pubmed']:
        # print('----')
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Planetoid(path, name, transform=T.NormalizeFeatures(), split="geom-gcn")
    elif name in ['computers', 'photo']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Amazon(path, name, T.NormalizeFeatures())
    elif name in ['chameleon', 'actor', 'squirrel']:
        if name in ['chameleon', 'squirrel']:
            dataset = WikipediaNetwork(root='../data/', name=name, geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
        if name in ['actor']:
            dataset = Actor(root='../data/', transform=T.NormalizeFeatures())
    elif name in ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']:
        dataset = HeterophilousGraphDataset(root='../data/', name=name, transform=T.NormalizeFeatures())
    elif name in ['texas', 'cornell', 'wisconsin']:
        dataset = WebKB(root='./data/',name=name, transform=T.NormalizeFeatures())
    elif name in ['arxiv']:
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='../data/', transform=T.ToSparseTensor())
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')

    return dataset
