import pickle
import os
import torch
from scipy import sparse
from data_utils import scipysparse2torchsparse
from torch_geometric.data import Data
from typing import List
from torch_sparse import SparseTensor, cat
import os.path as osp
import copy

# key execution
def get_data(pkl_fname, label="ycondition",verbose=True):
    """From pkl to Pytorch Geometric data object.

    Apply to both train/test.

    Arguments:
        pickle path (str): pkl is dict type with requirements, 'k':value desc:
            'X': features (can be sparse)
            'adj': adjacency matrix (can be scipy.sparse)
            'label' (pd.Series): name for y where y is 1d, length X, etc. and pre-encoded
        sample (str): coming from first argument along with job submission to save
            time consuming features to, grabbing pkl_fnames' path
        replicate (str): concatenate to sample to load model, not necessary if load_attn is none
        load_attn (str): name of attn from {}{}_{}.format(sample,replicate,load_attn) model. Should match label in datapkl
        preloadn2v (bool): if load_attn is not None, should preloaded node2vec edge attr be loaded?
        modelpkl_fname (str): if load_attn is not None, point to model_pkl file
        (default) args to GAT to load back up edge feats. Only used if load_attn is not None


    Returns:
        (torch_geometric.Data)

    """
    pdfp = os.path.split(pkl_fname)[0]

    with open(pkl_fname, 'rb') as f:
        datapkl = pickle.load(f)
        f.close()
    node_features = datapkl['X']
    if isinstance(node_features, sparse.csr_matrix):
        node_features = torch.from_numpy(node_features.todense()).float()
    else:
        node_features = torch.from_numpy(node_features).float()
    labels = datapkl[label]
    if False:
        # assume label_encoding is done in pre-processing steps
        label_encoder = {v: i for i, v in enumerate(labels.unique())}
        labels = labels.map(label_encoder)
        pd.DataFrame(label_encoder, index=[0]).T.to_csv(os.path.join(pdfp, 'ctype_labels_encoding.csv'))
    if False:
        # labels as pd.Series
        labels = torch.LongTensor(labels.to_numpy())
    else:
        labels = torch.LongTensor(labels)  # assumes labels as list
    edge_index, _,_ = scipysparse2torchsparse(datapkl['adj'])
    del datapkl  # clear space

    d = Data(x=node_features, edge_index=edge_index, y=labels)
    del node_features, edge_index, labels
    if verbose:
        print('\nData shapes:')
        print(d)
        print('')

    return d


# REF: pytorch geometric for below data loaders
class ClusterData(torch.utils.data.Dataset):
    r"""Clusters/partitions a graph data object into multiple subgraphs, as
    motivated by the `"Cluster-GCN: An Efficient Algorithm for Training Deep
    and Large Graph Convolutional Networks"
    <https://arxiv.org/abs/1905.07953>`_ paper.

    Args:
        data (torch_geometric.data.Data): The graph data object.
        num_parts (int): The number of partitions.
        recursive (bool, optional): If set to :obj:`True`, will use multilevel
            recursive bisection instead of multilevel k-way partitioning.
            (default: :obj:`False`)
        save_dir (string, optional): If set, will save the partitioned data to
            the :obj:`save_dir` directory for faster re-use.
    """
    def __init__(self, data, num_parts, recursive=False, save_dir=None):
        assert (data.edge_index is not None)

        self.num_parts = num_parts
        self.recursive = recursive
        self.save_dir = save_dir

        self.process(data)

    def process(self, data):
        recursive = '_recursive' if self.recursive else ''
        filename = f'part_data_{self.num_parts}{recursive}.pt'

        path = osp.join(self.save_dir or '', filename)
        if self.save_dir is not None and osp.exists(path):
            data, partptr, perm = torch.load(path)
        else:
            data = copy.copy(data)
            num_nodes = data.num_nodes

            (row, col), edge_attr = data.edge_index, data.edge_attr
            adj_ = SparseTensor(row=row, col=col, value=edge_attr)
            adj, partptr, perm = adj_.partition(self.num_parts, self.recursive)

            for key, item in data:
                if item.size(0) == num_nodes:
                    data[key] = item[perm]

            data.edge_index = None
            data.edge_attr = None
            data.adj = adj

            if self.save_dir is not None:
                torch.save((data, partptr, perm), path)

        self.data = data
        self.perm = perm
        self.partptr = partptr


    def __len__(self):
        return self.partptr.numel() - 1


    def __getitem__(self, idx):
        start = int(self.partptr[idx])
        length = int(self.partptr[idx + 1]) - start

        data = copy.copy(self.data)
        num_nodes = data.num_nodes

        for key, item in data:
            if item.size(0) == num_nodes:
                data[key] = item.narrow(0, start, length)

        data.adj = data.adj.narrow(1, start, length)

        row, col, value = data.adj.coo()
        data.adj = None
        data.edge_index = torch.stack([row, col], dim=0)
        data.edge_attr = value

        return data


    def __repr__(self):
        return (f'{self.__class__.__name__}({self.data}, '
                f'num_parts={self.num_parts})')



class ClusterLoader(torch.utils.data.DataLoader):
    r"""The data loader scheme from the `"Cluster-GCN: An Efficient Algorithm
    for Training Deep and Large Graph Convolutional Networks"
    <https://arxiv.org/abs/1905.07953>`_ paper which merges partioned subgraphs
    and their between-cluster links from a large-scale graph data object to
    form a mini-batch.

    Args:
        cluster_data (torch_geometric.data.ClusterData): The already
            partioned data object.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
    """
    def __init__(self, cluster_data, batch_size=1, shuffle=False, **kwargs):
        class HelperDataset(torch.utils.data.Dataset):
            def __len__(self):
                return len(cluster_data)

            def __getitem__(self, idx):
                start = int(cluster_data.partptr[idx])
                length = int(cluster_data.partptr[idx + 1]) - start

                data = copy.copy(cluster_data.data)
                num_nodes = data.num_nodes
                for key, item in data:
                    if item.size(0) == num_nodes:
                        data[key] = item.narrow(0, start, length)

                return data, idx

        def collate(batch):
            data_list = [data[0] for data in batch]
            parts: List[int] = [data[1] for data in batch]
            partptr = cluster_data.partptr

            adj = cat([data.adj for data in data_list], dim=0)

            adj = adj.t()
            adjs = []
            for part in parts:
                start = partptr[part]
                length = partptr[part + 1] - start
                adjs.append(adj.narrow(0, start, length))
            adj = cat(adjs, dim=0).t()
            row, col, value = adj.coo()

            data = cluster_data.data.__class__()
            data.num_nodes = adj.size(0)
            data.edge_index = torch.stack([row, col], dim=0)
            data.edge_attr = value

            ref = data_list[0]
            keys = ref.keys
            keys.remove('adj')

            for key in keys:
                if ref[key].size(0) != ref.adj.size(0):
                    data[key] = ref[key]
                else:
                    data[key] = torch.cat([d[key] for d in data_list],
                                          dim=ref.__cat_dim__(key, ref[key]))

            return data

        super(ClusterLoader,
              self).__init__(HelperDataset(), batch_size, shuffle,
                             collate_fn=collate, **kwargs)


def get_data_loader(data,number_parts=5000,batch_size=256,shuffle=True):
    # data loader for mini-batching
    cd = ClusterData(data,num_parts=number_parts)
    cl = ClusterLoader(cd,batch_size=batch_size,shuffle=shuffle)
    return cl

if __name__=="__main__":
    data=get_data(pkl_fname="F:/MyProjects_COVID19/datapkls/liao_test_200529.pkl",label="ycondition")

    cl=get_data_loader(data)

    print(len(cl))

    for batch in cl:
        print(len(batch))




