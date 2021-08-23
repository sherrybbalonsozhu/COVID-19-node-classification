#coding:utf-8
import torch

def scipysparse2torchsparse(x) :
    '''
    Input: scipy csr_matrix
    Returns: torch tensor in experimental sparse format

    REF: Code adatped from [PyTorch discussion forum](https://discuss.pytorch.org/t/better-way-to-forward-sparse-matrix/21915>)
    '''
    samples=x.shape[0]
    features=x.shape[1]
    values=x.data
    coo_data=x.tocoo()
    indices=torch.LongTensor([coo_data.row,coo_data.col]) # OR transpose list of index tuples
    t=torch.sparse.FloatTensor(indices,torch.from_numpy(values).float(),[samples,features])
    return indices,values,t