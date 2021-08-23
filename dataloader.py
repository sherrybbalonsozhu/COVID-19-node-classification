#coding:utf-8
import pickle
import numpy as np

class Example(object):
    def __init__(self,src_nodes,tgt_nodes,edge_index,edge_weights):
        self.src_nodes=src_nodes
        self.tgt_nodes=tgt_nodes
        self.edge_index=edge_index
        self.edge_weights=edge_weights

class InputFeature(object):
    def __init__(self,src_nodes,tgt_nodes,edge_index,edge_weights):
        self.src_nodes=src_nodes
        self.tgt_nodes=tgt_nodes
        self.edge_index=edge_index
        self.edge_weights=edge_weights


class Data(object):
    def __init__(self,path,label_name="ycondition"):
        self.datas=pickle.load(open(path,encoding="utf-8"))
        self.labels=self.datas[label_name]
        self.features=np.array(self.datas['X'])

        self.genes=






