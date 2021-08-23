#coding:utf-8
import numpy as np
import pandas as pd
import scanpy as sc
import os
from scipy import sparse
from sklearn.model_selection import train_test_split
import pickle
from data_utils import scipysparse2torchsparse

dir="F:/MyProjects_COVID19/GSE145926_RAW/"
# adatas=[]
# for name in os.listdir(dir):
#     fname=os.path.join(dir,name)
#     # 可以直接读取10Xgenomics的.h5格式数据
#     adata=sc.read_10x_h5(fname, genome=None, gex_only=True)
#     adata.var_names_make_unique()
#     adatas.append(adata)

adata141=sc.read_10x_h5(os.path.join(dir,"GSM4339769_C141_filtered_feature_bc_matrix.h5"),genome=None,gex_only=None)

adata142=sc.read_10x_h5(os.path.join(dir,"GSM4339770_C142_filtered_feature_bc_matrix.h5"),genome=None,gex_only=None)
adata143=sc.read_10x_h5(os.path.join(dir,"GSM4339771_C143_filtered_feature_bc_matrix.h5"),genome=None,gex_only=None)
adata144=sc.read_10x_h5(os.path.join(dir,"GSM4339772_C144_filtered_feature_bc_matrix.h5"),genome=None,gex_only=None)
adata145=sc.read_10x_h5(os.path.join(dir,"GSM4339773_C145_filtered_feature_bc_matrix.h5"),genome=None,gex_only=None)
adata146=sc.read_10x_h5(os.path.join(dir,"GSM4339774_C146_filtered_feature_bc_matrix.h5"),genome=None,gex_only=None)
adata51=sc.read_10x_h5(os.path.join(dir,"GSM4475048_C51_filtered_feature_bc_matrix.h5"),genome=None,gex_only=None)
adata52=sc.read_10x_h5(os.path.join(dir,"GSM4475049_C52_filtered_feature_bc_matrix.h5"),genome=None,gex_only=None)
adata100=sc.read_10x_h5(os.path.join(dir,"GSM4475050_C100_filtered_feature_bc_matrix.h5"),genome=None,gex_only=None)
adata148=sc.read_10x_h5(os.path.join(dir,"GSM4475051_C148_filtered_feature_bc_matrix.h5"),genome=None,gex_only=None)
adata149=sc.read_10x_h5(os.path.join(dir,"GSM4475052_C149_filtered_feature_bc_matrix.h5"),genome=None,gex_only=None)
adata152=sc.read_10x_h5(os.path.join(dir,"GSM4475053_C152_filtered_feature_bc_matrix.h5"),genome=None,gex_only=None)

#class
adata141.obs["Condition"]='mild'
adata142.obs['Condition']='mild'
adata143.obs['Condition']='severe'
adata144.obs['Condition']='mild'
adata145.obs['Condition']='severe'
adata146.obs['Condition']='severe'
adata148.obs['Condition']='severe'
adata149.obs['Condition']='severe'
adata152.obs['Condition']='severe'
adata51.obs['Condition']='healthy'
adata52.obs['Condition']='healthy'
adata100.obs['Condition']='healthy'


# adataGSM3660650=sc.read_10x_mtx("F:/MyProjects_COVID19/GSM3660650")
# adataGSM3660650.obs['Condition']='healthy'
adatas=[adata141,adata142,adata143,adata144,adata145,adata146,adata51,adata52,adata148,adata149,adata152]
for adata in adatas:
    adata.var_names_make_unique()

def filter(adata):
    sc.pp.filter_cells(adata, min_genes=200)  # 去除表达基因200以下的细胞
    sc.pp.filter_genes(adata, min_cells=3)  # 去除在3个细胞以下表达的基因
    return adata


adata=sc.concat(adatas)


print(adata)
adata=filter(adata)
print(adata)

a=adata.obs.groupby("Condition").count()
print(a)

#PCA
sc.tl.pca(adata, svd_solver='arpack') # PCA分析
# sc.pl.pca(adata, color='CST3') #绘图

#计算细胞间的距离
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
#聚类
sc.tl.louvain(adata)


print(adata)
print(adata.X.shape)

vars=adata.var_names
cells=adata.obs_names

edge_index,values,adj=scipysparse2torchsparse(adata.X)

print(values.max(),values.min())

# split the data AND kick out mock infected cells as these are noise
idx_train, idx_test = train_test_split(adata.obs.index, train_size=0.7)
tdata = sc.AnnData(X=adata[(adata.obs.index.isin(idx_train)),:].X,
                   obs=adata[(adata.obs.index.isin(idx_train)),:].obs,
                   )
idx_val, idx_test = train_test_split(idx_test, train_size=0.5)
val = sc.AnnData(X=adata[(adata.obs.index.isin(idx_val)) ,:].X,
                  obs=adata[(adata.obs.index.isin(idx_val)) ,:].obs)
test = sc.AnnData(X=adata[(adata.obs.index.isin(idx_test)) ,:].X,
                  obs=adata[(adata.obs.index.isin(idx_test)) ,:].obs)


def graph_pp(AnnData, bbknn=False):
    sc.tl.pca(AnnData, n_comps=100)
    if bbknn:
        sc.external.pp.bbknn(AnnData) # use default params
    else:
        sc.pp.neighbors(AnnData, n_pcs=100, n_neighbors=30)
    return AnnData

# make graph
tdata = graph_pp(tdata, bbknn=False)
val = graph_pp(val, bbknn=False)
test = graph_pp(test, bbknn=False)

# encode condition
condition_encoder = {v:i for i,v in enumerate(tdata.obs['Condition'].unique())}
tdata.obs['ycondition'] = tdata.obs['Condition'].map(condition_encoder)
val.obs['ycondition'] = val.obs['Condition'].map(condition_encoder)
test.obs['ycondition'] = test.obs['Condition'].map(condition_encoder)


