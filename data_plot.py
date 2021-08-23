#coding:utf-8
import scprep
import pickle
import seaborn as sns
import scanpy as sc
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd





def plot_scatter(X,labels,label_set=None):
    # 降维可视化
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=500)
    # 降维
    low_dim_embs = tsne.fit_transform(X)
    adata = sc.AnnData(X=X,obs={"labels": labels})
    # PCA
    sc.tl.pca(adata, svd_solver='arpack')  # PCA分析
    if label_set is None:
        cmap_ctype = {v: sns.color_palette('bright', 8)[i] for i, v in enumerate(adata.obs['labels'].unique())}
    else:
        cmap_ctype= {v: sns.color_palette('bright', 8)[i] for i, v in enumerate(label_set)}
    scprep.plot.scatter2d(low_dim_embs,#adata.obsm["X_pca"],
                          c=adata.obs['labels'],
                          cmap=cmap_ctype,
                          ticks=None,
                          legend_loc=(1, 0),
                          label_prefix=None)

# def plot_line()

def plot_umap(X,labels,label_set=None):
    # 降维可视化
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=500)
    # 降维
    low_dim_embs = tsne.fit_transform(X)
    adata = sc.AnnData(X=X,obs={"labels": labels})
    # PCA
    sc.tl.pca(adata, svd_solver='arpack')  # PCA分析
    sc.pl.umap(adata, edges=True, color='ctype', palette=sns.color_palette('colorblind'), edges_width=0.01,
               save='_hbec_ctype.pdf')

def plot_heatmap(data):
    # data = np.random.randn(50, 20)
    sns.heatmap(data, xticklabels=2, yticklabels=False,cbar=False)
    plt.show()

def plot_feature(model,nheads=2,n_hidden=16,featnames=None):
    # featnames = load_pkl(os.path.join(pfp, 'hbec_feat_names.pkl'))
    n_top = 10
    g_idx = []
    n_heads = nheads
    n_hidden_units = n_hidden
    weight_per_headz = model.gat1.weight.detach().cpu().numpy().reshape(-1, n_heads, n_hidden_units, order='F')
    weight_per_headz=np.mean(weight_per_headz,axis=1,keepdims=True)
    n_heads=1
    # weight_per_headz = model.linear.weight.detach().cpu().numpy().reshape(-1, n_heads, n_hidden_units, order='F')
    for i in range(n_heads):
        # [g_idx.append(i) for i in np.max(np.abs(weight_per_headz[:, :, i]), axis=1).argsort()[-n_top:]]
        [g_idx.append(i) for i in np.max(np.abs(weight_per_headz[:, i, :]), axis=1).argsort()[-n_top:]]
    g_idx = np.unique(g_idx)
    w_top = np.zeros([g_idx.shape[0], n_hidden_units, n_heads])
    # w_top = np.zeros([g_idx.shape[0], n_heads,n_hidden_units])
    for i in range(n_heads):
        w_top[:, :, i] = weight_per_headz[g_idx, i,:]
    w_top = w_top.reshape((w_top.shape[0], -1))
    w_top = pd.DataFrame(w_top, index=[g for i, g in enumerate(featnames) if i in g_idx])

    # plt.tick_params(labelsize=24)
    sns.set(font_scale=1.8)
    g=sns.clustermap(
        w_top,
        pivot_kws=None,
        method='average',
        metric='euclidean',
        z_score=None,
        standard_scale=1,
        figsize=None,
        cbar_kws=None,
        row_cluster=True,
        col_cluster=True,
        row_linkage=None,
        col_linkage=None,
        row_colors=None,
        col_colors=None,
        mask=None,
        cmap='RdYlBu_r',
        yticklabels=True,
        xticklabels=False,
    )

    plt.show()


if __name__=="__main__":
    data = pickle.load(open("../datapkls/liao_test_200529.pkl", "rb"))

    hbec_feat_names = pickle.load(open("dataset/hbec_feat_names.pkl", "rb"))
    liao_feat_names = pickle.load(open("dataset/liao_feat_names.pkl", "rb"))
    X=data["X"]
    adata = sc.AnnData(X=X)
    # PCA
    sc.tl.pca(adata, svd_solver='arpack')  # PCA分析


    pca=adata.obsm["X_pca"]
    # labels=data["Condition"]
    labels=data["init_ctype"]
    ax = sc.pl.heatmap(adata, None, groupby='')

    # plot_scatter(X,labels)
    # plot_umap(X,labels)
    X=np.array(X)
    plot_heatmap(pca[:10,:50])