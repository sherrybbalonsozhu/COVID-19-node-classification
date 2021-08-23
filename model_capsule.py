#coding:utf-8
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import math
from torch.autograd import Variable
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch.nn.init import xavier_uniform_ as glorot
from torch.nn.init import zeros_ as zeros
import numpy as np

def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)

# full model (GATConv from PyTorch Geometric)
class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, node_dim=0, **kwargs):
        super(GATConv, self).__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(in_channels,
                                             heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)


    def forward(self, x, edge_index, size=None,
                return_attention_weights=False):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index,
                                           num_nodes=x.size(self.node_dim))

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        out = self.propagate(edge_index, size=size, x=x,
                             return_attention_weights=return_attention_weights)

        if return_attention_weights:
            alpha, self.alpha = self.alpha, None
            return out, alpha
        else:
            return out


    def message(self, edge_index_i, x_i, x_j, size_i,
                return_attention_weights):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        # alpha = softmax(alpha, edge_index_i, size_i)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)

        if return_attention_weights:
            self.alpha = alpha

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels,
                 num_iterations=3):
        super(CapsuleLayer, self).__init__()
        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules
        self.softmax=nn.Softmax(dim=2)


        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList([nn.Linear(in_channels,out_channels) for _ in range(num_capsules)])

    @staticmethod
    def squash(tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
            logits = Variable(torch.zeros(*priors.size()))
            if torch.cuda.is_available():
                logits = logits.cuda()
                priors=priors.cuda()
            for i in range(self.num_iterations):
                # probs = F.softmax(logits, dim=2)
                probs = self.softmax(logits)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            # outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = [capsule(x) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=1)
            outputs = self.squash(outputs)

        return outputs


class CapsuleModel(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_label):
        super(CapsuleModel,self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.num_label=num_label

        self.gene_embedding=nn.Parameter(torch.Tensor(size=[self.input_dim,self.hidden_dim]))
        torch.nn.init.kaiming_uniform_(self.gene_embedding, a=math.sqrt(5))

        self.linear=nn.Linear(input_dim,hidden_dim)
        self.primaryCaps=CapsuleLayer(num_capsules=8,num_route_nodes=-1,in_channels=self.hidden_dim,out_channels=self.hidden_dim)
        self.digitCaps=CapsuleLayer(num_capsules=num_label,num_route_nodes=self.hidden_dim,in_channels=8,out_channels=self.hidden_dim)



        self.classify=nn.Linear(hidden_dim,num_label)
        self.dropout=nn.Dropout(p=0.4)
        self.activation=nn.ReLU()
        self.tanh=nn.Tanh()

    def forward(self,data):
        x, edge_index, edge_feats,y = data.x, data.edge_index, data.edge_attr,data.y
        hidden=self.linear(x)
        x=self.primaryCaps(hidden)
        x=self.digitCaps(x).squeeze().transpose(0, 1)

        output = (x ** 2).sum(dim=-1) ** 0.5
        # classes = F.softmax(classes, dim=-1)


        loss_func=CrossEntropyLoss()
        # loss_func=LabelSmoothingCrossEntropy(epsilon=0.1)
        loss=loss_func(output,y)

        return output,loss

class CapsuleGNNModel(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_label,nHeads=4,alpha=0.2,dropout=0.5,label_weight=[15766,8231,110,9354,991,17698]):
        super(CapsuleGNNModel,self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.num_label=num_label
        self.nHeads=nHeads
        # label_weight=[np.exp(-w*0.001) for w in label_weight]
        # self.label_weight=np.array(label_weight)/sum(label_weight)
        # print(self.label_weight)

        self.gat1 = GATConv(input_dim, out_channels=hidden_dim,
                            heads=nHeads, concat=True, negative_slope=alpha,
                            dropout=dropout, bias=True)
        self.gat2 = GATConv(hidden_dim*nHeads, hidden_dim,
                            heads=nHeads, concat=True, negative_slope=alpha,
                            dropout=dropout, bias=True)

        self.linear=nn.Linear(input_dim,hidden_dim*nHeads)
        self.primaryCaps=CapsuleLayer(num_capsules=2,num_route_nodes=-1,in_channels=self.hidden_dim,out_channels=8)
        self.digitCaps=CapsuleLayer(num_capsules=num_label,num_route_nodes=2*(nHeads+nHeads),in_channels=8,out_channels=self.hidden_dim)
        # self.digitCaps=CapsuleLayer(num_capsules=num_label,num_route_nodes=(nHeads+1),in_channels=self.hidden_dim,out_channels=self.hidden_dim)



        self.classify=nn.Linear(hidden_dim,num_label)
        self.dropout=nn.Dropout(p=0.5)
        self.bn=nn.BatchNorm1d(num_features=self.input_dim)
        self.bn2=nn.BatchNorm1d(num_features=self.hidden_dim)
        self.activation=nn.ReLU()
        self.tanh=nn.Tanh()

    def forward(self,data):
        x, edge_index, edge_feats,y = data.x, data.edge_index, data.edge_attr,data.y

        x=self.bn(x)
        hidden = self.linear(x)
        # hidden = hidden.unsqueeze(1)
        hidden=torch.reshape(hidden,shape=[-1,self.nHeads,self.hidden_dim])

        x = self.gat1(x, edge_index)
        # x = F.elu(x)
        # x = self.gat2(x, edge_index)
        x=x.view(-1,self.nHeads,self.hidden_dim)



        x=torch.cat([x,hidden],dim=1)
        x=self.activation(x)
        # self.bn2(x)



        x=self.primaryCaps(x)



        x=self.digitCaps(x)
        x=x.squeeze().transpose(0, 1) #[batch,num_label,hidden_dim]

        features=torch.reshape(x,[-1,self.num_label*self.hidden_dim])#torch.sum(x,1,keepdim=False)


        output = (x ** 2).sum(dim=-1) ** 0.5
        # classes = F.softmax(classes, dim=-1)

        # features = F.softmax(output)

        loss_func=CrossEntropyLoss()
        # loss_func = LabelSmoothingCrossEntropy(epsilon=0.1)
        loss=loss_func(output,y)

        return output,loss,features
