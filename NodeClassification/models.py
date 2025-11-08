import torch
import random
import math
import torch.nn.functional as F
import os.path as osp
import numpy as np
import torch_geometric.transforms as T
from GCNIIlayer import GCNIIdenseConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import Linear
from torch import Tensor
import torch_geometric
from torch_geometric.nn import GATConv, GCNConv, ChebConv, GCN2Conv, SGConv
from torch_geometric.nn import MessagePassing, APPNP
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp
from scipy.special import comb
from torch_sparse import SparseTensor
from torch_geometric.typing import (
    Adj,
    OptTensor,
)
from torch.nn import Linear, ModuleList, Module, Dropout, ReLU, GELU, Sequential
from einops import rearrange, repeat, einsum


class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[-1] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

class GCNII(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCNII, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(dataset.num_features, args.hidden))
        for _ in range(args.layer_num):
            self.convs.append(GCNIIdenseConv(args.hidden, args.hidden))
        self.dropout = args.dropout
        self.lamda = args.lamda
        self.alpha = args.alpha
        self.convs.append(torch.nn.Linear(args.hidden, dataset.num_classes))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters())+list(self.convs[-1:].parameters())

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        _hidden = []
        x = F.dropout(x, self.dropout ,training=self.training)
        x = F.relu(self.convs[0](x))
        _hidden.append(x)
        for i,con in enumerate(self.convs[1:-1]):
            x = F.dropout(x, self.dropout ,training=self.training)
            beta = math.log(self.lamda/(i+1)+1)
            x = F.relu(con(x, edge_index, self.alpha, _hidden[0], beta, edge_weight))
        x = F.dropout(x, self.dropout ,training=self.training)
        x = self.convs[-1](x)
        return F.log_softmax(x, dim=1)

class GPRGNN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)

        self.prop1 = GPR_prop(args.K, args.alpha, args.Init)

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        
class ActionNet(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = GCN_mamba_liner(args.d_inner, 2, with_bias=True)
        self.dropout = torch.nn.Dropout(0.6)
        self.act = torch.nn.ReLU()
        self.reset_parameters()
    def reset_parameters(self):
        self.net.reset_parameters()

    def forward(self, x) -> Tensor:

        x = self.dropout(self.net(x))

        return x
        
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        #output = adj @ support
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class SGC_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(SGC_Net, self).__init__()
        self.lin1 = MLP(dataset, args)
    
    def reset_parameters(self):
        self.lin1.reset_parameters()

    def forward(self, data):

        x = self.lin1(data)

        return x

class GCN_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCN_Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)
        self.conv = []
        self.dataset = args.dataset
        self.hidden_layer_num = 1
        for i in range(self.hidden_layer_num):
            self.conv.append(GCNConv(args.hidden, args.hidden).to(args.device))
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        if self.dataset in ['arxiv']:
            x, edge_index = data.x, data.adj_t
        else:
            x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.hidden_layer_num):
            x = F.relu(self.conv[i](x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)


class GAT_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GAT_Net, self).__init__()
        self.conv1 = GATConv(
            dataset.num_features,
            args.hidden,
            heads=args.heads,
            dropout=args.dropout)
        self.conv2 = GATConv(
            args.hidden * args.heads,
            dataset.num_classes,
            heads=args.output_heads,
            concat=False,
            dropout=args.dropout)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class MolConv(MessagePassing):
    def __init__(self, aggr='add'):
        super().__init__(aggr=aggr)  # 'add', 'mean' or 'max'

    def message(self, x_j: Tensor, edge_attr: OptTensor, edge_weight: OptTensor = None) -> Tensor:

        if edge_attr is None:
            if edge_weight is None:
                return x_j
            else:
                return edge_weight.view(-1, 1) * x_j
        else:
            if edge_weight is None:
                return x_j + edge_attr
            else:
                return edge_weight.view(-1, 1) * (x_j + edge_attr)

    def update(self, aggr_out: Tensor) -> Tensor:

        return aggr_out

class APPNP_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(APPNP_Net, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = APPNP(args.K, args.alpha)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)


class MLP(torch.nn.Module):
    def __init__(self, dataset, args):
        super(MLP, self).__init__()

        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.dropout =args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)
    
class GCN_mamba_liner(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, with_bias=False):
        super(GCN_mamba_liner, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = input @ self.weight
        # output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    
class GCN_mamba_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCN_mamba_Net, self).__init__()
        self.dropout = args.mamba_dropout
        self.args = args
        self.lin1 = GCN_mamba_liner(dataset.num_features, args.d_model, with_bias=args.bias)
        self.layer_num = args.layer_num
        self.mamba = GCN_mamba_block_New(args)
        self.norm_1 = RMSNorm(args.d_model)
        self.norm_2 = RMSNorm(args.d_model)
        self.lin2 = GCN_mamba_liner(args.d_model, dataset.num_classes, with_bias=args.bias)
        self.reset_parameters()
    def getmamba(self):
        return self.mamba
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x_input = data.x 
        if self.args.dataset in ['arxiv']:
            edge_index = data.adj_t
        else:
            edge_index = data.edge_index

        x = F.relu(self.lin1(x_input))

        x = F.dropout(x, p=self.dropout, training=self.training)

        output = self.mamba(self.norm_1(x), edge_index, data.adj_t, self.args.layer_num) + x

        y = self.norm_2(output)

        y = self.lin2(y)

        return F.log_softmax(y, dim=-1)
    
    def normalize_adj_tensor(self, adj):
        mx = adj
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx
    
class GCN_mamba_block(torch.nn.Module):
    def __init__(self, args):
        super(GCN_mamba_block, self).__init__()
        self.args = args
        self.in_proj = GCN_mamba_liner(args.d_model, args.d_inner * 2, with_bias=args.bias)
        self.x_proj = GCN_mamba_liner(args.d_inner, args.dt_rank + args.d_state * 2, with_bias=args.bias)
        self.dropout = args.dropout
        self.dt_proj = GCN_mamba_liner(args.dt_rank, args.d_inner, with_bias=args.bias)
        self.bns = torch.nn.ModuleList()

        for i in range(args.layer_num-1):
            self.bns.append(torch.nn.BatchNorm1d(args.d_inner))

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)

        self.A_log = torch.nn.Parameter(torch.log(A))
        self.D = torch.nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = GCN_mamba_liner(args.d_inner, args.d_model, with_bias=args.bias)

        self.in_act_net = ActionNet(args)
        self.out_act_net = ActionNet(args)
        self.reset_parameters()

    def reset_parameters(self):
        self.in_proj.reset_parameters()
        self.x_proj.reset_parameters()
        self.dt_proj.reset_parameters()
        self.out_proj.reset_parameters()
        self.in_act_net.reset_parameters()
        self.out_act_net.reset_parameters()


    def forward(self, x, edge_index, adj, layer_num):

        (b, d) = x.shape
        expanded_x = x.unsqueeze(1)
        x = expanded_x.expand(b, layer_num, d)
        (b, l, d) = x.shape
        
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)

        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = F.relu(x)

        y = self.ssm(x, self.args, edge_index, adj)
        
        y = y * F.relu(res)

        output = self.out_proj(y)

        return output[:,-1,:]
    
    def ssm(self, x, args, edge_index, adj):

        (d_in, n) = self.A_log.shape
        
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, edge_index, delta, A, B, C, D, args, adj)
        
        return y

    
    def selective_scan(self, u, edge_index, delta, A, B, C, D, args, adj):

        (b, l, d_in) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = [] 
        for i in range(args.layer_num):
            x = deltaA[:, i] * x + deltaB_u[:, i]

            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x.reshape(b, d_in * n)
            x = adj @ x
            x = x.reshape(b, d_in, n)

            in_logits = self.in_act_net(x=y)  # (N, 2)
            out_logits = self.out_act_net(x=y)  # (N, 2)
            temp = 0.01
            in_probs = F.gumbel_softmax(logits=in_logits, tau=temp, hard=True)
            out_probs = F.gumbel_softmax(logits=out_logits, tau=temp, hard=True)
            edge_weight = self.create_edge_weight(edge_index=edge_index,
                                                  keep_in_prob=in_probs[:, 0], keep_out_prob=out_probs[:, 0])
            adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight, sparse_sizes=(len(y), len(y)))

            adj = adj.set_diag()
            deg = adj.sum(dim=1).to(torch.float)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)

            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        y = y + u * D

        return y
    
    def create_edge_weight(self, edge_index, keep_in_prob: Tensor, keep_out_prob: Tensor) -> Tensor:
        u, v = edge_index
        edge_in_prob = keep_in_prob[v]
        edge_out_prob = keep_out_prob[u]
        return edge_in_prob * edge_out_prob
    
class RMSNorm(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model))

    def forward(self, x):

        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
