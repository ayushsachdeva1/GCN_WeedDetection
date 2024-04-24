import torch.nn as nn
import torch.nn.functional as F
from layers.graph_convolution import GraphConvolution

class GCN(nn.Module):
    def __init__(self, feat_dim, num_class, nhid=16, dropout=0.5):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(feat_dim, nhid)
        self.gc2 = GraphConvolution(nhid, num_class)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)