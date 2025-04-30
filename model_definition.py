import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F

class LightGCNLayer(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class LightGCNModel(nn.Module):
    def __init__(self, num_nodes, embedding_dim=64, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.convs = nn.ModuleList([LightGCNLayer() for _ in range(num_layers)])
        self.num_layers = num_layers
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, edge_index):
        x = self.embedding.weight
        all_emb = [x]

        for conv in self.convs:
            x = conv(x, edge_index)
            all_emb.append(x)

        out = torch.stack(all_emb, dim=0).mean(dim=0)
        return out
