from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import torch
from message_passing import SAGEConv


# These are the dimensions of graph embeddings
embed_dim = 128

class Net(torch.nn.Module):
    def __init__(self, num_nodes, num_edges):
        super(Net, self).__init__()

        self.conv1 = SAGEConv(embed_dim, 128)
        # self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = SAGEConv(128, 128)
        # self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(128, 128)
        # self.pool3 = TopKPooling(128, ratio=0.8)
        self.item_embedding = torch.nn.Embedding(num_embeddings=num_nodes, embedding_dim=128)
        self.edge_embedding = torch.nn.Embedding(num_embeddings=num_edges, embedding_dim=128)
        self.lin1 = torch.nn.Linear(embed_dim, 128)
        # self.lin2 = torch.nn.Linear(128, 64)
        # self.lin3 = torch.nn.Linear(64, 1)
        # self.bn1 = torch.nn.BatchNorm1d(128)
        # self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(embed_dim, 128)
        self.act2 = torch.nn.ReLU()
        # self.act2 = torch.nn.ReLU()        
  
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.item_embedding(x)
        x = x.squeeze(1)
        # print(torch.range(0, edge_index[0].size(dim=0)).to(torch.device('cpu')).long())
        # print(self.edge_embedding)
        # print(edge_index[0].size(dim=0))
        e = self.edge_embedding(torch.arange(0, edge_index[0].size(dim=0)).to(torch.device('cpu')).long())        

        x = F.relu(self.conv1(x, edge_index, e))

        # x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        # x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.relu(self.conv2(x, edge_index, e))
     
        # x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        # x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, e))

        # x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        # x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # x = x1 + x2 + x3

        x = self.lin1(x)
        x = self.act1(x)
        # x = self.lin2(x)
        # x = self.act2(x)      
        x = F.dropout(x, p=0.5, training=self.training)
        e = self.lin2(e)
        e = self.act2(e)
        e = F.dropout(e, p=0.5, training=self.training)
        # x = torch.sigmoid(self.lin3(x)).squeeze(1)
        
        return x, e