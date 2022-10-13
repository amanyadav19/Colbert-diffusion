import torch
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv, self).__init__(aggr='sum') #  "Sum" aggregation.
        # inchannels is sum of 2 * dimensions of node features and edge features
        self.lin = torch.nn.Linear(3* in_channels,  3 * in_channels)
        self.act = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(3 * in_channels, out_channels)
        self.act2 = torch.nn.ReLU()
        self.update_lin = torch.nn.Linear(in_channels + out_channels, in_channels, bias=False)
        self.update_act = torch.nn.ReLU()
        
    def forward(self, x, edge_index, e):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]


        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, e=e)

    def message(self, x_i, x_j, e):

        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        # x_ij has shape [E, in_channels]

        x_i = self.lin(torch.cat([x_i, x_j, e], dim=1))
        x_i = self.act(x_i)
        x_i = self.lin2(x_i)
        x_i = self.act2(x_i)
        
        return x_i

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]

        new_embedding = torch.cat([aggr_out, x], dim=1)
        
        new_embedding = self.update_lin(new_embedding)
        new_embedding = self.update_act(new_embedding)

        return new_embedding