import os.path as osp

import torch
from torch_geometric.data import Dataset, download_url
from torch_geometric.data import InMemoryDataset, HeteroData, Data
import os
from pathlib import Path


class MetaQADataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)


    @property
    def raw_file_names(self):
        return ['kb_entity_dict.txt', 'kb.txt']

    @property
    def processed_file_names(self):
        return ['meta_qa.pt']

    def download(self):
        pass

    def process(self):
        f = open(self.raw_paths[0], "r")
        x_temp = []
        self.entities = {}
        for l in f:
            d = l.strip().split("\t")
            x_temp.append([int(d[0])])
            self.entities[d[1]] = int(d[0])
        x = torch.tensor(x_temp, dtype=torch.long)
        print(x)

        self.relations = {}
        i = 0
        
        f = open(self.raw_paths[1], "r")

        edge_index_temp1 = []
        edge_index_temp2 = []
        edge_attr_temp = []
        for l in f:
            d = l.strip().split("|")
            edge_index_temp1.append(self.entities[d[0]])
            edge_index_temp2.append(self.entities[d[2]])
            if d[1] not in self.relations.keys():
                self.relations[d[1]] = i
                i += 1
            edge_attr_temp.append([self.relations[d[1]]])
        
       
        edge_index = torch.tensor([edge_index_temp1,
                           edge_index_temp2], dtype=torch.long)
        edge_attr = torch.tensor(edge_attr_temp, dtype=torch.long)
        # print(edge_index)
        # print(edge_attr)
        data = Data(x = x, edge_index = edge_index, edge_attr = edge_attr)

        torch.save(data, self.processed_paths[0])
    
    def len(self):
        return len(self.processed_file_names)

    def get(self):
        data = torch.load(self.processed_paths[0])
        return data
    