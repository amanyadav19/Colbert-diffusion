import os.path as osp

import torch
from torch_geometric.data import Dataset, download_url
from torch_geometric.data import InMemoryDataset, HeteroData, Data
from torch_geometric.utils import remove_self_loops, add_self_loops
import os
from pathlib import Path


class MetaQADataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)


    @property
    def raw_file_names(self):
        return ['kb_entity_dict.txt', 'kb.txt', 'vanilla/qa_train.txt', 'vanilla/qa_dev.txt', 'vanilla/qa_test.txt']

    @property
    def processed_file_names(self):
        return ['meta_qa.pt', 'qs_train.pt', 'qs_dev.pt', 'qs_test.pt']

    def download(self):
        pass

    def _get_qa_from_file(self, file):
        query_subgraph_data = []
        with open(file) as f:
            for line in f.readlines():
                line = line.strip()
                if line == '':
                    continue
                line = line.split('\t')
                question = line[0].replace('[','').strip()
                answer_subgraphs = [(self.entities[answer], [], []) for answer in line[1].split('|')]

                query_subgraph_data.append([question, answer_subgraphs])
        return query_subgraph_data

    def process(self):
        f = open(self.raw_paths[0], "r")
        x_temp = []
        self.entities = {}
        for l in f:
            d = l.strip().split("\t")
            x_temp.append([int(d[0])])
            self.entities[d[1]] = int(d[0])
        self.x = torch.tensor(x_temp, dtype=torch.long)
        # print(x)

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
        
       
        self.edge_index = torch.tensor([edge_index_temp1,
                           edge_index_temp2], dtype=torch.long)
        self.edge_index, _= remove_self_loops(self.edge_index)
        self.edge_index, _ = add_self_loops(self.edge_index, num_nodes=self.x.size(0))
        self.edge_attr = torch.tensor(edge_attr_temp, dtype=torch.long)
        data = Data(x = self.x, edge_index = self.edge_index, edge_attr = self.edge_attr)

        torch.save(data, self.processed_paths[0])

        for i in range(1,4):
            torch.save(self._get_qa_from_file(self.raw_paths[i+1]), self.processed_paths[i])

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.processed_paths[0])
        return data
    
    def getQSdata(self, type="train"):
        typeToIndex = {
            "train" : 1,
            "dev" : 2,
            "test" : 3
        }

        if type in typeToIndex:
            return torch.load(self.processed_paths[typeToIndex[type]])
        else:
            raise Exception(f"Unknown type of data {type}, expected : 'train', 'test' or 'dev'")