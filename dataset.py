import os.path as osp

import torch
import random
from torch_geometric.data import Dataset, download_url
from torch_geometric.data import InMemoryDataset, HeteroData, Data
from torch_geometric.utils import remove_self_loops, add_self_loops
import networkx as nx

from misc.networkx_utils import generate_random_paths
from visualize.utils import add_neighbours


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
        self.x = torch.tensor(x_temp, dtype=torch.long)
        # print(x)

        self.relations = {}
        i = 0
        
        f = open(self.raw_paths[1], "r")

        self.edge_node1 = []
        self.edge_node2 = []
        self.edge_type = []
        for l in f:
            d = l.strip().split("|")
            self.edge_node1.append(self.entities[d[0]])
            self.edge_node2.append(self.entities[d[2]])
            if d[1] not in self.relations.keys():
                self.relations[d[1]] = i
                i += 1
            self.edge_type.append(self.relations[d[1]])
        
       
        self.edge_index = torch.tensor([self.edge_node1,
                           self.edge_node2], dtype=torch.long)
        self.edge_index, _= remove_self_loops(self.edge_index)
        self.edge_index, _ = add_self_loops(self.edge_index, num_nodes=self.x.size(0))
        self.edge_attr = torch.tensor([[e] for e in self.edge_type], dtype=torch.long)
        data = Data(x = self.x, edge_index = self.edge_index, edge_attr = self.edge_attr,
                    entities=self.entities, relations = self.relations, edge_type = self.edge_type,
                    edge_node1 = self.edge_node1, edge_node2 = self.edge_node2)

        torch.save(data, self.processed_paths[0])

    def len(self):
        return len(self.processed_paths)

    def get(self, idx=0):
        data = torch.load(self.processed_paths[0])
        return data

class QuerySubgraphDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, root, type, kg):
        'Initialization'
        assert(type in ['train', 'test', 'dev'])
        self.root = root
        self.type = type
        self.graph = kg
        self.data = self._get_qa_from_file(osp.join(self.root, 'raw', 'vanilla', f'qa_{self.type}.txt'))

    def _get_qa_from_file(self, file):
        query_subgraph_data = []
        with open(file) as f:
            for line in f.readlines():
                line = line.strip()
                if line == '':
                    continue
                line = line.split('\t')
                topic_entity = self.graph.entities[line[0].split('[')[1].split(']')[0].strip()]
                question = line[0].replace('[','').replace(']','').strip()
                answer_subgraphs = [([self.graph.entities[answer]], [], []) for answer in line[1].split('|')]

                query_subgraph_data.append((question, topic_entity, answer_subgraphs))
        return query_subgraph_data

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return self.data[index]


def collate_batch(batch, tokenizer, graph, visualize=False):

    questions, entities,  answer_subg = list(zip(*batch))

    nx_graph = nx.DiGraph()
    inv_relations_map = {v: k for k, v in graph.relations.items()}
    _nodes = [(y, {"label" : x}) for x,y in graph.entities.items()]
    _edges = [(x, y, {"label" : inv_relations_map[etype]} ) for x,y,etype in zip(graph.edge_node1, graph.edge_node2, graph.edge_type)]
    nx_graph.add_nodes_from(_nodes)
    nx_graph.add_edges_from(_edges)

    subg_pairs = [(positive_sample(nx_graph, random.choice(answers), entity, return_nx=visualize) , negative_sample(nx_graph, answers, return_nx=visualize)) for entity, answers in  zip(entities, answer_subg)]
    pos_subg, neg_subg = list(zip(*subg_pairs))

    return questions if visualize else tokenizer(list(questions)), pos_subg, neg_subg


def positive_sample(nx_graph, answer_subg, entity, return_nx=False):
    "Answer subg is a subgraph with a singleton node"
    positive_subg = nx.shortest_path(nx_graph.to_undirected(), answer_subg[0][0], entity)

    if return_nx:
        subgraph = nx.subgraph(nx_graph, positive_subg).copy()
        subgraph = add_neighbours(subgraph, nx_graph)
        subgraph.nodes[answer_subg[0][0]]['type'] = 'answer'
        subgraph.nodes[entity]['type'] = 'entity'
        return subgraph
    else:
        edges = [(i,j) if nx_graph.has_edge(i,j) else (j,i) for i,j in zip(positive_subg[1:], positive_subg[:-1])]
        return (positive_subg, *list(zip(*edges)))

def negative_sample(nx_graph, answers, return_nx=False):
    "Answer is a list of subgraphs with a singleton node"

    answers = [answer[0][0] for answer in answers]
    ans_neighbourhood = [list(nx_graph.adj[answer].keys()) for answer in answers]
    ans_neighbourhood = list(set([node for sublist in ans_neighbourhood for node in sublist if node not in answers]))

    nx_graph_undir = nx_graph.to_undirected()

    nx_graph_undir.remove_edges_from(nx_graph.subgraph(ans_neighbourhood + answers).edges)
    nx_graph_undir.remove_nodes_from(answers)

    start_node = random.choice(ans_neighbourhood)
    negative_subg = next(generate_random_paths(nx_graph_undir, 1, start_node=start_node))
    negative_subg = [int(e) for e in negative_subg]

    if return_nx:
        subgraph = nx.subgraph(nx_graph, negative_subg).copy()
        subgraph = add_neighbours(subgraph, nx_graph)
        subgraph.nodes[start_node]['type'] = 'start_node'
        return subgraph
    else:
        edges = [(i,j) if nx_graph.has_edge(i,j) else (j,i) for i,j in zip(negative_subg[1:], negative_subg[:-1])]
        return (negative_subg, *list(zip(*edges)))
