# %%
from dataset import MetaQADataset, QuerySubgraphDataset, collate_batch
from question_tokenizer import QuestionTokenizer
import torch
from torch.utils.data import DataLoader
from visualize.utils import visualize

# %%
DATA_ROOT = '/mnt/infonas/data/jayeshs/data/MetaQA'
DEVICE = torch.device('cpu')
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# %%
metaqa_graph = MetaQADataset(DATA_ROOT).get()
len(metaqa_graph.entities)

# %%
qs_train_data = QuerySubgraphDataset(DATA_ROOT, 'train', metaqa_graph)
len(qs_train_data)

# %%
tokenizer = QuestionTokenizer()
qs_dataloader = DataLoader(qs_train_data, collate_fn=lambda x: collate_batch(x, tokenizer.encode, metaqa_graph, visualize=True))

# %%
for batch in qs_dataloader:
    q, p_sg, n_sg = batch
    visualize(p_sg[0], 'visualize/pos.html', heading=q[0])
    visualize(n_sg[0], 'visualize/neg.html', heading=q[0])
    break

# %%



